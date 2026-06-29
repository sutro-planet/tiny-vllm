from collections import deque
from dataclasses import dataclass, field

from tiny_vllm.config import EngineConfig
from tiny_vllm.kv_cache import KVBlockAllocator
from tiny_vllm.model_runner import ExecutionBatch, MockModelRunner, ModelRunner
from tiny_vllm.request import GenerationOutput, GenerationRequest
from tiny_vllm.scheduler import Scheduler
from tiny_vllm.sequence import SequenceState


@dataclass
class EngineStats:
    submitted_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    generated_tokens: int = 0


@dataclass
class _ExecutionPlan:
    token_budget: int
    sequences: list[SequenceState] = field(default_factory=list)
    num_scheduled_tokens: list[int] = field(default_factory=list)
    admitted_sequences: list[SequenceState] = field(default_factory=list)
    failed_outputs: list[GenerationOutput] = field(default_factory=list)
    consumed_tokens: int = 0

    @property
    def remaining_tokens(self) -> int:
        return self.token_budget - self.consumed_tokens

    @property
    def batch(self) -> ExecutionBatch:
        return ExecutionBatch(
            sequences=self.sequences,
            num_scheduled_tokens=self.num_scheduled_tokens,
        )

    def schedule(self, sequence: SequenceState, num_scheduled_tokens: int) -> None:
        if num_scheduled_tokens <= 0:
            raise ValueError("num_scheduled_tokens must be positive")
        if num_scheduled_tokens > self.remaining_tokens:
            raise ValueError("scheduled tokens exceed remaining token budget")

        self.sequences.append(sequence)
        self.num_scheduled_tokens.append(num_scheduled_tokens)
        self.consumed_tokens += num_scheduled_tokens

    def admit(self, sequence: SequenceState, num_scheduled_tokens: int) -> None:
        self.schedule(sequence, num_scheduled_tokens)
        self.admitted_sequences.append(sequence)

    def fail(self, *, request_id: str, error: str) -> None:
        self.failed_outputs.append(
            GenerationOutput(
                request_id=request_id,
                text="",
                generated_tokens=0,
                error=error,
            )
        )


class _RequestAdmissionError(RuntimeError):
    def __init__(self, *, request_id: str, message: str) -> None:
        super().__init__(message)
        self.request_id = request_id
        self.message = message


class _RequestCapacityError(_RequestAdmissionError):
    pass


class Engine:
    def __init__(
        self,
        config: EngineConfig | None = None,
        *,
        model_runner: ModelRunner | None = None,
    ) -> None:
        self.config = config or EngineConfig()
        self.scheduler = Scheduler(max_batch_size=self.config.max_batch_size)
        self.kv_cache = KVBlockAllocator(
            num_blocks=self.config.max_num_blocks,
            block_size=self.config.block_size,
        )
        self.model_runner = model_runner or MockModelRunner()
        self.stats = EngineStats()
        self._active_sequences: dict[str, SequenceState] = {}
        self._waiting_sequences: deque[SequenceState] = deque()

    def submit(self, request: GenerationRequest) -> None:
        self.scheduler.add(request)
        self.stats.submitted_requests += 1

    def run_until_complete(self) -> list[GenerationOutput]:
        outputs: list[GenerationOutput] = []
        while (
            self.scheduler.pending_count
            or self._active_sequences
            or self._waiting_sequences
        ):
            outputs.extend(self.run_step())
        return outputs

    def run_step(self) -> list[GenerationOutput]:
        outputs: list[GenerationOutput] = []
        generated_tokens = 0
        execution_plan: _ExecutionPlan | None = None

        try:
            execution_plan = self._build_execution_plan()
            for sequence in execution_plan.admitted_sequences:
                self._active_sequences[sequence.request_id] = sequence

            execution_batch = execution_plan.batch
            if execution_batch.sequences:
                runner_output = self.model_runner.execute(execution_batch)
                generated_tokens += self._append_sampled_tokens(
                    execution_batch,
                    runner_output.sampled_token_ids,
                )

            outputs.extend(execution_plan.failed_outputs)
            outputs.extend(self._complete_finished_sequences())
        except Exception:
            self._release_sequences(self._owned_sequences(execution_plan))
            raise

        self.stats.failed_requests += len(execution_plan.failed_outputs)
        self.stats.generated_tokens += generated_tokens

        return outputs

    def _build_execution_plan(self) -> _ExecutionPlan:
        plan = _ExecutionPlan(token_budget=self.config.max_num_scheduled_tokens)

        can_schedule_more = self._schedule_running(plan)
        if can_schedule_more:
            self._schedule_waiting(plan)

        return plan

    def _schedule_running(self, plan: _ExecutionPlan) -> bool:
        for sequence in list(self._active_sequences.values()):
            if plan.remaining_tokens == 0:
                return False
            if sequence.is_complete or sequence.num_new_tokens == 0:
                continue

            scheduled_tokens = self._scheduled_tokens_for(sequence, plan)
            try:
                protected_request_ids = self._protected_request_ids(plan, sequence)
                preempted = self._ensure_slots_with_preemption(
                    sequence,
                    num_scheduled_tokens=scheduled_tokens,
                    protected_request_ids=protected_request_ids,
                )
            except RuntimeError:
                if plan.sequences:
                    return False
                raise

            plan.schedule(sequence, scheduled_tokens)
            if preempted:
                return False

        return True

    def _schedule_waiting(self, plan: _ExecutionPlan) -> None:
        while plan.remaining_tokens > 0:
            waiting_sequence = self._pop_waiting_sequence()
            request: GenerationRequest | None = None
            if waiting_sequence is None:
                if not self._has_capacity_for_new_sequence(plan):
                    break

                request = self.scheduler.pop_next()
                if request is None:
                    break

                try:
                    sequence = self._create_sequence(request)
                except _RequestAdmissionError as exc:
                    self._fail_request(plan, request_id=exc.request_id, error=exc.message)
                    continue
            else:
                sequence = waiting_sequence

            scheduled_tokens = self._scheduled_tokens_for(sequence, plan)
            try:
                protected_request_ids = self._protected_request_ids(plan, sequence)
                preempted = self._ensure_slots_with_preemption(
                    sequence,
                    num_scheduled_tokens=scheduled_tokens,
                    protected_request_ids=protected_request_ids,
                )
            except RuntimeError as exc:
                if waiting_sequence is not None:
                    self._waiting_sequences.appendleft(waiting_sequence)
                    if plan.sequences:
                        break
                    raise

                assert request is not None
                self._handle_new_sequence_slot_failure(
                    request,
                    exc,
                    has_scheduled_work=bool(plan.sequences),
                )
                break

            plan.admit(sequence, scheduled_tokens)
            if preempted:
                break

    def _pop_waiting_sequence(self) -> SequenceState | None:
        if not self._waiting_sequences:
            return None
        return self._waiting_sequences.popleft()

    def _has_capacity_for_new_sequence(self, plan: _ExecutionPlan) -> bool:
        return len(self._active_sequences) + len(plan.admitted_sequences) < (
            self.config.max_batch_size
        )

    @staticmethod
    def _scheduled_tokens_for(sequence: SequenceState, plan: _ExecutionPlan) -> int:
        scheduled_tokens = min(sequence.num_new_tokens, plan.remaining_tokens)
        if scheduled_tokens <= 0:
            raise RuntimeError(f"sequence has no tokens to schedule: {sequence.request_id}")
        return scheduled_tokens

    @staticmethod
    def _protected_request_ids(
        plan: _ExecutionPlan, sequence: SequenceState
    ) -> set[str]:
        protected_request_ids = {scheduled.request_id for scheduled in plan.sequences}
        protected_request_ids.add(sequence.request_id)
        return protected_request_ids

    def _ensure_slots_with_preemption(
        self,
        sequence: SequenceState,
        *,
        num_scheduled_tokens: int,
        protected_request_ids: set[str],
    ) -> bool:
        preempted = False
        released_victim_request_ids: set[str] = set()
        while True:
            try:
                self._ensure_sequence_slots(
                    sequence,
                    num_scheduled_tokens=num_scheduled_tokens,
                )
                return preempted
            except RuntimeError:
                victim = self._select_preemption_victim(
                    protected_request_ids=protected_request_ids,
                    released_victim_request_ids=released_victim_request_ids,
                )
                if victim is None:
                    raise

                self._preempt_sequence(victim)
                released_victim_request_ids.add(victim.request_id)
                preempted = True

    def _select_preemption_victim(
        self,
        *,
        protected_request_ids: set[str],
        released_victim_request_ids: set[str],
    ) -> SequenceState | None:
        for victim in reversed(list(self._active_sequences.values())):
            if (
                victim.request_id not in protected_request_ids
                and victim.request_id not in released_victim_request_ids
            ):
                return victim
        return None

    def _preempt_sequence(self, sequence: SequenceState) -> None:
        self._active_sequences.pop(sequence.request_id, None)
        self.kv_cache.release(sequence.request_id)
        sequence.reset_computed_tokens()
        self._waiting_sequences.appendleft(sequence)

    def _create_sequence(self, request: GenerationRequest) -> SequenceState:
        try:
            prompt_token_ids = self.model_runner.encode_prompt(request.prompt)
        except Exception as exc:
            raise _RequestAdmissionError(
                request_id=request.request_id,
                message=str(exc),
            ) from exc

        if not prompt_token_ids:
            raise _RequestAdmissionError(
                request_id=request.request_id,
                message="encoded prompt is empty",
            )

        max_new_tokens = request.max_new_tokens or self.config.max_new_tokens
        sequence = SequenceState(
            request_id=request.request_id,
            prompt=request.prompt,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
        )
        self._validate_sequence_kv_capacity(sequence)
        return sequence

    def _validate_sequence_kv_capacity(self, sequence: SequenceState) -> None:
        required_tokens = max(
            1,
            len(sequence.prompt_token_ids) + sequence.max_new_tokens - 1,
        )
        capacity_tokens = self.kv_cache.num_blocks * self.kv_cache.block_size
        if required_tokens <= capacity_tokens:
            return

        raise _RequestCapacityError(
            request_id=sequence.request_id,
            message=(
                f"request {sequence.request_id} requires {required_tokens} KV token "
                f"slots but KV cache capacity is {capacity_tokens}"
            ),
        )

    def _fail_request(
        self,
        plan: _ExecutionPlan,
        *,
        request_id: str,
        error: str,
    ) -> None:
        self.scheduler.finish(request_id)
        plan.fail(request_id=request_id, error=error)

    def _ensure_sequence_slots(
        self, sequence: SequenceState, *, num_scheduled_tokens: int
    ) -> None:
        self.kv_cache.ensure_slots(
            sequence.request_id,
            sequence.num_computed_tokens + num_scheduled_tokens,
        )

    def _handle_new_sequence_slot_failure(
        self,
        request: GenerationRequest,
        error: RuntimeError,
        *,
        has_scheduled_work: bool,
    ) -> None:
        if has_scheduled_work:
            self.scheduler.requeue_front([request])
            return

        self.scheduler.finish(request.request_id)
        raise error

    def _append_sampled_tokens(
        self,
        batch: ExecutionBatch,
        token_ids: list[int],
    ) -> int:
        self._validate_token_count(batch.sequences, token_ids)
        generated_tokens = 0
        for sequence, num_scheduled_tokens, token_id in zip(
            batch.sequences, batch.num_scheduled_tokens, token_ids, strict=True
        ):
            should_sample = sequence.should_sample_after(
                num_scheduled_tokens=num_scheduled_tokens
            )
            sequence.advance_computed_tokens(num_scheduled_tokens)
            if should_sample:
                sequence.append_sampled_token(token_id)
                generated_tokens += 1
        return generated_tokens

    @staticmethod
    def _validate_token_count(sequences: list[SequenceState], token_ids: list[int]) -> None:
        if len(token_ids) != len(sequences):
            raise RuntimeError(
                f"model runner returned {len(token_ids)} tokens for {len(sequences)} sequences"
            )

    def _complete_finished_sequences(self) -> list[GenerationOutput]:
        outputs: list[GenerationOutput] = []
        for sequence in list(self._active_sequences.values()):
            if not sequence.is_complete:
                continue

            outputs.append(
                GenerationOutput(
                    request_id=sequence.request_id,
                    text=self._format_output_text(sequence),
                    generated_tokens=len(sequence.generated_token_ids),
                )
            )
            self.stats.completed_requests += 1
            self.kv_cache.release(sequence.request_id)
            self.scheduler.finish(sequence.request_id)
            del self._active_sequences[sequence.request_id]

        return outputs

    def _owned_sequences(
        self, execution_plan: _ExecutionPlan | None = None
    ) -> list[SequenceState]:
        sequences = [
            *self._active_sequences.values(),
            *self._waiting_sequences,
        ]
        if execution_plan is not None:
            sequences.extend(execution_plan.admitted_sequences)
        return sequences

    def _release_sequences(self, sequences: list[SequenceState]) -> None:
        released_request_ids: set[str] = set()
        for sequence in sequences:
            if sequence.request_id in released_request_ids:
                continue
            released_request_ids.add(sequence.request_id)
            self.kv_cache.release(sequence.request_id)
            self.scheduler.finish(sequence.request_id)
            self._active_sequences.pop(sequence.request_id, None)

        self._waiting_sequences = deque(
            sequence
            for sequence in self._waiting_sequences
            if sequence.request_id not in released_request_ids
        )

    def _format_output_text(self, sequence: SequenceState) -> str:
        generated_text = self.model_runner.detokenize(sequence.generated_token_ids)
        if sequence.prompt and generated_text:
            return f"{sequence.prompt} {generated_text}"
        return generated_text or sequence.prompt

from dataclasses import dataclass

from tiny_vllm.config import EngineConfig
from tiny_vllm.kv_cache import KVBlockAllocator
from tiny_vllm.model_runner import ExecutionBatch, MockModelRunner, ModelRunner
from tiny_vllm.request import GenerationOutput, GenerationRequest
from tiny_vllm.scheduler import Scheduler
from tiny_vllm.sequence import SequencePhase, SequenceState


@dataclass
class EngineStats:
    submitted_requests: int = 0
    completed_requests: int = 0
    generated_tokens: int = 0


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

    def submit(self, request: GenerationRequest) -> None:
        self.scheduler.add(request)
        self.stats.submitted_requests += 1

    def run_until_complete(self) -> list[GenerationOutput]:
        outputs: list[GenerationOutput] = []
        while self.scheduler.pending_count or self._active_sequences:
            outputs.extend(self.run_step())
        return outputs

    def run_step(self) -> list[GenerationOutput]:
        outputs: list[GenerationOutput] = []
        generated_tokens = 0
        decode_batch = [
            sequence
            for sequence in self._active_sequences.values()
            if sequence.phase is SequencePhase.DECODING
        ]

        try:
            prefill_batch = self._admit_prefill_batch()
            execution_batch = self._build_execution_batch(decode_batch, prefill_batch)
            if execution_batch.sequences:
                runner_output = self.model_runner.execute(execution_batch)
                generated_tokens += self._append_sampled_tokens(
                    execution_batch,
                    runner_output.sampled_token_ids,
                )

            outputs.extend(self._complete_finished_sequences())
        except Exception:
            self._release_sequences(list(self._active_sequences.values()))
            raise

        self.stats.generated_tokens += generated_tokens

        return outputs

    def _admit_prefill_batch(self) -> list[SequenceState]:
        admitted: list[SequenceState] = []
        while len(self._active_sequences) + len(admitted) < self.config.max_batch_size:
            request = self.scheduler.pop_next()
            if request is None:
                break

            try:
                prompt_token_ids = self.model_runner.encode_prompt(request.prompt)
            except Exception:
                self.scheduler.finish(request.request_id)
                raise

            max_new_tokens = request.max_new_tokens or self.config.max_new_tokens
            sequence = SequenceState(
                request_id=request.request_id,
                prompt=request.prompt,
                prompt_token_ids=prompt_token_ids,
                max_new_tokens=max_new_tokens,
            )

            try:
                self.kv_cache.allocate(sequence.request_id, sequence.token_budget)
            except Exception:
                if self._active_sequences or admitted:
                    self.scheduler.requeue_front([request])
                    break

                self.scheduler.finish(request.request_id)
                raise

            admitted.append(sequence)

        for sequence in admitted:
            self._active_sequences[sequence.request_id] = sequence

        return admitted

    def _build_execution_batch(
        self,
        decode_batch: list[SequenceState],
        prefill_batch: list[SequenceState],
    ) -> ExecutionBatch:
        sequences = [*decode_batch, *prefill_batch]
        num_scheduled_tokens = [
            *[1 for _ in decode_batch],
            *[max(1, len(sequence.prompt_token_ids)) for sequence in prefill_batch],
        ]
        return ExecutionBatch(
            sequences=sequences,
            num_scheduled_tokens=num_scheduled_tokens,
        )

    def _append_sampled_tokens(
        self,
        batch: ExecutionBatch,
        token_ids: list[int],
    ) -> int:
        self._validate_token_count(batch.sequences, token_ids)
        for sequence, token_id in zip(batch.sequences, token_ids, strict=True):
            if sequence.phase is SequencePhase.WAITING_PREFILL:
                sequence.append_prefill_token(token_id)
            elif sequence.phase is SequencePhase.DECODING:
                sequence.append_decode_token(token_id)
            else:
                raise RuntimeError(
                    f"cannot append token to finished sequence: {sequence.request_id}"
                )
        return len(token_ids)

    @staticmethod
    def _validate_token_count(sequences: list[SequenceState], token_ids: list[int]) -> None:
        if len(token_ids) != len(sequences):
            raise RuntimeError(
                f"model runner returned {len(token_ids)} tokens for {len(sequences)} sequences"
            )

    def _complete_finished_sequences(self) -> list[GenerationOutput]:
        outputs: list[GenerationOutput] = []
        for sequence in list(self._active_sequences.values()):
            if sequence.phase is not SequencePhase.FINISHED:
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

    def _release_sequences(self, sequences: list[SequenceState]) -> None:
        for sequence in sequences:
            self.kv_cache.release(sequence.request_id)
            self.scheduler.finish(sequence.request_id)
            self._active_sequences.pop(sequence.request_id, None)

    def _format_output_text(self, sequence: SequenceState) -> str:
        generated_text = self.model_runner.detokenize(sequence.generated_token_ids)
        if sequence.prompt and generated_text:
            return f"{sequence.prompt} {generated_text}"
        return generated_text or sequence.prompt

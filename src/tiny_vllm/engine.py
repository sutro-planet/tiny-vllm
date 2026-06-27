from dataclasses import dataclass

from tiny_vllm.config import EngineConfig
from tiny_vllm.kv_cache import KVBlockAllocator
from tiny_vllm.model_runner import MockModelRunner, ModelRunner
from tiny_vllm.request import GenerationOutput, GenerationRequest
from tiny_vllm.scheduler import Scheduler


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

    def submit(self, request: GenerationRequest) -> None:
        self.scheduler.add(request)
        self.stats.submitted_requests += 1

    def run_until_complete(self) -> list[GenerationOutput]:
        outputs: list[GenerationOutput] = []
        while self.scheduler.pending_count:
            outputs.extend(self.run_step())
        return outputs

    def run_step(self) -> list[GenerationOutput]:
        batch = self.scheduler.next_batch()
        outputs: list[GenerationOutput] = []
        scheduled: list[tuple[GenerationRequest, int, int]] = []

        for request in batch:
            max_new_tokens = request.max_new_tokens or self.config.max_new_tokens
            token_budget = self._count_prompt_tokens(request.prompt) + max_new_tokens
            scheduled.append((request, max_new_tokens, token_budget))

        successful_scheduled: list[tuple[GenerationRequest, int, int]] = []
        for index, (request, max_new_tokens, token_budget) in enumerate(scheduled):
            try:
                self.kv_cache.allocate(request.request_id, token_budget)
                successful_scheduled.append((request, max_new_tokens, token_budget))
            except Exception:
                # If the first request cannot allocate, it cannot fit even in an
                # empty step; finish it so run_until_complete does not retry forever.
                if index == 0:
                    self.scheduler.finish(request.request_id)
                    self.scheduler.requeue_front(item[0] for item in scheduled[1:])
                    raise

                # Later failures mean the current step is full, not that the request
                # is invalid. Run the already-allocated prefix and retry the tail.
                self.scheduler.requeue_front(item[0] for item in scheduled[index:])
                break

        try:
            for request, max_new_tokens, _ in successful_scheduled:
                output = self.model_runner.generate(request, max_new_tokens)

                self.stats.completed_requests += 1
                self.stats.generated_tokens += output.generated_tokens
                outputs.append(output)
        finally:
            for request, _, _ in successful_scheduled:
                self.kv_cache.release(request.request_id)
                self.scheduler.finish(request.request_id)

        return outputs

    @staticmethod
    def _count_prompt_tokens(prompt: str) -> int:
        return max(1, len(prompt.split()))

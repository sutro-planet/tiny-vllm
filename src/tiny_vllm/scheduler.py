from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from tiny_vllm.request import GenerationRequest


@dataclass
class Scheduler:
    max_batch_size: int
    _queue: deque[GenerationRequest] = field(default_factory=deque, init=False)
    _active_request_ids: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    def add(self, request: GenerationRequest) -> None:
        if request.request_id in self._active_request_ids:
            raise ValueError(f"duplicate request_id: {request.request_id}")

        self._active_request_ids.add(request.request_id)
        self._queue.append(request)

    def next_batch(self) -> list[GenerationRequest]:
        batch: list[GenerationRequest] = []
        while self._queue and len(batch) < self.max_batch_size:
            batch.append(self._queue.popleft())
        return batch

    def requeue_front(self, requests: Iterable[GenerationRequest]) -> None:
        for request in reversed(list(requests)):
            self._queue.appendleft(request)

    def finish(self, request_id: str) -> None:
        self._active_request_ids.discard(request_id)

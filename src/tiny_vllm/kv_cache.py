from dataclasses import dataclass, field
from math import ceil


@dataclass(frozen=True)
class KVBlock:
    index: int
    block_size: int


@dataclass
class KVBlockAllocator:
    num_blocks: int
    block_size: int
    _free_blocks: list[KVBlock] = field(default_factory=list, init=False)
    _allocations: dict[str, list[KVBlock]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")

        self._free_blocks = [
            KVBlock(index=index, block_size=self.block_size) for index in range(self.num_blocks)
        ]

    @property
    def available_blocks(self) -> int:
        return len(self._free_blocks)

    def allocate(self, request_id: str, num_tokens: int) -> list[KVBlock]:
        if request_id in self._allocations:
            raise ValueError(f"request already has KV blocks: {request_id}")
        return self.ensure_slots(request_id, num_tokens)

    def ensure_slots(self, request_id: str, num_tokens: int) -> list[KVBlock]:
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")

        needed_blocks = ceil(num_tokens / self.block_size)
        current_blocks = self._allocations.get(request_id, [])
        additional_blocks = needed_blocks - len(current_blocks)
        if additional_blocks <= 0:
            return []
        if additional_blocks > len(self._free_blocks):
            raise RuntimeError("KV cache capacity exhausted")

        blocks = self._free_blocks[:additional_blocks]
        self._free_blocks = self._free_blocks[additional_blocks:]
        current_blocks.extend(blocks)
        self._allocations[request_id] = current_blocks
        return blocks

    def release(self, request_id: str) -> None:
        blocks = self._allocations.pop(request_id, [])
        self._free_blocks.extend(blocks)

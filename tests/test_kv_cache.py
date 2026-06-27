from tiny_vllm.kv_cache import KVBlockAllocator


def test_allocator_reserves_and_releases_blocks() -> None:
    allocator = KVBlockAllocator(num_blocks=3, block_size=16)

    first = allocator.allocate("req-a", num_tokens=17)
    second = allocator.allocate("req-b", num_tokens=1)

    assert [block.index for block in first] == [0, 1]
    assert [block.index for block in second] == [2]
    assert allocator.available_blocks == 0

    allocator.release("req-a")

    assert allocator.available_blocks == 2
    assert [block.index for block in allocator.allocate("req-c", 16)] == [0]


def test_allocator_fails_when_capacity_is_exhausted() -> None:
    allocator = KVBlockAllocator(num_blocks=1, block_size=16)

    allocator.allocate("req-a", num_tokens=16)

    try:
        allocator.allocate("req-b", num_tokens=1)
    except RuntimeError as exc:
        assert "KV cache capacity" in str(exc)
    else:
        raise AssertionError("expected allocation to fail")


def test_allocator_reuses_still_free_blocks_before_released_blocks() -> None:
    allocator = KVBlockAllocator(num_blocks=4, block_size=16)

    allocator.allocate("req-a", num_tokens=17)
    allocator.allocate("req-b", num_tokens=1)

    allocator.release("req-a")

    assert [block.index for block in allocator.allocate("req-c", num_tokens=1)] == [3]

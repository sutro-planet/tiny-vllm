from tiny_vllm.request import GenerationRequest
from tiny_vllm.scheduler import Scheduler


def test_scheduler_dispatches_fifo_batch_and_keeps_remaining_requests() -> None:
    scheduler = Scheduler(max_batch_size=2)
    first = GenerationRequest(request_id="req-a", prompt="alpha")
    second = GenerationRequest(request_id="req-b", prompt="beta")
    third = GenerationRequest(request_id="req-c", prompt="gamma")

    scheduler.add(first)
    scheduler.add(second)
    scheduler.add(third)

    batch = scheduler.next_batch()

    assert batch == [first, second]
    assert scheduler.pending_count == 1
    assert scheduler.next_batch() == [third]


def test_scheduler_rejects_duplicate_request_ids() -> None:
    scheduler = Scheduler(max_batch_size=2)

    scheduler.add(GenerationRequest(request_id="req-a", prompt="alpha"))

    try:
        scheduler.add(GenerationRequest(request_id="req-a", prompt="again"))
    except ValueError as exc:
        assert "duplicate request_id" in str(exc)
    else:
        raise AssertionError("expected duplicate request rejection")

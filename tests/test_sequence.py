from tiny_vllm.sequence import SequencePhase, SequenceState


def assert_phase(sequence: SequenceState, phase: SequencePhase) -> None:
    assert sequence.phase is phase


def test_sequence_tracks_prefill_decode_and_completion() -> None:
    sequence = SequenceState(
        request_id="req-a",
        prompt="hello",
        prompt_token_ids=[10, 11],
        max_new_tokens=2,
    )

    assert_phase(sequence, SequencePhase.WAITING_PREFILL)
    assert sequence.token_budget == 4
    assert sequence.all_token_ids == [10, 11]
    assert sequence.is_complete is False

    sequence.append_prefill_token(12)

    assert_phase(sequence, SequencePhase.DECODING)
    assert sequence.generated_token_ids == [12]
    assert sequence.latest_token_id == 12
    assert sequence.all_token_ids == [10, 11, 12]
    assert sequence.is_complete is False

    sequence.append_decode_token(13)

    assert_phase(sequence, SequencePhase.FINISHED)
    assert sequence.generated_token_ids == [12, 13]
    assert sequence.latest_token_id == 13
    assert sequence.is_complete is True


def test_sequence_with_one_token_budget_finishes_after_prefill() -> None:
    sequence = SequenceState(
        request_id="req-a",
        prompt="hello",
        prompt_token_ids=[10],
        max_new_tokens=1,
    )

    sequence.append_prefill_token(11)

    assert_phase(sequence, SequencePhase.FINISHED)
    assert sequence.generated_token_ids == [11]
    assert sequence.is_complete is True


def test_latest_token_id_fails_clearly_when_sequence_has_no_tokens() -> None:
    sequence = SequenceState(
        request_id="req-a",
        prompt="",
        prompt_token_ids=[],
        max_new_tokens=1,
    )

    try:
        _ = sequence.latest_token_id
    except ValueError as exc:
        assert "sequence has no tokens" in str(exc)
    else:
        raise AssertionError("expected empty sequence to fail clearly")

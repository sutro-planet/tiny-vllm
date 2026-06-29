from tiny_vllm.sequence import SequenceState


def test_sequence_tracks_token_progress_and_completion() -> None:
    sequence = SequenceState(
        request_id="req-a",
        prompt="hello",
        prompt_token_ids=[10, 11],
        max_new_tokens=2,
    )

    assert sequence.token_budget == 4
    assert sequence.all_token_ids == [10, 11]
    assert sequence.num_tokens == 2
    assert sequence.num_new_tokens == 2
    assert sequence.num_computed_tokens == 0
    assert sequence.is_complete is False

    assert sequence.should_sample_after(num_scheduled_tokens=1) is False
    sequence.advance_computed_tokens(2)

    assert sequence.num_computed_tokens == 2
    assert sequence.num_new_tokens == 0

    sequence.append_sampled_token(12)

    assert sequence.generated_token_ids == [12]
    assert sequence.latest_token_id == 12
    assert sequence.all_token_ids == [10, 11, 12]
    assert sequence.num_tokens == 3
    assert sequence.num_new_tokens == 1
    assert sequence.is_complete is False

    sequence.advance_computed_tokens(1)
    sequence.append_sampled_token(13)

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

    sequence.advance_computed_tokens(1)
    sequence.append_sampled_token(11)

    assert sequence.generated_token_ids == [11]
    assert sequence.is_complete is True


def test_sequence_resets_computed_tokens_for_recompute() -> None:
    sequence = SequenceState(
        request_id="req-a",
        prompt="hello",
        prompt_token_ids=[10],
        max_new_tokens=3,
        generated_token_ids=[11, 12],
        num_computed_tokens=2,
    )

    sequence.reset_computed_tokens()

    assert sequence.num_computed_tokens == 0
    assert sequence.num_new_tokens == 3
    assert sequence.generated_token_ids == [11, 12]

    sequence.advance_computed_tokens(3)
    sequence.append_sampled_token(13)

    assert sequence.generated_token_ids == [11, 12, 13]
    assert sequence.is_complete is True


def test_sequence_rejects_sampling_before_computing_current_tokens() -> None:
    sequence = SequenceState(
        request_id="req-a",
        prompt="hello",
        prompt_token_ids=[10, 11],
        max_new_tokens=1,
    )

    try:
        sequence.append_sampled_token(12)
    except RuntimeError as exc:
        assert "cannot sample before computed tokens reach current tokens" in str(exc)
    else:
        raise AssertionError("expected early sampling to fail")


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


def test_sequence_rejects_invalid_computed_token_advancement() -> None:
    sequence = SequenceState(
        request_id="req-a",
        prompt="hello",
        prompt_token_ids=[10],
        max_new_tokens=1,
    )

    try:
        sequence.advance_computed_tokens(0)
    except ValueError as exc:
        assert "num_tokens must be positive" in str(exc)
    else:
        raise AssertionError("expected zero-token advancement to fail")


def test_sequence_rejects_computed_token_advancement_beyond_known_tokens() -> None:
    sequence = SequenceState(
        request_id="req-a",
        prompt="hello",
        prompt_token_ids=[10],
        max_new_tokens=1,
    )

    try:
        sequence.advance_computed_tokens(2)
    except ValueError as exc:
        assert "cannot compute beyond known tokens" in str(exc)
    else:
        raise AssertionError("expected over-advancement to fail")

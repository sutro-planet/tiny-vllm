from tiny_vllm.config import EngineConfig


def test_engine_config_has_small_deterministic_defaults() -> None:
    config = EngineConfig()

    assert config.model_name == "mock-gemma"
    assert config.max_batch_size == 4
    assert config.max_num_scheduled_tokens == 16
    assert config.max_num_blocks == 16
    assert config.block_size == 16
    assert config.max_new_tokens == 8


def test_engine_config_rejects_non_positive_limits() -> None:
    try:
        EngineConfig(max_batch_size=0)
    except ValueError as exc:
        assert "max_batch_size" in str(exc)
    else:
        raise AssertionError("expected max_batch_size validation to fail")

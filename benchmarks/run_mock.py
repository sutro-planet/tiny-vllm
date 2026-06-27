from tiny_vllm import Engine, EngineConfig, GenerationRequest


def main() -> None:
    engine = Engine(EngineConfig(max_batch_size=4))
    for index in range(4):
        engine.submit(
            GenerationRequest(
                request_id=f"bench-{index}",
                prompt="mock prompt",
                max_new_tokens=4,
            )
        )

    outputs = engine.run_until_complete()
    total_tokens = sum(output.generated_tokens for output in outputs)
    print(f"completed_requests={len(outputs)} generated_tokens={total_tokens}")


if __name__ == "__main__":
    main()

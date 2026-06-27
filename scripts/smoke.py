from tiny_vllm import Engine, GenerationRequest


def main() -> None:
    engine = Engine()
    engine.submit(
        GenerationRequest(
            request_id="smoke-0",
            prompt="hello tiny vllm",
            max_new_tokens=3,
        )
    )
    output = engine.run_until_complete()[0]
    print(output.text)


if __name__ == "__main__":
    main()

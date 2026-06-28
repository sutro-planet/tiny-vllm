import argparse

from tiny_vllm import Engine, EngineConfig, GenerationRequest, TransformersModelRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real causal LM through tiny-vLLM.")
    parser.add_argument("--model", required=True, help="Hugging Face model name or local path.")
    parser.add_argument("--prompt", default="Hello", help="Prompt text.")
    parser.add_argument("--request-id", default="smoke-real", help="Request id.")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="Tokens to generate.")
    parser.add_argument("--device", default="cpu", help="Model device, such as cpu or cuda.")
    parser.add_argument(
        "--torch-dtype",
        default=None,
        help="Optional torch dtype name, such as float16.",
    )
    safetensors_group = parser.add_mutually_exclusive_group()
    safetensors_group.add_argument(
        "--use-safetensors",
        dest="use_safetensors",
        action="store_true",
        default=None,
        help="Force safetensors weight loading when supported by Transformers.",
    )
    safetensors_group.add_argument(
        "--no-safetensors",
        dest="use_safetensors",
        action="store_false",
        help="Disable safetensors weight loading.",
    )
    parser.add_argument("--max-num-blocks", type=int, default=1024, help="Mock KV block count.")
    parser.add_argument("--block-size", type=int, default=16, help="Mock KV block size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = TransformersModelRunner.from_pretrained(
        args.model,
        device=args.device,
        torch_dtype=args.torch_dtype,
        use_safetensors=args.use_safetensors,
    )
    engine = Engine(
        EngineConfig(
            max_num_blocks=args.max_num_blocks,
            block_size=args.block_size,
            max_new_tokens=args.max_new_tokens,
        ),
        model_runner=runner,
    )
    engine.submit(
        GenerationRequest(
            request_id=args.request_id,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
        )
    )

    output = engine.run_until_complete()[0]
    print(output.text)


if __name__ == "__main__":
    main()

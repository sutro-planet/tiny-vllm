from tiny_vllm.request import GenerationOutput, GenerationRequest


class MockModelRunner:
    def generate(self, request: GenerationRequest, max_new_tokens: int) -> GenerationOutput:
        generated = [f"<mock-{index}>" for index in range(max_new_tokens)]
        text = " ".join([request.prompt, *generated]) if request.prompt else " ".join(generated)
        return GenerationOutput(
            request_id=request.request_id,
            text=text,
            generated_tokens=max_new_tokens,
        )

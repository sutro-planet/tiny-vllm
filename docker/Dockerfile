# CUDA 12.8 supports the RTX 5090 with the cluster's 575 driver.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385
ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.source="https://github.com/sutro-planet/tiny-vllm" \
      org.opencontainers.image.revision=$SOURCE_REVISION
ENV PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
WORKDIR /app
RUN python -m pip install --no-cache-dir transformers==4.56.2 aiohttp==3.12.15
COPY docker/download_model.py /tmp/download_model.py
RUN python /tmp/download_model.py
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-deps --no-cache-dir .
EXPOSE 8000
ENTRYPOINT ["tiny-vllm-serve"]
CMD ["--model", "/models/gpt2", "--host", "0.0.0.0", "--device", "cuda", "--torch-dtype", "float16", "--no-safetensors"]

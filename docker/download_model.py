"""Bake immutable benchmark snapshots (only tokenizers in the AIPerf image)."""

import sys

from huggingface_hub import snapshot_download

patterns = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]
if "--tokenizer-only" not in sys.argv:
    patterns.append("pytorch_model.bin")
for model, revision, name in (
    ("sshleifer/tiny-gpt2", "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be", "tiny-gpt2"),
    ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e", "gpt2"),
):
    snapshot_download(
        model, revision=revision, local_dir=f"/models/{name}", allow_patterns=patterns
    )

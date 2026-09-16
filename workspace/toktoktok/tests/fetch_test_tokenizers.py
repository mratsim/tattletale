#!/usr/bin/env python3
"""Stage the toktoktok test tokenizer data.

Same URLs and target directory as the shared test-data dir
download_tokenizers.nim, but stdlib-urllib instead of chronos:

the chronos downloader needs the chronos dev dependency, which is
not installed on every box and must not be installed by agents
(no installs rule). stdlib only, no packages.

Files land in workspace/toktoktok/tests/tokenizers/
(gitignored, the shared tokenizer-data directory the tokenizer test suites read).
"""

import urllib.request
from pathlib import Path

TEST_DIR = Path(__file__).parent.resolve()
TARGET = TEST_DIR.parent.parent.parent / "toktoktok" / "tests" / "tokenizers"

FILES = [
    ("https://huggingface.co/anthony/tokenizers-test/resolve/gpt-2/tokenizer.json?download=true",
     "gpt2-tokenizer.json"),
    ("https://huggingface.co/hf-internal-testing/llama3-tokenizer/resolve/main/tokenizer.json",
     "llama3-tokenizer.json"),
    ("https://huggingface.co/MiniMaxAI/MiniMax-M2.1/resolve/main/tokenizer.json?download=true",
     "minimax-m2.1-tokenizer.json"),
    ("https://huggingface.co/zai-org/GLM-4.7/resolve/main/tokenizer.json?download=true",
     "glm-4.7-tokenizer.json"),
    ("https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B/resolve/main/tokenizer.json?download=true",
     "exaone-tokenizer.json"),
    ("https://huggingface.co/stepfun-ai/Step-3.5-Flash/resolve/main/tokenizer.json?download=true",
     "step-3.5-flash-tokenizer.json"),
    ("https://huggingface.co/moonshotai/Kimi-K2.5/resolve/main/tiktoken.model?download=true",
     "kimik2.5.tiktoken"),
    ("https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
     "r50k_base.tiktoken"),
    ("https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
     "p50k_base.tiktoken"),
    ("https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
     "cl100k_base.tiktoken"),
    ("https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
     "o200k_base.tiktoken"),
]


def main():
    """Downloads the tokenizer fixture files into the target directory, skipping files already present."""
    TARGET.mkdir(parents=True, exist_ok=True)
    for url, name in FILES:
        dest = TARGET / name
        if dest.exists() and dest.stat().st_size > 0:
            print("skip (exists):", name)
            continue
        print("fetch:", name, flush=True)
        with urllib.request.urlopen(url, timeout=120) as resp:
            if resp.status != 200:
                raise SystemExit(f"HTTP {resp.status} for {name}")
            data = resp.read()
        dest.write_bytes(data)
        print(f"  {len(data)} bytes -> {dest}", flush=True)
    print("done")


if __name__ == "__main__":
    main()

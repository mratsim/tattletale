# Toktoktok
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/os
import chronos/apps/http/httpclient

const
  TOKENIZERS_DIR = "tokenizers"

  # HuggingFace tokenizers (HF JSON format)
  GPT2_HF_URL = "https://huggingface.co/anthony/tokenizers-test/resolve/gpt-2/tokenizer.json?download=true"
  GPT2_HF_FILENAME = "gpt2-tokenizer.json"
  LLAMA3_HF_URL = "https://huggingface.co/hf-internal-testing/llama3-tokenizer/resolve/main/tokenizer.json"
  LLAMA3_HF_FILENAME = "llama3-tokenizer.json"

  # Moonshot AI Kimi-K2.5 (tiktoken format)
  KIMIK25_URL = "https://huggingface.co/moonshotai/Kimi-K2.5/resolve/main/tiktoken.model?download=true"
  KIMIK25_FILENAME = "kimik2.5.tiktoken"

  # MiniMax AI MiniMax-M2.1 (HF format)
  MINIMAXM21_HF_URL = "https://huggingface.co/MiniMaxAI/MiniMax-M2.1/resolve/main/tokenizer.json?download=true"
  MINIMAXM21_HF_FILENAME = "minimax-m2.1-tokenizer.json"

  # Zhipu AI GLM-4.7 (HF format)
  GLM47_HF_URL = "https://huggingface.co/zai-org/GLM-4.7/resolve/main/tokenizer.json?download=true"
  GLM47_HF_FILENAME = "glm-4.7-tokenizer.json"

  # LGAI EXAONE K-EXAONE (HF format)
  # SuperBPE strategy with superword tokens for Korean, English and multilingual coverage
  EXAONE_HF_URL = "https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B/resolve/main/tokenizer.json?download=true"
  EXAONE_HF_FILENAME = "exaone-tokenizer.json"

  # StepFun AI Step-3.5-Flash (HF format)
  # Has multiple Split patterns for numbers, CJK, and general text
  STEP35_HF_URL = "https://huggingface.co/stepfun-ai/Step-3.5-Flash/resolve/main/tokenizer.json?download=true"
  STEP35_HF_FILENAME = "step-3.5-flash-tokenizer.json"

  # OpenAI tiktoken format (GPT-2 original BPE format)
  GPT2_VOCAB_URL = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe"
  GPT2_VOCAB_FILENAME = "gpt2-vocab.bpe"
  GPT2_ENCODER_URL = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json"
  GPT2_ENCODER_FILENAME = "gpt2-encoder.json"

  # OpenAI tiktoken format (.tiktoken files)
  R50K_BASE_URL = "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken"
  R50K_BASE_FILENAME = "r50k_base.tiktoken"
  P50K_BASE_URL = "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken"
  P50K_BASE_FILENAME = "p50k_base.tiktoken"
  CL100K_BASE_URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
  CL100K_BASE_FILENAME = "cl100k_base.tiktoken"
  O200K_BASE_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
  O200K_BASE_FILENAME = "o200k_base.tiktoken"

proc getProjectDir(): string {.compileTime.} =
  currentSourcePath.parentDir()

proc downloadTokenizer*(url, targetDir, filename: string) {.async.} =
  createDir(targetDir)
  let targetPath = targetDir / filename
  if fileExists(targetPath):
    echo "File already exists: ", targetPath
    return
  echo "Downloading: ", url
  echo "Target: ", targetPath
  let httpSession = HttpSessionRef.new()
  let resp = await httpSession.fetch(parseUri(url))
  if resp.status != 200:
    raise newException(IOError, "HTTP error: " & $resp.status)
  writeFile(targetPath, resp.data)
  echo "Download complete: ", filename
  await noCancel(httpSession.closeWait())

proc downloadAllTokenizers*() {.async.} =
  echo "======================================================================"
  echo "Downloading tokenizer test fixtures"
  echo "======================================================================"
  let targetDir = getProjectDir() / TOKENIZERS_DIR

  # HuggingFace tokenizers (HF JSON format)
  await downloadTokenizer(GPT2_HF_URL, targetDir, GPT2_HF_FILENAME)
  await downloadTokenizer(LLAMA3_HF_URL, targetDir, LLAMA3_HF_FILENAME)

  # New state-of-the-art tokenizers
  await downloadTokenizer(KIMIK25_URL, targetDir, KIMIK25_FILENAME)
  await downloadTokenizer(MINIMAXM21_HF_URL, targetDir, MINIMAXM21_HF_FILENAME)
  await downloadTokenizer(GLM47_HF_URL, targetDir, GLM47_HF_FILENAME)
  await downloadTokenizer(EXAONE_HF_URL, targetDir, EXAONE_HF_FILENAME)
  await downloadTokenizer(STEP35_HF_URL, targetDir, STEP35_HF_FILENAME)

  # OpenAI tiktoken format (GPT-2 original BPE format)
  await downloadTokenizer(GPT2_VOCAB_URL, targetDir, GPT2_VOCAB_FILENAME)
  await downloadTokenizer(GPT2_ENCODER_URL, targetDir, GPT2_ENCODER_FILENAME)

  # OpenAI tiktoken format (.tiktoken files)
  await downloadTokenizer(R50K_BASE_URL, targetDir, R50K_BASE_FILENAME)
  await downloadTokenizer(P50K_BASE_URL, targetDir, P50K_BASE_FILENAME)
  await downloadTokenizer(CL100K_BASE_URL, targetDir, CL100K_BASE_FILENAME)
  await downloadTokenizer(O200K_BASE_URL, targetDir, O200K_BASE_FILENAME)

  echo "======================================================================"
  echo "All tokenizers downloaded successfully"
  echo "======================================================================"

when isMainModule:
  waitFor downloadAllTokenizers()

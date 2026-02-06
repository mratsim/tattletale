# Toktoktok
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[os, strutils]
import chronos/apps/http/httpclient

const
  TOKENIZERS_DIR = "tokenizers"
  GPT2_URL = "https://huggingface.co/anthony/tokenizers-test/resolve/gpt-2/tokenizer.json?download=true"
  GPT2_FILENAME = "gpt2-tokenizer.json"
  LLAMA3_URL = "https://huggingface.co/hf-internal-testing/llama3-tokenizer/resolve/main/tokenizer.json"
  LLAMA3_FILENAME = "llama3-tokenizer.json"

proc getProjectDir(): string {.compileTime.} =
  currentSourcePath.rsplit(DirSep, 1)[0]

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
  await downloadTokenizer(GPT2_URL, targetDir, GPT2_FILENAME)
  await downloadTokenizer(LLAMA3_URL, targetDir, LLAMA3_FILENAME)
  echo "======================================================================"
  echo "All tokenizers downloaded successfully"
  echo "======================================================================"

when isMainModule:
  waitFor downloadAllTokenizers()

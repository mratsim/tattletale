## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## EXL3 Hadamard transform test.
##
## Compares ``hadamard_rotate_128`` (Nim reimpl) against ``ext.had_r_128``
## (production CUDA kernel) via dedicated fixtures.
##
## This isolates the Hadamard precision from the GEMM step so we can
## pinpoint whether a linear-layer test failure comes from the Hadamard
## reimplementation or the matrix multiply.
##
## Runs on CUDA via LD_PRELOAD of libtorch_cuda.so (same as other EXL3 tests).
##
## Usage:
##   1. Generate fixtures:
##      CUDA_HOME=... PATH=... python testgen/gen_exl3_hadamard_fixtures.py
##   2. Build and run:
##      nim cpp -r --hints:off q_exl3/test_exl3_hadamard.nim

import
  std/os,
  std/options,
  std/strformat,
  std/memfiles,
  workspace/safetensors,
  workspace/libtorch as F,
  workspace/positron,
  workspace/libtorch_testutils

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "exl3-hadamard"
  Tol = 1e-4

# ─── Test cases ─────────────────────────────────────────────────────

proc testCase(name: string) =
  let path = FixtureDir / &"hadamard_{name}.safetensor"
  if not fileExists(path):
    echo &"  SKIP: {path} not found"
    return

  var memFile = memFiles.open(path, mode = fmRead)
  defer: close(memFile)
  var st = safetensors.load(memFile)

  let input = st.getTensorOwned("input")
  let suh = st.getTensorOwned("suh")
  let svh = st.getTensorOwned("svh")
  let expPre = st.getTensorOwned("output_pre")
  let expPost = st.getTensorOwned("output_post")
  let expNone = st.getTensorOwned("output_none")

  echo &"  {name}: input={input.shape}"

  # Test 1: no scale
  let yNone = hadamard_rotate_128(input, pre_scale = none(Tensor), post_scale = none(Tensor))
  assertAllClose(yNone, expNone, rtol = Tol, abstol = Tol,
    msg = &"Hadamard [{name}] none: FWHT/√128 mismatch")

  # Test 2: pre_scale only (input Hadamard: suh before FWHT)
  let yPre = hadamard_rotate_128(input, pre_scale = some(suh), post_scale = none(Tensor))
  assertAllClose(yPre, expPre, rtol = Tol, abstol = Tol,
    msg = &"Hadamard [{name}] pre_scale mismatch")

  # Test 3: post_scale only (output Hadamard: svh after FWHT)
  let yPost = hadamard_rotate_128(input, pre_scale = none(Tensor), post_scale = some(svh))
  assertAllClose(yPost, expPost, rtol = Tol, abstol = Tol,
    msg = &"Hadamard [{name}] post_scale mismatch")


  echo &"  ✅ Hadamard [{name}] all 3 cases PASSED"

# ─── Main ──────────────────────────────────────────────────────────

when isMainModule:
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "EXL3 Hadamard transform: Nim reimpl vs production kernel"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  let cases = @["single_block", "two_blocks", "eight_blocks",
                "batch2_eight_blocks", "odd_blocks"]

  var passed, failed: int
  for name in cases:
    try:
      testCase(name)
      inc passed
    except:
      echo &"  ❌ Hadamard [{name}] FAILED"
      inc failed
      let e = getCurrentException()
      echo "    " & e.msg

  echo ""
  echo &"PASSED: {passed}/{passed+failed}"
  echo &"FAILED: {failed}/{passed+failed}"

  if failed > 0:
    quit 1

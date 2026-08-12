# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
## compound-assign `a[i] += v` / `acc += k` — read-modify-write store-back (OpenCL roundtrip)
##
## The compound-assign rewrite is registered as a COMMON pass (all backends),
## so OpenCL must desugar `x += y` → `x = x + y` and execute correctly.
## Two shapes are covered: an indexed write followed by indexed `+=` (proves
## store-back on a real lvalue) and a kernel-local `var acc` accumulator.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/opencl --nimcache:nimcache/tests/opencl \
##     workspace/crucible/tests/codegen/opencl/test_opencl_compound_assign.nim
##
##   cd workspace/crucible
##   nim c -r --hints:off --warnings:off --outdir:build/wip --nimcache:nimcache/wip \
##     tests/codegen/opencl/test_opencl_compound_assign.nim

import std/strutils
import workspace/crucible

# ── OpenCL C generation via `opencl:` macro ────────────────────────────────

const accCl = opencl:
  proc accKernel(a: ptr UncheckedArray[uint32];
                 output: ptr UncheckedArray[uint32]) {.global.} =
    for i in 0 ..< 2:
      var acc = uint32(0)
      for k in 0 ..< 4:
        acc += uint32(k)
      output[i] = a[i]
      output[i] += acc
      output[i] += 1

# ── Codegen test (always runs) ─────────────────────────────────────────────

echo "=== OpenCL compound-assign generation ===\n"
echo accCl
echo ""

doAssert not accCl.contains("+="), "compound assignment must be desugared, got: " & accCl

# ── Execution via OpenCL runtime ────────────────────────────────────────────

echo "=== OpenCL execution ===\n"

block: # accKernel: output[i] = a[i] + acc(0..3) + 1, acc(k) = 0+1+2+3 = 6
  var engine = bkOpenCL.init()
  engine.ingest(accCl)

  var a: array[2, uint32] = [10'u32, 20'u32]
  var out32: array[2, uint32]

  engine.run("accKernel", out32, (a))
  echo "  accKernel: output = [", out32[0], ", ", out32[1], "]"
  doAssert out32[0] == 17  # 10 + 6 + 1
  doAssert out32[1] == 27  # 20 + 6 + 1
  echo "  OK"

echo "All compound-assign execution tests passed"

## compound-assign `a[i] += v` — read-modify-write store-back (NVRTC roundtrip)
##
## Regression (Class A of the sgemm_1 port): ceramic `tv[m,n] += ...` — a
## statement-list-expression LHS wrapped in HiddenAddr — was by-value-blitted
## into a discarded temp and emitted as `((&_blit_N) += (...))`, which NVRTC
## rejects ("expression must be a modifiable lvalue") and which silently lost
## the accumulation. The compound-assign rewrite now runs as a common pass
## BEFORE legalization, desugaring `x += y` → `x = x + y` on the real lvalue.
##
## This test proves BOTH: NVRTC compiles a `+=` kernel, and the read-modify-
## write stores back (host-initialized values are read, accumulated, written).
##
## Run:
##   cd tattletale
##   nim cpp -r --hints:off --warnings:off \
##     --outdir:build/tests/nvrtc --nimcache:nimcache/tests/nvrtc \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_compound_assign.nim
##
##   cd workspace/crucible
##   nim cpp -r --hints:off --warnings:off --outdir:build/wip --nimcache:nimcache/wip \
##     tests/codegen/nvrtc/test_nvrtc_compound_assign.nim

import std/[unittest]
import workspace/crucible

const kernel = cuda:
  proc compoundKernel(C: ptr UncheckedArray[int32]) {.global.} =
    for i in 0 ..< 4:
      C[i] += 5
      C[i] *= 2

suite "compound assignment (a[i] += v)":
  test "accumulates and stores back (read-modify-write)":
    # Host-initialized values are copied H->D before launch: the kernel's `+=`
    # must READ the stored value and write the accumulation back, otherwise
    # the result would not depend on the initialization at all.
    var output: array[4, int32] = [10'i32, 11, 12, 13]
    var engine = bkCuda.init()
    engine.ingest(kernel)
    engine.run<<(1, 1)>>("compoundKernel", output, ())
    check output[0] == 30  # (10 + 5) * 2
    check output[1] == 32  # (11 + 5) * 2
    check output[2] == 34  # (12 + 5) * 2
    check output[3] == 36  # (13 + 5) * 2

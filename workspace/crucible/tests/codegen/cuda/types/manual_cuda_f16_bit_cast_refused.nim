## CUDA: a bit-reinterpret cast over a 16-bit float refuses at compile time.
##
## Test that a `cast[uint16]` over a float16 value in a cuda: kernel dies in the CUDA printer before any PTX is emitted.
## The refusal names the pass:
##
##   Error: cirv2 printer cuda: at kern: bit cast of a 16-bit float is unwired
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cuda/types/manual_cuda_f16_bit_cast_refused.nim

import workspace/crucible

const f16BitCastCuda = cuda:
  proc kern(output: ptr UncheckedArray[uint16], input: ptr UncheckedArray[float16]) {.global.} =
    output[0] = cast[uint16](input[0])

proc runTest() =
  var engine = newEngine(runCuda)
  engine.ingest(f16BitCastCuda)
  echo "UNREACHABLE — the 16-bit float bit cast must be refused at compile time"

when isMainModule:
  runTest()

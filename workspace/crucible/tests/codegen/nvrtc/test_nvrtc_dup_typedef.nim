## Reproduce: duplicate struct definitions in generated CUDA
##
## A simple tuple constructor `(Int[8](), Int[16]())` generates
## duplicate type definitions. The first has 4 fields (doubled),
## the second has 2 fields (correct).
##
## Run:
##   cd tattletale
##   CUDA_HOME=... PATH=... nim cpp -r \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_dup_typedef.nim

import std/[unittest]
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/runtime/engines

type Int*[V: static int] = object
  discard

const kernel = cuda:
  proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
    let t = (Int[8](), Int[16]())
    C[0] = 1.0

suite "Crucible - duplicate type definitions":
  test "tuple constructor generates single definition":
    let code = kernel
    var output: array[1, float32]
    var engine = bkCuda.init()
    engine.ingest(code)
    engine.run<<(1, 1)>>("kernel", output, ())
    check output[0] == 1.0

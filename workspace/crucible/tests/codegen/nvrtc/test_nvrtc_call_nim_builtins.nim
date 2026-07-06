## Reproduces the `inst.isBuiltIn()` assertion: Nim's getImpl() for
## magic builtins (system.* with magic: MulI) returns a procdef where
## isBuiltIn() == false, crashing registerGenericInstOrExternalProc.
##
## Triggered by fillWith inside cuda: — the for-loop index decomposition
## (foldDim) produces nnkCall to system.* that reaches the codegen's
## generic-proc registration path. The 3-arg crd2idx overload
## (scalar coord + tuple shape/stride) is the critical ingredient.
##
## Without ceramic imports this compiles (templates expand math away).
## With ceramic's int_tuples + layouts + tensors template chain the
## isBuiltIn() assertion fires.
##
## This test uses actual ceramic modules to reproduce identically.
##
## Run:
##   cd tattletale
##   CUDA_HOME=... PATH=... nim cpp -r \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_call_nim_builtins.nim

import std/[unittest]
import workspace/crucible/src/codegen/nvrtc
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/tensor_datatypes
import workspace/ceramic/src/tensors
import workspace/ceramic/src/kernel_fillwith_gpu

const kernel = cuda:
  proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((8, 16))
    var tv = make_view(C, L)
    fillWith(tv, 42.0'f32)

suite "NVRTC - builtin isBuiltIn assertion":
  test "fillWith inside cuda: triggers isBuiltIn() assertion":
    var buf: array[128, float32]
    var nv = initNvrtc(kernel)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    nv.execute("kernel", buf, ())
    check buf[0] == 42.0'f32

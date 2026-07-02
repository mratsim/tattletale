import std/[unittest]
import workspace/crucible/src/codegen/nvrtc
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/tensor_datatypes
import workspace/ceramic/src/tensors
import workspace/ceramic/src/kernel_gemm_gpu
import workspace/ceramic/src/kernel_fillwith_gpu

# ═════════════════════════════════════════════════════════════════════════
# Issue 1 (FIXED): PragmaExpr / genSym
#
# make_layout expands with evalOnceAs → {.genSym.} pragma.
# Crucible now strips PragmaExpr inside nnkConstDef and nnkCall.
# Also fixed: nnkObjConstr with Empty type child (Int[N]() in const).
# ═════════════════════════════════════════════════════════════════════════

const kernelLayout = cuda:
  proc kernel1(output: ptr UncheckedArray[uint32]) {.global.} =
    let L = make_layout((8, 16))
    output[0] = uint32(size(L.mode(0)).toIntVal)
    output[1] = uint32(size(L.mode(1)).toIntVal)

# ═════════════════════════════════════════════════════════════════════════
# Issue 2 (FIXED): nnkObjConstr with Empty type child
#
# The const x {.genSym.} = Int[N]() pattern generates nnkObjConstr where
# the type child is optimized away to Empty. Crucible now returns a proper
# gpuObjConstr with empty ocFields instead of gpuVoid.
# ═════════════════════════════════════════════════════════════════════════

const kernelObjConstr = cuda:
  proc kernel2(C: ptr UncheckedArray[uint32]) {.global.} =
    const x {.genSym.} = Int[8]()
    const tup {.genSym.} = (Int[1](), Int[8]())
    C[0] = 1'u32

# ═════════════════════════════════════════════════════════════════════════
# Issue 3 (FIXED): let-block-RHS with evalOnceAs pattern
#
# Inside a kernel, let L = block: const tmp; tmp  must not leak the
# constexpr into the assignment RHS. The unnestBlockInits pass lifts
# preceding statements before the variable declaration.
# ═════════════════════════════════════════════════════════════════════════

const kernelLetBlock = cuda:
  proc kernel3(C: ptr UncheckedArray[uint32]) {.global.} =
    let L = block:
      const tmp {.genSym.} = Int[8]()
      tmp
    C[0] = 1'u32

suite "Ceramic × Crucible anti-regression":
  test "Issue 1 — make_layout with genSym":
    var output: array[2, uint32]
    var nv = initNvrtc(kernelLayout)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    nv.execute("kernel1", output, ())
    check output[0] == 8
    check output[1] == 16

  test "Issue 2 — nnkObjConstr Empty child":
    var output: array[1, uint32]
    var nv = initNvrtc(kernelObjConstr)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    nv.execute("kernel2", output, ())
    check output[0] == 1

  test "Issue 3 — let-block-RHS":
    var output: array[1, uint32]
    var nv = initNvrtc(kernelLetBlock)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    nv.execute("kernel3", output, ())
    check output[0] == 1

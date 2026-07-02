import std/[unittest]
import workspace/crucible/src/codegen/nvrtc
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/tensor_datatypes
import workspace/ceramic/src/tensors

type
  FixMe*[V: static int] = object

# ═════════════════════════════════════════════════════════════════════════
# Issue 1 (FIXED): nnkPragmaExpr / genSym
# ═════════════════════════════════════════════════════════════════════════
const kernelGensym = cuda:
  proc kernel1(C: ptr UncheckedArray[uint32]) {.global.} =
    const x {.genSym.} = FixMe[8]()
    C[0] = 1'u32

# ═════════════════════════════════════════════════════════════════════════
# Issue 2 (FIXED): nnkObjConstr with Empty type child + Tuple
# ═════════════════════════════════════════════════════════════════════════
const kernelObjConstr = cuda:
  proc kernel2(C: ptr UncheckedArray[uint32]) {.global.} =
    const x {.genSym.} = FixMe[8]()
    const tup {.genSym.} = (FixMe[1](), FixMe[8]())
    C[0] = 1'u32

suite "Ceramic × Crucible anti-regression":
  test "Issue 1 — nnkPragmaExpr with genSym":
    var output: array[1, uint32]
    var nv = initNvrtc(kernelGensym)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    nv.execute("kernel1", output, ())
    check output[0] == 1

  test "Issue 2 — nnkObjConstr Empty child":
    var output: array[1, uint32]
    var nv = initNvrtc(kernelObjConstr)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    nv.execute("kernel2", output, ())
    check output[0] == 1

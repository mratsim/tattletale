import std/[unittest]
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/runtime/engines
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/tensor_datatypes
import workspace/ceramic/src/tensors
import workspace/ceramic/src/kernel_gemm_gpu
import workspace/ceramic/src/kernel_fillwith_gpu

# ═════════════════════════════════════════════════════════════════════════
# Issue 1 (FIXED): make_layout crashes cuda: with "expected an expression"
#
# make_layout uses evalOnceAs which wraps temporaries in {.genSym.}.
# Crucible couldn't handle nnkPragmaExpr — the genSym pragma crashed
# codegen with "expected an expression" at the constexpr declaration.
# ═════════════════════════════════════════════════════════════════════════
const kernelLayout = cuda:
  proc kernel1(output: ptr UncheckedArray[uint32]) {.global.} =
    let L = make_layout((8, 16))
    output[0] = uint32(size(L.mode(0)).toIntVal)
    output[1] = uint32(size(L.mode(1)).toIntVal)

# ═════════════════════════════════════════════════════════════════════════
# Issue 2 (FIXED): Int[N]() in const produces "= ;" in generated code
#
# const x = Int[8]() generates nnkObjConstr where Nim's const-folder
# replaces the type child with Empty. Crucible returned gpuVoid, causing
# the backend to emit "constexpr Type x = ;" — missing initializer.
# ═════════════════════════════════════════════════════════════════════════
const kernelObjConstr = cuda:
  proc kernel2(C: ptr UncheckedArray[uint32]) {.global.} =
    const x {.genSym.} = Int[8]()
    const tup {.genSym.} = (Int[1](), Int[8]())
    C[0] = 1'u32

# ═════════════════════════════════════════════════════════════════════════
# Issue 3 (FIXED): let L = block: const tmp; tmp leaks constexpr into RHS
#
# let L = block: const tmp = ...; tmp  generates "Type L = constexpr Type tmp = ...;"
# which fails NVRTC with "expected an expression" because the constexpr
# declaration is embedded in the middle of the assignment.
# ═════════════════════════════════════════════════════════════════════════
const kernelLetBlock = cuda:
  proc kernel3(C: ptr UncheckedArray[uint32]) {.global.} =
    let L = block:
      const tmp {.genSym.} = Int[8]()
      tmp
    C[0] = 1'u32

# ═════════════════════════════════════════════════════════════════════════
# Issue 4 (FIXED): make_layout inside cuda: fails NVRTC
#
# make_layout expands with evalOnceAs-generated constexpr in a let block.
# The constexpr leaked into the assignment RHS, producing the same
# "expected an expression" error from Issue 3. Fixed by the same pass.
# ═════════════════════════════════════════════════════════════════════════
const kernelTensorView = cuda:
  proc kernel4(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((8, 16))
    let tv = make_view(C, L)
    tv[0, 0] = 42.0'f32

suite "Ceramic × Crucible anti-regression":

  test "Issue 1 — make_layout with genSym":
    var output: array[2, uint32]
    var engine = bkCuda.init()
    engine.ingest(kernelLayout)
    engine.run<<(1, 1)>>("kernel1", output, ())
    check output[0] == 8
    check output[1] == 16

  test "Issue 2 — nnkObjConstr Empty child":
    var output: array[1, uint32]
    var engine = bkCuda.init()
    engine.ingest(kernelObjConstr)
    engine.run<<(1, 1)>>("kernel2", output, ())
    check output[0] == 1

  test "Issue 3 — let-block-RHS":
    var output: array[1, uint32]
    var engine = bkCuda.init()
    engine.ingest(kernelLetBlock)
    engine.run<<(1, 1)>>("kernel3", output, ())
    check output[0] == 1

  test "Issue 4 — make_layout + make_view + tv[]=":
    var buf: array[16, float32]
    var engine = bkCuda.init()
    engine.ingest(kernelTensorView)
    for i in 0 ..< buf.len: buf[i] = -1.0'f32
    engine.run<<(1, 1)>>("kernel4", buf, ())
    check buf[0] == 42.0'f32

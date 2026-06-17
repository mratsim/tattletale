## NVRTC: if-expressions (nnkIfExpr) in GPU code
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_if_expr.nim
##
## Tests the nnkIfExpr handler (nim_to_gpu.nim ~line 1089).
## Catches:
##   BUG-B-001: nnkElse vs nnkElseExpr — else branch silently dropped
##   BUG-A-001: hardcoded "result" as assignment target
##
## An if-expression is any `if cond: a else: b` that produces a value
## (as opposed to an if-statement). In Nim both use different AST nodes:
##   nnkIfStmt  → statement (not tested here)
##   nnkIfExpr  → expression (tested here)
import std/strformat
import workspace/crucible/src/codegen/nvrtc

# ── Helper: compile kernel and run ──────────────────────────────────

template runKernel(kernelCode: string; buf: var auto; kernelName: string) =
  var nv = initNvrtc(kernelCode)
  nv.compile()
  nv.getPtx()
  nv.execute(kernelName, buf, ())

# ═════════════════════════════════════════════════════════════════════
# Test 1: Basic if-else expression via let binding
# ═════════════════════════════════════════════════════════════════════

const kIfExprLet = cuda:
  proc ifExprLetKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let x = if true: 42'u32 else: 0'u32
    output[0] = x

var buf1: array[1, uint32]
runKernel(kIfExprLet, buf1, "ifExprLetKernel")
doAssert buf1[0] == 42, &"if-expr let (true): got {buf1[0]}, expected 42"

# ═════════════════════════════════════════════════════════════════════
# Test 2: If-expr else-branch taken (false condition)
# ═════════════════════════════════════════════════════════════════════

const kIfExprElse = cuda:
  proc ifExprElseKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let x = if false: 99'u32 else: 100'u32
    output[0] = x

var buf2: array[1, uint32]
runKernel(kIfExprElse, buf2, "ifExprElseKernel")
doAssert buf2[0] == 100, &"if-expr else-branch (false): got {buf2[0]}, expected 100"

# ═════════════════════════════════════════════════════════════════════
# Test 3: If-expr on RHS of assignment
# ═════════════════════════════════════════════════════════════════════

const kIfExprAsgn = cuda:
  proc ifExprAsgnKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = if true: 10'u32 else: 20'u32

var buf3: array[1, uint32]
runKernel(kIfExprAsgn, buf3, "ifExprAsgnKernel")
doAssert buf3[0] == 10, &"if-expr asgn (true): got {buf3[0]}, expected 10"

# ═════════════════════════════════════════════════════════════════════
# Test 4: If-expr on RHS of assignment — else branch
# ═════════════════════════════════════════════════════════════════════

const kIfExprAsgnElse = cuda:
  proc ifExprAsgnElseKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = if false: 10'u32 else: 20'u32

var buf4: array[1, uint32]
runKernel(kIfExprAsgnElse, buf4, "ifExprAsgnElseKernel")
doAssert buf4[0] == 20, &"if-expr asgn else (false): got {buf4[0]}, expected 20"

# ═════════════════════════════════════════════════════════════════════
# Test 5: If-expr with computed condition
# ═════════════════════════════════════════════════════════════════════

const kIfExprCond = cuda:
  proc ifExprCondKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let tid = blockIdx.x * blockDim.x + threadIdx.x
    let val = if tid < 2: 100'u32 else: 200'u32
    output[tid] = val

var buf5: array[4, uint32]
block:
  var nv = initNvrtc(kIfExprCond)
  nv.compile()
  nv.getPtx()
  nv.numBlocks = 1
  nv.threadsPerBlock = 4
  nv.execute("ifExprCondKernel", buf5, ())
doAssert buf5[0] == 100, &"if-expr cond (tid=0): got {buf5[0]}, expected 100"
doAssert buf5[1] == 100, &"if-expr cond (tid=1): got {buf5[1]}, expected 100"
doAssert buf5[2] == 200, &"if-expr cond (tid=2): got {buf5[2]}, expected 200"
doAssert buf5[3] == 200, &"if-expr cond (tid=3): got {buf5[3]}, expected 200"

# ═════════════════════════════════════════════════════════════════════
# Test 6: Multiple if-expr assignments in one kernel
# ═════════════════════════════════════════════════════════════════════

const kIfExprMulti = cuda:
  proc ifExprMultiKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = if true: 10'u32 else: 20'u32
    output[1] = if false: 30'u32 else: 40'u32

var buf6: array[2, uint32]
runKernel(kIfExprMulti, buf6, "ifExprMultiKernel")
doAssert buf6[0] == 10, &"if-expr multi (0): got {buf6[0]}, expected 10"
doAssert buf6[1] == 40, &"if-expr multi (1): got {buf6[1]}, expected 40"

# ═════════════════════════════════════════════════════════════════════
# Test 7: If-elif-else chain
# ═════════════════════════════════════════════════════════════════════

const kIfElifElse = cuda:
  proc ifElifElseKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let tid = blockIdx.x * blockDim.x + threadIdx.x
    let val = if tid == 0: 100'u32
               elif tid == 1: 200'u32
               elif tid == 2: 300'u32
               else: 999'u32
    output[tid] = val

var buf7: array[4, uint32]
block:
  var nv = initNvrtc(kIfElifElse)
  nv.compile()
  nv.getPtx()
  nv.numBlocks = 1
  nv.threadsPerBlock = 4
  nv.execute("ifElifElseKernel", buf7, ())
doAssert buf7[0] == 100, &"if-elif-else (tid=0): got {buf7[0]}, expected 100"
doAssert buf7[1] == 200, &"if-elif-else (tid=1): got {buf7[1]}, expected 200"
doAssert buf7[2] == 300, &"if-elif-else (tid=2): got {buf7[2]}, expected 300"
doAssert buf7[3] == 999, &"if-elif-else (tid=3): got {buf7[3]}, expected 999"

# ═════════════════════════════════════════════════════════════════════

echo "  OK — all if-expression tests pass"

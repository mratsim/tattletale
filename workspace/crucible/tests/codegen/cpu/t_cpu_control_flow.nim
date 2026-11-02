## CPU engine: control flow through the CIRv2-to-Nim and CIRv2-to-C emitters.
##
## Test that for-loop ranges, if/else statements, if-expressions and compound assignment produce the right output on the CPU.
##
## Run:
##   nim test_crucible_cpu
import std/strformat
import workspace/crucible

proc infixRangeKernel(output: ptr UncheckedArray[uint32]) =
  for i in 0 .. 3:
    output[i] = uint32(i) + 10'u32

proc varBoundKernel(output: ptr UncheckedArray[uint32]) =
  let n = 2'u32
  for i in 0 ..< n:
    output[i] = uint32(i) + 200'u32

proc ifElseKernel(output: ptr UncheckedArray[uint32]) =
  let lane = thread_index_in_threadgroup
  if lane < 2:
    output[lane] = 1'u32
  else:
    output[lane] = 2'u32

proc ifExprKernel(output: ptr UncheckedArray[uint32]) =
  let lane = thread_index_in_threadgroup
  let x = if lane mod 2 == 0: 7'u32 else: 9'u32
  output[lane] = x

proc compoundKernel(C: ptr UncheckedArray[int32]) =
  for i in 0 ..< 4:
    C[i] += 5
    C[i] *= 2

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runCpu)

  block:   # for-loop, infix inclusive range
    var buf: array[4, uint32]
    engine.launchCpuFn infixRangeKernel(cast[ptr UncheckedArray[uint32]](addr buf[0])) do:
      grid = (1, 1)
      blk = (1, 1)
    for i in 0 ..< 4:
      doAssert buf[i] == uint32(i) + 10, &"infix range [{i}]: got {buf[i]}"
    echo "[OK] for-loop infix inclusive range"

  block:   # for-loop, variable bound
    var buf: array[3, uint32]
    engine.launchCpuFn varBoundKernel(cast[ptr UncheckedArray[uint32]](addr buf[0])) do:
      grid = (1, 1)
      blk = (1, 1)
    doAssert buf[0] == 200, &"var bound [0]: got {buf[0]}"
    doAssert buf[1] == 201, &"var bound [1]: got {buf[1]}"
    doAssert buf[2] == 0, &"var bound [2] untouched: got {buf[2]}"
    echo "[OK] for-loop variable bound"

  block:   # if/else statement
    var buf: array[4, uint32]
    engine.launchCpuFn ifElseKernel(cast[ptr UncheckedArray[uint32]](addr buf[0])) do:
      grid = (1, 1)
      blk = (4, 1)
    doAssert buf[0] == 1 and buf[1] == 1, &"then branch: got {buf[0..1]}"
    doAssert buf[2] == 2 and buf[3] == 2, &"else branch: got {buf[2..3]}"
    echo "[OK] if/else statement"

  block:   # if-expression in a let binding
    var buf: array[4, uint32]
    engine.launchCpuFn ifExprKernel(cast[ptr UncheckedArray[uint32]](addr buf[0])) do:
      grid = (1, 1)
      blk = (4, 1)
    doAssert buf[0] == 7 and buf[2] == 7, &"if-expr even lanes: got {buf[0]}, {buf[2]}"
    doAssert buf[1] == 9 and buf[3] == 9, &"if-expr odd lanes: got {buf[1]}, {buf[3]}"
    echo "[OK] if-expression in a let binding"

  block:   # compound assignment, read-modify-write
    var buf: array[4, int32]
    for i in 0 ..< 4:
      buf[i] = int32(i)
    engine.launchCpuFn compoundKernel(cast[ptr UncheckedArray[int32]](addr buf[0])) do:
      grid = (1, 1)
      blk = (1, 1)
    for i in 0 ..< 4:
      doAssert buf[i] == (int32(i) + 5) * 2, &"compound [{i}]: got {buf[i]}"
    echo "[OK] compound assignment read-modify-write"

runTest()
echo "t_cpu_control_flow: all checks passed"

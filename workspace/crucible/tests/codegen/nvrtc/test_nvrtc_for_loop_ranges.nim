## NVRTC: for-loop range parsing (nnkForStmt)
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_for_loop_ranges.nim
##
## Exercises the range parsing at nim_to_gpu.nim ~line 1116:
##   - Infix range: `0 .. N`
##   - Generic bound
##   - Body access via node[^1]
import std/strformat
import workspace/crucible

# ── Test 1: Basic infix range (0 .. N) ──
# Note: codegen uses `i < N`, so Nim's inclusive `..` is off by one.
# Use upper bound = desired_count to compensate.

const kInfix = cuda:
  proc infixRangeKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    for i in 0 .. 3:
      output[i] = uint32(i) + 10'u32

var buf1: array[4, uint32]
block:
  var engine = bkCuda.init()
  engine.ingest(kInfix)
  engine.run<<(1, 4)>>("infixRangeKernel", buf1, ())
doAssert buf1[0] == 10, &"range [0]: got {buf1[0]}"
doAssert buf1[1] == 11, &"range [1]: got {buf1[1]}"
doAssert buf1[2] == 12, &"range [2]: got {buf1[2]}"
doAssert buf1[3] == 13, &"range [3]: got {buf1[3]}"
# ── Test 2: Loop with body access via node[^1] ──

const kBodyAccess = cuda:
  proc bodyAccessKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    for i in 0 .. 2:
      output[i] = uint32(i) + 200'u32

var buf2: array[3, uint32]
block:
  var engine = bkCuda.init()
  engine.ingest(kBodyAccess)
  engine.run<<(1, 2)>>("bodyAccessKernel", buf2, ())
doAssert buf2[0] == 200, &"body access [0]: got {buf2[0]}"
doAssert buf2[1] == 201, &"body access [1]: got {buf2[1]}"
doAssert buf2[2] == 202, &"body access [2]: got {buf2[2]}"
# ── Test 3: Loop where range uses a variable

const kVarBound = cuda:
  proc varBoundKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let n = 2'u32
    for i in 0 .. n:
      if uint32(i) < n:
        output[i] = 300'u32

var buf3: array[2, uint32]
block:
  var engine = bkCuda.init()
  engine.ingest(kVarBound)
  engine.run<<(1, 2)>>("varBoundKernel", buf3, ())
doAssert buf3[0] == 300, &"var bound [0]: got {buf3[0]}"
doAssert buf3[1] == 300, &"var bound [1]: got {buf3[1]}"

echo "  OK — all for-loop range tests pass"

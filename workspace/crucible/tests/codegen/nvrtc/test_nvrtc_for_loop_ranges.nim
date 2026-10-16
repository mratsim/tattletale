## NVRTC: for-loop range parsing (nnkForStmt)
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/test_nvrtc_for_loop_ranges.nim
import std/strformat
import workspace/crucible

# ── Test 1: Basic infix range (0 .. N) ──
# Note: codegen uses `i < N`, so Nim's inclusive `..` is off by one.
# Use upper bound = desired_count to compensate.

const kInfix = cuda:
  proc infixRangeKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    for i in 0 .. 3:
      output[i] = uint32(i) + 10'u32

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf1: array[4, uint32]
  block:
    var engine = bkCuda.init()
    engine.ingest(kInfix)
    let oDev = engine.allocDevice(sizeof(buf1))
    engine.launchKernel<<(1, 4)>>("infixRangeKernel", oDev, ())
    engine.copyToHostFromDevice(addr buf1[0], oDev, sizeof(buf1))
    engine.freeDevice(oDev)
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
    let oDev = engine.allocDevice(sizeof(buf2))
    engine.launchKernel<<(1, 2)>>("bodyAccessKernel", oDev, ())
    engine.copyToHostFromDevice(addr buf2[0], oDev, sizeof(buf2))
    engine.freeDevice(oDev)
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
    let oDev = engine.allocDevice(sizeof(buf3))
    engine.launchKernel<<(1, 2)>>("varBoundKernel", oDev, ())
    engine.copyToHostFromDevice(addr buf3[0], oDev, sizeof(buf3))
    engine.freeDevice(oDev)
  doAssert buf3[0] == 300, &"var bound [0]: got {buf3[0]}"
  doAssert buf3[1] == 300, &"var bound [1]: got {buf3[1]}"

  echo "  OK — all for-loop range tests pass"

when isMainModule:
  runTest()

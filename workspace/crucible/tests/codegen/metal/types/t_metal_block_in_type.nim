## Metal: gpuBlock(isExpr) in symbol/mutability helpers. The `` `()` ``
## template on MySpan expands to a block with local temps and a result value,
## which the frontend lowers to an expression-shaped gpuBlock.
## A device fn reads `.idx` off the block value.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/types/t_metal_block_in_type.nim

import std/unittest
import workspace/crucible

# ── Types ────────────────────────────────────────────────────────────────────
type
  MySpan = object
    idx: int32
    len: int32

# ── `()` operator: block: + local temps + result → gpuBlock(isExpr) ────────
template `()`(s: MySpan, a, b: int32): auto =
  block:
    let coord = (a, b)
    let offset = coord[0] * s.len + coord[1]
    var result: MySpan
    result.idx = s.idx + int32(offset)
    result.len = s.len
    result

# ── Device function: inline `()` + field access ────────────────────────────
proc deviceFn(span: MySpan): int32 =
  result = span(0, 0).idx

# ── Kernel ──────────────────────────────────────────────────────────────────
const kernel = metal:
  proc reproKernel(output: ptr UncheckedArray[float32],
                   M, N: int32) {.global.} =
    let s = MySpan(idx: M, len: N)
    output[0] = float32(deviceFn(s))

# ── Harness ─────────────────────────────────────────────────────────────────

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo kernel

  suite "Metal - block-in-type":
    test "block-valued `()` inside a device fn executes on the device":
      var engine = bkMetal.newEngine()
      engine.ingest(kernel)
      let oDev = engine.allocDevice(float32, 1)
      engine.launchKernel("reproKernel", oDev, (7'i32, 3'i32))
      var res: array[1, float32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 7.0'f32

when isMainModule:
  runTest()

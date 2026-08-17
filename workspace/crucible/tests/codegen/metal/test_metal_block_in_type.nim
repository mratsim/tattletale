## Metal: gpuBlock(isExpr) in symbol/mutability helpers. The `` `()` ``
## template on MySpan expands to a block with local temps and a result value,
## which the frontend lowers to an expression-shaped gpuBlock.
## The device fn reads `.idx` off the block value. The kernel writes the resulting value
## into the output buffer, verifying the block-in-type lowering on the device
## rather than only in the printed MSL.
##
## Tested ABI (macOS 26.6.1, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10, MSL 4.0.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_block_in_type.nim

import std/unittest
import workspace/crucible

# ── Types ────────────────────────────────────────────────────────────────────
type
  MySpan = object
    idx: int32
    len: int32

# ── `()` operator: block: + local temps + result → gpuBlock(isExpr) ────────
template `()`(s: MySpan; a, b: int32): auto =
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
      var engine = bkMetal.init()
      engine.ingest(kernel)
      var res: array[1, float32]
      engine.run("reproKernel", res, (7'i32, 3'i32))
      check res[0] == 7.0'f32

when isMainModule:
  runTest()

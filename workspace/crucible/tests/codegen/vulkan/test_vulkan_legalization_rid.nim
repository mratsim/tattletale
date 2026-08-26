## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## Vulkan: RID regression tests for the IR legalization passes.
##
## Locks the RID review's verified silent-wrong-codegen findings to
## regression tests (engine.run value runs + GLSL shape asserts):
##   (a) BUG-A-001 — ptr-arith index folding must preserve the WHOLE index
##       (`(A +% off)[i + j]` folds to `A[off + (i + j)]`, never `A[off + j]`);
##   (b) HIDN-A-001 — re-assignment of a tainted struct var must update the
##       flattened value leaves (the second value wins, not the stale first);
##   (c) BUG-A-003 — array-var-param device fns inline in statement position,
##       and any `return` in the callee body is rejected loudly (an inlined
##       return would return from the host kernel).
##
## Run:
##   nim cpp -r --hints:off --warnings:off \
##     --outdir:build/tests/vulkan --nimcache:nimcache/tests/vulkan \
##     workspace/crucible/tests/codegen/vulkan/test_vulkan_legalization_rid.nim

import std/[strutils, unittest]
import workspace/crucible
import workspace/ceramic/src/ptr_arithmetic

# ── (a) BUG-A-001: a `+`-binop index must keep both operands ─────────────

const ptrArithVk = vulkan:
  proc gather(dst: ptr UncheckedArray[uint32], src: ptr UncheckedArray[uint32],
              off, i, j: uint32) {.global.} =
    dst[0] = (src +% off)[i + j]

# Same shape with a uint32 offset against an int32 index: the fold must
# coerce the offset to the index's type (COMP-B-003) — GLSL forbids
# mixed-type arithmetic operands.
const ptrArithMixVk = vulkan:
  proc gatherMix(dst: ptr UncheckedArray[uint32], src: ptr UncheckedArray[uint32],
                 off: uint32, i, j: int32) {.global.} =
    dst[0] = (src +% off)[i + j]

# ── (b) HIDN-A-001: tainted-struct re-assignment keeps the NEW value ─────

type
  View = object
    data: ptr UncheckedArray[uint32]
    scale: uint32

const reAssignVk = vulkan:
  proc pick(v: View, dst: ptr UncheckedArray[uint32]) {.device.} =
    dst[0] = v.data[0] * v.scale

  proc reAssignKernel(dst: ptr UncheckedArray[uint32],
                      a, b: ptr UncheckedArray[uint32], useB: uint32) {.global.} =
    var v = View(data: a, scale: 2'u32)
    if useB == 1'u32:
      v = View(data: b, scale: 3'u32)
    pick(v, dst)

# ── (c) BUG-A-003: array-var-param inlining (happy + return rejection) ───

const earlyRetVk = vulkan:
  proc bumpLimit(x: var int32, limit: int32) {.device.} =
    if x > limit:
      return
    x = x + 1'i32

  proc earlyRetKernel(dst: ptr UncheckedArray[uint32]) {.global.} =
    var v: int32 = 5
    bumpLimit(v, 4)         # 5 > 4 → early return, v stays 5
    dst[0] = uint32(v)

# ── (d) BUG-A-005: unwritten var param keeps an addr-wrapped arg ─────────

type
  Holder = object
    val: uint32

const unwrittenVk = vulkan:
  proc readOnly(x: var uint32): uint32 {.device.} =
    x + 1'u32

  proc unwrittenKernel(dst: ptr UncheckedArray[uint32]) {.global.} =
    var h: Holder
    h.val = 41'u32
    dst[0] = readOnly(h.val)

const zeroFillVk = vulkan:
  proc zeroFill(d: var array[4, uint32]) {.device.} =
    for i in 0 ..< 4:
      d[i] = 0'u32

  proc zeroKernel(dst: ptr UncheckedArray[uint32]) {.global.} =
    var buf: array[4, uint32]
    buf[0] = 9'u32
    buf[1] = 8'u32
    buf[2] = 7'u32
    buf[3] = 6'u32
    zeroFill(buf)
    dst[0] = buf[0] + buf[1] + buf[2] + buf[3]

static:
  # An array-var-param fn whose body contains a `return` must be rejected
  # loudly: inlining it would splice the `return` into the host kernel,
  # silently truncating it (BUG-A-003).
  doAssert not compiles(block:
    const bad = vulkan:
      proc zeroFillRet(d: var array[4, uint32]) {.device.} =
        if d[0] == 0'u32:
          return
        for i in 0 ..< 4:
          d[i] = 0'u32
      proc k(dst: ptr UncheckedArray[uint32]) {.global.} =
        var buf: array[4, uint32]
        buf[0] = 9'u32
        zeroFillRet(buf)
        dst[0] = buf[0]
  )

  # COV-A-005: the passes' rejection guards fire loudly instead of emitting
  # valid-but-wrong GLSL.
  # (1) two written var params — GLSL fns return one value
  doAssert not compiles(block:
    const bad = vulkan:
      proc twoWritten(a: var uint32, b: var uint32) {.device.} =
        a = a + 1'u32
        b = b + 1'u32
      proc k(dst: ptr UncheckedArray[uint32]) {.global.} =
        var x: uint32 = 1
        var y: uint32 = 2
        twoWritten(x, y)
        dst[0] = x + y
  )
  # (2) written var param + non-void return — cannot lower
  doAssert not compiles(block:
    const bad = vulkan:
      proc retAndMutate(x: var uint32): uint32 {.device.} =
        x = x + 1'u32
        return x
      proc k(dst: ptr UncheckedArray[uint32]) {.global.} =
        var x: uint32 = 1
        dst[0] = retAndMutate(x)
  )

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Vulkan - RID legalization regressions":
    test "BUG-A-001: (A +% off)[i + j] folds to A[off + (i + j)]":
      # The whole index must survive the fold: the emitted SSBO index is
      # `src[(off + (i + j))]` — the old special `+` branch dropped `i` and
      # emitted `src[(off + j)]` (silent wrong-address codegen).
      check "src[(off + (i + j))]" in ptrArithVk
      check "src[(off + j)]" notin ptrArithVk
      var engine = bkVulkan.init()
      engine.ingest(ptrArithVk)
      var res: array[1, uint32]
      let src = [10'u32, 20, 30, 40, 50, 60, 70]
      engine.run << (grid: (1, 1), blk: (1, 1)) >> (
        "gather", res, (src, 1'u32, 2'u32, 3'u32))
      # src[off + i + j] = src[1 + 2 + 3] = src[6] = 70
      check res[0] == 70'u32

    test "COMP-B-003: uint32 offset folds with an int32 index (coerced)":
      # The offset keeps its uint32 type while the index is int32 — the fold
      # must coerce the offset (`int(off)`) so the emitted binop is
      # same-typed; without the coercion glslang rejects the mixed-type add.
      check "src[(int(off) + (i + j))]" in ptrArithMixVk
      var engine = bkVulkan.init()
      engine.ingest(ptrArithMixVk)
      var res: array[1, uint32]
      let src = [10'u32, 20, 30, 40, 50, 60, 70]
      engine.run << (grid: (1, 1), blk: (1, 1)) >> (
        "gatherMix", res, (src, 1'u32, 2'i32, 3'i32))
      check res[0] == 70'u32

    test "HIDN-A-001: tainted-struct re-assignment uses the second value":
      # `v = View(data: b, scale: 3)` inside the conditional must update the
      # flattened value leaf: pick(v, dst) reads b[0] * 3. The old code
      # dropped the assign and left the declared scale leaf at 2 (stale).
      # NOTE: ptr leaves are compile-time expressions (GLSL has no pointer
      # locals), so the LAST textual assignment wins statically; the value
      # leaf (`v_scale`) is a real GLSL var and follows the runtime branch.
      var engine = bkVulkan.init()
      engine.ingest(reAssignVk)
      var res: array[1, uint32]
      let a = [10'u32]
      let b = [14'u32]
      engine.run << (grid: (1, 1), blk: (1, 1)) >> (
        "reAssignKernel", res, (a, b, 1'u32))
      check res[0] == 42'u32

    test "BUG-A-002: early `return` in a written-var fn carries the value":
      # bumpLimit(x: var int32) is converted to return-by-value; the early
      # `return` inside the if must become `return x;` (the old code left a
      # bare `return;` — invalid GLSL in a non-void fn).
      check "return x;" in earlyRetVk
      var engine = bkVulkan.init()
      engine.ingest(earlyRetVk)
      var res: array[1, uint32]
      engine.run << (grid: (1, 1), blk: (1, 1)) >> ("earlyRetKernel", res, ())
      # 5 > 4 → the early return keeps v = 5
      check res[0] == 5'u32

    test "BUG-A-005: unwritten var param arg loses its addr wrap":
      # readOnly(x: var uint32): uint32 reads x without writing it — the
      # param converts to a value param and the call-site arg `h.val` must
      # be unwrapped (addr → value). The old code left the addr-wrapped arg
      # and codegen raised "Vulkan GLSL does not support addr".
      var engine = bkVulkan.init()
      engine.ingest(unwrittenVk)
      var res: array[1, uint32]
      engine.run << (grid: (1, 1), blk: (1, 1)) >> ("unwrittenKernel", res, ())
      check res[0] == 42'u32

    test "BUG-A-003: array-var-param fn inlines in statement position":
      # zeroFill(d: var array[4, uint32]) is inlined into zeroKernel — the
      # inlined body runs and zeroes the buffer (9+8+7+6 → 0+0+0+0). The
      # return-rejection path is pinned by the static `not compiles` above.
      check "for(int i = 0; i < 4; i += 1)" in zeroFillVk
      var engine = bkVulkan.init()
      engine.ingest(zeroFillVk)
      var res: array[1, uint32]
      engine.run << (grid: (1, 1), blk: (1, 1)) >> ("zeroKernel", res, ())
      check res[0] == 0'u32

when isMainModule:
  runTest()

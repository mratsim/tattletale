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

# Same shape, but the offset is a uint32-returning DEVICE-FN CALL: the fold
# must resolve the callee's return type before coercing, otherwise it skips
# the coercion and ships mixed-type GLSL that glslang rejects only at
# ingest (opaque shader error instead of a fold-time diagnostic).
const ptrArithCallVk = vulkan:
  proc getOff(): uint32 {.device.} =
    result = 1'u32

  proc gatherCall(dst: ptr UncheckedArray[uint32], src: ptr UncheckedArray[uint32],
                  i, j: int32) {.global.} =
    dst[0] = (src +% getOff())[i + j]

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

# ── (e) SEC-B-001: a user device fn shadowing a builtin keeps its body ──
#        only `{.builtin.}` procs remap to the backend spelling

const userRsqrtVk = vulkan:
  proc rsqrt(x: float32): float32 {.device.} =
    x * 0.25'f32          # user math — deliberately NOT 1/sqrt(x)

  proc rsqrtKernel(dst: ptr UncheckedArray[float32], v: float32) {.global.} =
    dst[0] = rsqrt(v)

# Same-body twin for the device value run: an MSL-family device cannot
# compile a user fn literally named `rsqrt` — MoltenVK's SPIR-V→MSL
# translation emits the call as `rsqrt(param)`, ambiguous between the user
# fn and Metal stdlib `rsqrt` (platform limitation, not a codegen issue;
# GLSL has no `rsqrt` builtin so the name is free there). The emission
# asserts on userRsqrtVk lock in the name handling; this twin locks in the
# user math on device.
const userRsqrtTwinVk = vulkan:
  proc rsqrtUser(x: float32): float32 {.device.} =
    x * 0.25'f32

  proc rsqrtKernel(dst: ptr UncheckedArray[float32], v: float32) {.global.} =
    dst[0] = rsqrtUser(v)

const builtinRsqrtVk = vulkan:
  proc rsqrtKernel(dst: ptr UncheckedArray[float32], v: float32) {.global.} =
    dst[0] = rsqrt(v)     # the `{.builtin.}` rsqrt → GLSL inversesqrt

# ── (f) multi-var-arg calls — unwritten args are value params ─────────────────

const add2Vk = vulkan:
  proc add2(a: var uint32, b: var uint32): uint32 {.device.} =
    a + b

  proc add2Kernel(dst: ptr UncheckedArray[uint32]) {.global.} =
    var x: uint32 = 1
    var y: uint32 = 2
    dst[0] = add2(x, y)

const bumpVk = vulkan:
  proc bump(x: var uint32, limit: var uint32) {.device.} =
    if x > limit:
      return
    x = x + 1'u32

  proc bumpKernel(dst: ptr UncheckedArray[uint32]) {.global.} =
    var x: uint32 = 1
    var lim: uint32 = 5
    bump(x, lim)
    dst[0] = x

# ── (g) fixReturn stops at nested proc boundaries ─────────────────────────────

const nestedProcVk = vulkan:
  proc outer(x: var int32) {.device.} =
    proc inner(y: int32): int32 =
      if y > 10:
        return y
      y + 1
    x = inner(x)

  proc nestedKernel(dst: ptr UncheckedArray[uint32]) {.global.} =
    var v: int32 = 5
    outer(v)
    dst[0] = uint32(v)

# ── (h) lane-id rewrite is node-local (same-module scoping) ───────────────────

const laneShareVk = vulkan:
  proc leafShuffle(acc: float32, lane: uint32): float32 {.device.} =
    simdShuffleDown(acc, lane)

  proc mid(acc: float32): float32 {.device.} =
    let lane = uint32(thread_index_in_threadgroup)
    leafShuffle(acc, lane)

  proc nonShuffle(acc: float32): float32 {.device.} =
    let lane = uint32(thread_index_in_threadgroup)
    acc + float32(lane)

  proc k1(dst: ptr UncheckedArray[float32]) {.global.} =
    dst[0] = mid(dst[0])

  proc k2(dst: ptr UncheckedArray[float32]) {.global.} =
    dst[0] = nonShuffle(dst[0])

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
  # (1) two written var params — GLSL fns return one value. `not compiles`
  # asserts only "the module must not compile": with the fn-level guard
  # present the DEFINITION is rejected during conversion; without it the
  # call-site guard (written var args > 1) rejects the CALL. Both guards
  # enforce the same rule, so deleting either leaves this test passing —
  # guard-level attribution is not asserted in-tree. The
  # positive control below fixes the boundary: one written + one
  # unwritten var arg MUST compile (the call-site guard counts only
  # written args).
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

    test "COMP-B-003: device-fn call offset folds with an int32 index (coerced)":
      # The offset is `getOff()` (a uint32-returning device fn): the fold
      # resolves its return type via the fn table and coerces to the int32
      # index type — the emitted binop is same-typed and the call survives.
      check "src[(int(getOff()) + (i + j))]" in ptrArithCallVk
      var engine = bkVulkan.init()
      engine.ingest(ptrArithCallVk)
      var res: array[1, uint32]
      let src = [10'u32, 20, 30, 40, 50, 60, 70]
      engine.run << (grid: (1, 1), blk: (1, 1)) >> (
        "gatherCall", res, (src, 2'i32, 3'i32))
      check res[0] == 70'u32

    test "HIDN-A-001: tainted-struct re-assignment uses the new value":
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
      # return-rejection path is locked in by the static `not compiles` above.
      check "for(int i = 0; i < 4; i += 1)" in zeroFillVk
      var engine = bkVulkan.init()
      engine.ingest(zeroFillVk)
      var res: array[1, uint32]
      engine.run << (grid: (1, 1), blk: (1, 1)) >> ("zeroKernel", res, ())
      check res[0] == 0'u32

    test "two unwritten var args lower to plain value params":
      # add2 reads both var params without writing them — GLSL's
      # one-return-value rule is not violated, so the call must lower to
      # `dst[0] = add2(x, y)` with `uint add2(uint a, uint b)`. The old
      # call-site guard rejected ANY multi-var-arg call with a factually
      # wrong message.
      check "uint add2(uint a, uint b)" in add2Vk
      check "dst[0] = add2(x, y);" in add2Vk
      var engine = bkVulkan.init()
      engine.ingest(add2Vk)
      var res: array[1, uint32]
      engine.run << (grid: (1, 1), blk: (1, 1)) >> ("add2Kernel", res, ())
      check res[0] == 3'u32

    test "written + unwritten var args mix (one return value)":
      # bump writes only x — the fn-level guard permits the fn (one written
      # param → one return value), so the call must lower to
      # `x = bump(x, lim)` with `uint bump(uint x, uint limit)`. The old
      # call-site guard rejected the call site anyway.
      check "uint bump(uint x, uint limit)" in bumpVk
      check "x = bump(x, lim);" in bumpVk
      var engine = bkVulkan.init()
      engine.ingest(bumpVk)
      var res: array[1, uint32]
      engine.run << (grid: (1, 1), blk: (1, 1)) >> ("bumpKernel", res, ())
      # 1 ≤ 5 → bump applies: x = 2
      check res[0] == 2'u32

    test "fixReturn stops at nested proc boundaries":
      # outer(x: var int32) converts to return-by-value; the return rewrite
      # must touch ONLY outer's own returns. The old full-tree recursion
      # descended into the nested `inner`'s gpuProc body and rewrote its
      # returns with outer's retIdent (`return y;` / `return result;` became
      # `return x;` — an undeclared x inside the hoisted top-level inner).
      # The module cannot run on device yet: codegen emits the nested
      # definition in place AND a hoisted top-level copy, which glslang
      # rejects (pre-existing nested-fn emission defect) —
      # the asserts lock in the return rewrite itself.
      check "return y;" in nestedProcVk
      check "return result;" in nestedProcVk

    test "lane-id rewrite is node-local — non-shuffle fn keeps gl_LocalInvocationIndex":
      # mid (shuffle-reachable) and nonShuffle (not) both read
      # thread_index_in_threadgroup in the same vulkan: block. The catalog
      # ident node is sigTab-shared, so an in-place symbol swap on mid's
      # copy would leak gl_SubgroupInvocationID into nonShuffle (an
      # in-place swap made nonShuffle emit the subgroup lane). The rewrite
      # must replace the
      # node, leaving the shared node intact for non-shuffle fns.
      check "uint lane = uint(gl_SubgroupInvocationID);" in laneShareVk
      check "uint lane = uint(gl_LocalInvocationIndex);" in laneShareVk

    test "SEC-B-001: user device fn shadowing a builtin name keeps its body":
      # The user `rsqrt` must be emitted as its own GLSL fn with its call
      # intact. Remapping by name alone would rebind the call to the GLSL
      # builtin `inversesqrt`, silently changing the math to 1/sqrt(x) and
      # dead-coding the user body.
      check "float rsqrt(float x)" in userRsqrtVk
      check "rsqrt(v)" in userRsqrtVk
      check "result = (x * 0.25f)" in userRsqrtVk
      check "inversesqrt" notin userRsqrtVk
      # device value run on the same-body twin (see the kernel comment for
      # why the `rsqrt`-named fn itself cannot run on an MSL-family device):
      # user math 4 × 0.25 = 1.0, not 0.5 (inversesqrt).
      var engine = bkVulkan.init()
      engine.ingest(userRsqrtTwinVk)
      var res: array[1, float32]
      engine.run << (grid: (1, 1), blk: (1, 1)) >> ("rsqrtKernel", res, (4.0'f32,))
      check res[0] == 1.0'f32

    test "SEC-B-001: the {.builtin.} rsqrt still remaps to inversesqrt":
      check "inversesqrt(v)" in builtinRsqrtVk
      var engine = bkVulkan.init()
      engine.ingest(builtinRsqrtVk)
      var res: array[1, float32]
      engine.run << (grid: (1, 1), blk: (1, 1)) >> ("rsqrtKernel", res, (4.0'f32,))
      check abs(res[0] - 0.5'f32) < 1e-6'f32

when isMainModule:
  runTest()

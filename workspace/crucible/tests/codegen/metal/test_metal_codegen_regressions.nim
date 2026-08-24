## Metal: four codegen contracts the tile kernels depend on.
##
## 1. A `{.builtin.}` proc called from an overloaded proc body keeps its plain name in the emitted MSL:
##    the DSL forwards the backend's native function, which knows no mangled hash.
##    The tile layer's rsqrt/exp2 scalar math builtins resolve this way from inside generic op bodies.
## 2. A gpuAssign whose LHS needs a blit keeps the LHS preamble: the RHS
##    blit appends to it, so the assignment's blitted index expression
##    stays declared. The local store fixture below emits exactly this shape.
##    The MSL compile (the ingest below) fails with an undeclared `_blit_N`
##    if the preamble is dropped.
## 3. MSL struct emission drops gtString fields: MSL has no string type.
##    A struct carrying a string field would otherwise emit an illegal
##    `const char*` member, which the MSL compiler rejects.
## 4. A simdgroup-matrix var param crossing a device-function boundary
##    lowers to a thread-space pointer, and the mma intrinsic needs
##    a thread reference, so the argument renders dereferenced (`(*d)`).
##    A plain-array var param (the FMA fragment) indexes directly,
##    never dereferenced. Dropping the deref emits the intrinsic
##    over a pointer, which the MSL compiler rejects.
##
## Each test is a committed regression: reverting the corresponding codegen
## fix breaks it, whether by a mangled builtin name, a dropped blit
## preamble, or an illegal `const char*` member. The fixed code ingests
## and runs.
##
## Run:
##   cd tattletale
##   nim test_crucible_metal
## or directly:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_codegen_regressions.nim

import std/[strutils, unittest]
import workspace/crucible

# ── 1. {.builtin.} plain-name forwarding ─────────────────────────────────────
#
# The name carries two signatures: the builtin scalar proc and the per-lane
# device proc below. The DSL sees the name twice and applies
# its overload handling to the inner builtin call, which must stay plain.

proc rsqrt(x: float32): float32 {.builtin.} = discard

proc rsqrt(dst: ptr UncheckedArray[float32]; src: ptr UncheckedArray[float32]) {.device.} =
  dst[0] = rsqrt(src[0])
  dst[1] = rsqrt(src[1])

const builtinMsl = metal:
  proc builtinKernel(outp: ptr UncheckedArray[float32];
                     inp: ptr UncheckedArray[float32]) {.global.} =
    rsqrt(outp, inp)

# ── 2. gpuAssign blit keeps the LHS preamble ────────────────────────────────
#
# The store's LHS index is a multi-statement expression that the blit
# machinery hoists into a preamble temp, and the assignment references
# that temp. Dropping the preamble leaves the reference undeclared.
# Lane 0 writes frags[0][0] to outBuf[0..1] and frags[1][0]
# to outBuf[8..9] through the fixture's store view.

# Local fixture: the tile-store surface, built from crucible types only.
# The ceramic tile layer's rt_r fp32 store emits the pinned blit shape.
# These stand-ins reproduce it with a C-outer tile of per-lane simdgroup
# fragments and a view whose data base the store indexes.

type
  StoreFrag = object
    ## One atom subtile's per-lane register fragment.
    frag: SimdgroupMatrix[float32, false]
  StoreTile = object
    ## C-outer register tile: 2 col subtiles × 1 row subtile of 8×8 fragments.
    frags: array[2, array[1, StoreFrag]]
  StoreView = object
    ## Global view: the store's data base.
    data: ptr UncheckedArray[float32]

proc storeTile(tile: StoreTile; dst: StoreView) {.device.} =
  ## Writes each fragment's per-lane elements through the view's blitted
  ## index, the LHS preamble the contract pins.
  for m in 0 ..< 2:
    for v in 0 ..< 2:
      dst.data[block:
        let mm = m
        let vv = v
        mm * 8 + vv
      ] = threadElements(tile.frags[m][0].frag, uint32(v))

const storeMsl = metal:
  proc storeKernel(Out: ptr UncheckedArray[float32]) {.global.} =
    var tile: StoreTile
    let lane = thread_index_in_threadgroup
    threadElements(tile.frags[0][0].frag, 0'u32) = float32(lane) + 1.0'f32
    threadElements(tile.frags[0][0].frag, 1'u32) = float32(lane) + 2.0'f32
    threadElements(tile.frags[1][0].frag, 0'u32) = float32(lane) + 3.0'f32
    threadElements(tile.frags[1][0].frag, 1'u32) = float32(lane) + 4.0'f32
    let gl_out = StoreView(data: Out)
    storeTile(tile, gl_out)

# ── 3. gtString field drop in MSL structs ───────────────────────────────────

type Desc = object
  name: string
  scale: float32

const descMsl = metal:
  proc descKernel(outp: ptr UncheckedArray[float32]) {.global.} =
    var d: Desc
    d.scale = 2.0'f32
    outp[0] = d.scale

# ── 4. var-param simdgroup matrix deref at device-fn boundaries ─────────────
#
# The register-surface mma takes `d: var SimdgroupMatrix`. The var param
# lowers to a thread-space pointer at a device-function boundary
# (`thread simdgroup_float8x8* d`), and the MSL intrinsic needs a thread
# reference, so genMatrixRef renders `(*d)`. The plain-array fragment
# (the FMA storage) indexes directly (`a[0]`), never dereferenced.

proc mmaWrap(d: var SimdgroupMatrix[float32, false];
             a: SimdgroupMatrix[float16, false];
             b: SimdgroupMatrix[float16, true]) {.device.} =
  simdgroupMultiplyAccumulate(d, a, b)

proc arrSink(a: var array[4, float32]) {.device.} =
  a[0] = a[1]

proc teBoundary(d: var SimdgroupMatrix[float32, false]; v: uint32) {.device.} =
  # The threadElements accessor through a device-fn boundary is emitted raw:
  # the fragment arg renders as itself, never deref-wrapped, and the shape
  # comes from the resolved overload rather than from deref machinery.
  threadElements(d, v) = 1.0'f32

proc teArrBoundary(a: var array[4, float32]; v: uint32) {.device.} =
  threadElements(a, v) = 2.0'f32

const derefMsl = metal:
  proc derefKernel(D: ptr UncheckedArray[float32]) {.global.} =
    var d: SimdgroupMatrix[float32, false]
    var a: SimdgroupMatrix[float16, false]
    var b: SimdgroupMatrix[float16, true]
    mmaWrap(d, a, b)
    threadElements(d, 0'u32) = 1.0'f32
    D[0] = threadElements(d, 0'u32)
    var arr: array[4, float32]
    arr[1] = 2.0'f32
    arrSink(arr)
    teArrBoundary(arr, 1'u32)
    D[1] = arr[0]

const teBoundaryMsl = metal:
  proc teBoundaryKernel(C: ptr UncheckedArray[float32]) {.global.} =
    var d: SimdgroupMatrix[float32, false]
    teBoundary(d, 0'u32)
    C[0] = threadElements(d, 0'u32)

proc runTest() =
  suite "Metal - tile-layer codegen contracts":
    test "builtin procs keep their plain name from overloaded bodies":
      # The builtin call must emit the plain backend name, not a mangled hash.
      check "= rsqrt(src[0])" in builtinMsl
      check "= rsqrt(src[1])" in builtinMsl
      var engine = bkMetal.init()
      engine.ingest(builtinMsl)
      var inp: array[2, float32] = [4.0'f32, 16.0'f32]
      var res: array[2, float32]
      engine.run("builtinKernel", res, (inp,))
      check res[0] == 0.5'f32
      check res[1] == 0.25'f32

    test "gpuAssign blits append the RHS to the LHS preamble":
      # The store assignment indexes through a blitted LHS temp.
      check "dst.data[_blit_" in storeMsl
      var engine = bkMetal.init()
      engine.ingest(storeMsl)
      var outBuf: array[8 * 16, float32]
      for i in 0 ..< outBuf.len:
        outBuf[i] = -1.0'f32
      engine.run<<(grid: (1, 1), blk: (32, 1))>>("storeKernel", outBuf, ())
      check outBuf[0] == 1.0'f32
      check outBuf[1] == 2.0'f32
      check outBuf[8] == 3.0'f32
      check outBuf[9] == 4.0'f32

    test "MSL structs drop gtString fields":
      # A string field must be dropped, not spelled as an illegal pointer member.
      check "const char*" notin descMsl
      check "struct Desc" in descMsl
      var engine = bkMetal.init()
      engine.ingest(descMsl)
      var res: array[1, float32]
      engine.run("descKernel", res, ())
      check res[0] == 2.0'f32

    test "var-param simdgroup matrices deref at device-fn boundaries":
      # The mma intrinsic takes thread references: a var-param matrix
      # lowers to a thread pointer, so the argument renders `(*d)`.
      # A plain-array var param (the FMA fragment) indexes directly.
      check "simdgroup_multiply_accumulate((*d), a, b, (*d))" in derefMsl
      check "thread simdgroup_float8x8* d" in derefMsl
      check "(*a)" notin derefMsl
      check "a[0] = a[1]" in derefMsl
      var engine = bkMetal.init()
      engine.ingest(derefMsl)
      var res: array[2, float32]
      engine.run("derefKernel", res, ())
      check res[0] == 1.0'f32
      check res[1] == 2.0'f32

    test "threadElements stays raw through a device-fn var-param boundary":
      # The accessor's fragment arg is the plain lvalue, never deref'd:
      # a var-param simdgroup matrix spells `d.thread_elements()[v]`
      # and a var-param per-lane array spells `a[v]`. The deref-wrapped
      # `(*d)[v]` spelling belongs to the mma intrinsic above, not this
      # accessor.
      check "(*d)[v]" notin teBoundaryMsl
      check "d.thread_elements()[v] = 1.0f" in teBoundaryMsl
      check "a[v] = 2.0f" in derefMsl

when isMainModule:
  runTest()

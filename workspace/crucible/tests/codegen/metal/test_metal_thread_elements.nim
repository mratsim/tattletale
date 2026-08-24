## Metal: the `threadElements(frag, vpt)` builtin emits the per-fragment
## MSL access spelling (read and lvalue positions) and indexes the real
## storage on the host.
##
## A simdgroup matrix fragment emits its per-lane element accessor
## (`frag.thread_elements()[vpt]`), a plain per-lane value array emits
## a direct index (`frag[vpt]`). The register-surface field chains
## (`FragmentOf.frag`, a tile's subtile grid) must emit through the same
## accessor, and no fragment access may leak a `.data[` field spelling.
## The fragment types come from crucible builtins (SimdgroupMatrix)
## and the local fixture types defined below.
##
## The device-run section covers the two plain fragment shapes.
## The field-chain kernels pin their accessor spelling with the string
## asserts instead.
##
## Run:
##   cd tattletale
##   nim test_crucible_metal
## or directly:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_thread_elements.nim

import std/[strutils, unittest]
import workspace/crucible

# ── Local fixture: the tile-layer fragment surface, crucible types only ────
# The ceramic tile layer's FragmentOf/RtLeft/atom records define
# the fragment shapes these kernels pin. The stand-ins below reproduce
# them with crucible's SimdgroupMatrix: a per-lane simdgroup fragment
# behind `frag` and a per-lane value array behind the FMA fragment.

type
  StrideOrder = enum
    LayoutLeft, LayoutRight
  MmaAtomKind = enum
    bkGPU_TensorCore
    bk_FMA
  AppleLayout = object
    ## Placeholder layout type of the atom's compile-time layout params.
  MmaAtom[LA, LB, LC] = object
    ## Compile-time MMA atom record: the static generic param selects
    ## the fragment kind below.
    name: string
    mnk: tuple[m, n, k: int]
    kind: MmaAtomKind
  FragmentOf[A: static MmaAtom; T; L: static StrideOrder] = object
    ## Per-lane register fragment of one atom subtile: a simdgroup matrix
    ## for the Apple atoms, a per-lane value array for the FMA atom.
    when A.kind == bkGPU_TensorCore:
      frag: SimdgroupMatrix[T, L == LayoutRight]
    elif A.kind == bk_FMA:
      frag: array[1, T]
    else:
      frag: array[1, T]
  RtLeft[T; R, C: static int; A: static MmaAtom] = object
    ## R-outer register tile: the subtile grid of per-lane fragments.
    frags: array[R div A.mnk.m, array[C div A.mnk.n, FragmentOf[A, T, LayoutLeft]]]

const APPLE_8x8x8_F32 = MmaAtom[AppleLayout, AppleLayout, AppleLayout](
  name: "APPLE_8x8x8_F32", mnk: (m: 8, n: 8, k: 8), kind: bkGPU_TensorCore)
  ## The 8×8×8 simdgroup atom: one per-lane simdgroup matrix per subtile.

const UNIVERSAL_FMA_F32 = MmaAtom[AppleLayout, AppleLayout, AppleLayout](
  name: "UNIVERSAL_FMA_F32", mnk: (m: 1, n: 1, k: 1), kind: bk_FMA)
  ## The 1×1×1 scalar-FMA atom: one per-lane value per thread.

const teMsl = metal:
  proc teKernel(C: ptr UncheckedArray[float32]) {.global.} =
    var d: SimdgroupMatrix[float32, false]
    let v = thread_index_in_threadgroup and 1'u32
    threadElements(d, v) = 1.0'f32
    C[0] = threadElements(d, 0'u32)
    var arr: array[4, float32]
    threadElements(arr, 1'u32) = 2.0'f32
    C[1] = threadElements(arr, 1'u32)

  proc teFieldKernel(C: ptr UncheckedArray[float32]) {.global.} =
    var f: FragmentOf[APPLE_8x8x8_F32, float32, LayoutLeft]
    threadElements(f.frag, 1'u32) = 4.0'f32
    C[0] = threadElements(f.frag, 1'u32)

  proc teTileKernel(C: ptr UncheckedArray[float32]) {.global.} =
    var tile: RtLeft[float32, 8, 8, APPLE_8x8x8_F32]
    threadElements(tile.frags[0][0].frag, 0'u32) = 6.0'f32
    let e = threadElements(tile.frags[0][0].frag, 0'u32)
    C[0] = e

  proc teFmaKernel(C: ptr UncheckedArray[float32]) {.global.} =
    var f: FragmentOf[UNIVERSAL_FMA_F32, float32, LayoutLeft]
    threadElements(f.frag, 0'u32) = 7.0'f32
    C[0] = threadElements(f.frag, 0'u32)

# Device-runnable subset: the two plain fragment shapes only. See the header
# note on why the field-chain kernels stay string-asserted.
const teDeviceMsl = metal:
  proc teDeviceKernel(C: ptr UncheckedArray[float32]) {.global.} =
    var d: SimdgroupMatrix[float32, false]
    threadElements(d, 0'u32) = 1.0'f32
    threadElements(d, 1'u32) = 2.0'f32
    C[0] = threadElements(d, 0'u32)
    var arr: array[4, float32]
    threadElements(arr, 1'u32) = 2.0'f32
    C[1] = threadElements(arr, 1'u32)

proc runTest() =
  suite "Metal - threadElements builtin":
    test "host body indexes per-lane elements, pinned to vpt":
      # Distinct values written through the accessor must land
      # at the exact storage indices: reading back through the fragment's
      # own storage, rather than the accessor, catches a host body
      # that permutes indices.
      var frag: SimdgroupMatrix[float32, false]
      threadElements(frag, 0'u32) = 1.0'f32
      threadElements(frag, 1'u32) = 2.0'f32
      check frag.data[0] == 1.0'f32
      check frag.data[1] == 2.0'f32
      var arr: array[4, float32]
      threadElements(arr, 0'u32) = 1.0'f32
      threadElements(arr, 1'u32) = 2.0'f32
      check arr[0] == 1.0'f32
      check arr[1] == 2.0'f32

    test "MSL spells the per-lane access per fragment shape":
      check "d.thread_elements()[v] = 1.0f" in teMsl
      check "C[0] = d.thread_elements()[0U]" in teMsl
      check "arr[1U] = 2.0f" in teMsl
      check "C[1] = arr[1U]" in teMsl
      check "f.frag.thread_elements()[1U] = 4.0f" in teMsl
      check "C[0] = f.frag.thread_elements()[1U]" in teMsl
      check "tile.frags[0][0].frag.thread_elements()[0U] = 6.0f" in teMsl
      check "C[0] = e" in teMsl
      check "float e = tile.frags[0][0].frag.thread_elements()[0U];" in teMsl
      check "f.frag[0U] = 7.0f" in teMsl
      check "C[0] = f.frag[0U]" in teMsl
      check ".data[" notin teMsl
      check "(*" notin teMsl

    test "emitted MSL compiles and runs on the device":
      var engine = bkMetal.init()
      engine.ingest(teDeviceMsl)
      var res: array[2, float32]
      engine.run("teDeviceKernel", res, ())
      check res[0] == 1.0'f32
      check res[1] == 2.0'f32

when isMainModule:
  runTest()

## Manual GPU test: the Apple simdgroup microtile via Metal, one
## register-level MMA.
##
## C(8×8) = A(8×8)·B(8×8): gemm_atom(tma.atom, cFrag, aFrag, bFrag)
## is one 8×8×8 simdgroup atom, 32 lanes. Fragment gathering is the
## library path (partition_A/B/C, make_fragment_A/B/C, copyFrom/fillWith),
## lowered by the MSL printer to the hardware's simdgroup_load/store
## gathers and simdgroup_multiply_accumulate. Both forms run: the 4-arg
## in-place MMA and the explicit-destination variant with cFrag = 1.0.
##
## Atom is the parameter, APPLE_8x8x8_F32. Tiling is 1×1×1 (single atom).
## Geometry derives inside the driver func. Reference harness lives in
## gemm_test_lib; the tf32 fixtures' bit patterns are byte-identical to
## their float32 readings (integers in -15..15), so the harness passes
## seq[uint32] into float32 kernel buffers unchanged.
##
## Requires an Apple GPU with Metal. Run (from the tattletale root; the
## outdir convention keeps test binaries out of the source tree):
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/manual_apple_gpu_metal.nim \
##     --nimcache:nimcache/tests/manual_apple_gpu_metal.nim \
##     workspace/ceramic/tests/atoms_mma/manual_apple_gpu_metal.nim

import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_constructors
import workspace/ceramic/src/layout_indexing
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/atoms
import workspace/ceramic/src/kernel_gemm/atoms_apple
import workspace/ceramic/src/atoms_mma_partitioning
import workspace/ceramic/src/tensors
import workspace/ceramic/src/ptr_arithmetic
import workspace/ceramic/src/kernel_copy_gpu
import workspace/ceramic/src/kernel_fillwith_gpu
import workspace/ceramic/src/kernel_gemm_epilogues
import workspace/ceramic/src/kernel_gemm_gpu
import workspace/ceramic/tests/gemm/gemm_test_lib
import workspace/crucible

{.experimental: "callOperator".}

const atom = APPLE_8x8x8_F32
const tiled = TiledMma[typeof(atom), typeof(make_layout((1, 1, 1)))](
  atom: atom, threadLayout: make_layout((1, 1, 1)))

func mmaMicrotile(tma: static TiledMma; t: int;
                  C: ptr UncheckedArray[float32];
                  A, B: ptr UncheckedArray[float32]) {.inline.} =
  ## C(8×8) = A(8×8)·B(8×8): one 8×8×8 simdgroup atom, 32 lanes, in-place.
  ## Fragment gathering: partition_A/B/C, fragment registers as simdgroup
  ## fragments (make_fragment_A/B/C), copyFrom/fillWith. All layout
  ## algebra, no loops, no offsets, no raw-addr views.
  const
    M = tma.atom.mnk.m
    N = tma.atom.mnk.n
    K = tma.atom.mnk.k
  let Aview = make_view(A, make_layout((M, K), (1, M)))
  let Bview = make_view(B, make_layout((N, K), (1, N)))
  var Cview = make_view(C, make_layout((M, N), (1, M)))
  let thr = tma.get_slice(t)
  # the partitions give the thread's fragment views (per-thread offsets);
  # the simdgroup gathers read the tile views (all-lane base pointers)
  let tAv = tma.partition_A(thr, Aview)
  let tBv = tma.partition_B(thr, Bview)
  var tCv = tma.partition_C(thr, Cview)
  # fragment registers as simdgroup fragments, one declaration per operand
  var aFrag = make_fragment_A(tma.atom, tAv)
  aFrag.copyFrom(Aview)
  var bFrag = make_fragment_B(tma.atom, tBv)
  bFrag.copyFrom(Bview)
  # the accumulator is a simdgroup fragment: make_fragment_C yields the
  # SimdgroupFragment type gemm_atom's simdgroup overload requires
  var cFrag = make_fragment_C(tma.atom, tCv)
  cFrag.fillWith(0.0'f32)

  gemm_atom(tma.atom, cFrag, aFrag, bFrag)   # one simdgroup_multiply_accumulate

  # identity epilogue: fragment scattered straight to C (simdgroup_store)
  Cview.copyFrom(cFrag)

func mmaMicrotileExplicit(tma: static TiledMma; t: int;
                          C: ptr UncheckedArray[float32];
                          A, B: ptr UncheckedArray[float32]) {.inline.} =
  ## C(8×8) = A(8×8)·B(8×8) + 1: explicit destination, dFrag starts as
  ## a copy of cFrag, then accumulates in place.
  const
    M = tma.atom.mnk.m
    N = tma.atom.mnk.n
    K = tma.atom.mnk.k
  let Aview = make_view(A, make_layout((M, K), (1, M)))
  let Bview = make_view(B, make_layout((N, K), (1, N)))
  var Cview = make_view(C, make_layout((M, N), (1, M)))
  let thr = tma.get_slice(t)
  let tAv = tma.partition_A(thr, Aview)
  let tBv = tma.partition_B(thr, Bview)
  var tCv = tma.partition_C(thr, Cview)
  var aFrag = make_fragment_A(tma.atom, tAv)
  aFrag.copyFrom(Aview)
  var bFrag = make_fragment_B(tma.atom, tBv)
  bFrag.copyFrom(Bview)
  var cFrag = make_fragment_C(tma.atom, tCv)
  cFrag.fillWith(1.0'f32)                        # nonzero accumulator input
  var dFrag = make_fragment_C(tma.atom, tCv)

  dFrag.copyFrom(cFrag)                        # seed the accumulator input
  gemm_atom(tma.atom, dFrag, aFrag, bFrag)   # dFrag = aFrag·bFrag + cFrag

  # identity epilogue: fragment scattered straight to C (simdgroup_store)
  Cview.copyFrom(dFrag)

const kernelCode = metal:
  proc mmaMicrotileKernel(C: ptr UncheckedArray[float32],
                          A, B: ptr UncheckedArray[float32]) {.global.} =
    mmaMicrotile(tiled, int(thread_index_in_simdgroup), C, A, B)

  proc mmaMicrotileExplicitKernel(C: ptr UncheckedArray[float32],
                                  A, B: ptr UncheckedArray[float32]) {.global.} =
    mmaMicrotileExplicit(tiled, int(thread_index_in_simdgroup), C, A, B)

proc runTest() =
  var engine = bkMetal.init()
  engine.ingest(kernelCode)
  testMicrotile(engine, atom, "APPLE")
  # Row-major operands are not exercised here: the harness fixtures are
  # col-major (the fragment algebra's contract, thrfrg_A/B asserted).

when isMainModule:
  runTest()

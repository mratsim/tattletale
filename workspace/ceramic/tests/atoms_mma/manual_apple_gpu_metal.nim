## Manual GPU test: Apple simdgroup microtile.
##
## Requires an Apple GPU. Run from the tattletale root:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/manual_apple_gpu_metal.nim \
##     --nimcache:nimcache/tests/manual_apple_gpu_metal.nim \
##     workspace/ceramic/tests/atoms_mma/manual_apple_gpu_metal.nim

import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_constructors
import workspace/ceramic/src/layout_indexing
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/hardware/h_configgen
import workspace/ceramic/src/hardware/h_registry
import workspace/ceramic/src/hardware/h_properties

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
  ## One 8×8×8 simdgroup atom (C = A·B), in-place, via the library path.
  const
    M = tma.atom.getM()
    N = tma.atom.getN()
    K = tma.atom.getK()
  let Aview = make_view(A, make_layout((M, K), (1, M)))
  let Bview = make_view(B, make_layout((N, K), (1, N)))
  var Cview = make_view(C, make_layout((M, N), (1, M)))
  let thr = tma.get_slice(t)
  let tAv = tma.partition_A(thr, Aview)
  let tBv = tma.partition_B(thr, Bview)
  var tCv = tma.partition_C(thr, Cview)
  var aFrag = make_fragment_A(tma.atom, tAv)
  simdgroupLoad(aFrag, Aview.data, uint32(M), 0'u32, true)
  var bFrag = make_fragment_B(tma.atom, tBv)
  simdgroupLoad(bFrag, Bview.data, uint32(N), 0'u32, false)
  # make_fragment_C yields the SimdgroupMatrix type gemm_atom's simdgroup overload requires.
  var cFrag = make_fragment_C(tma.atom, tCv)
  cFrag.fillWith(0.0'f32)

  gemm_atom(tma.atom, cFrag, aFrag, bFrag)   # one simdgroup_multiply_accumulate

  simdgroupStore(cFrag, Cview.data, uint32(M), 0'u32, true)

func mmaMicrotileExplicit(tma: static TiledMma; t: int;
                          C: ptr UncheckedArray[float32];
                          A, B: ptr UncheckedArray[float32]) {.inline.} =
  ## Same atom, explicit destination (C = A·B + cFrag, cFrag = 1.0).
  const
    M = tma.atom.getM()
    N = tma.atom.getN()
    K = tma.atom.getK()
  let Aview = make_view(A, make_layout((M, K), (1, M)))
  let Bview = make_view(B, make_layout((N, K), (1, N)))
  var Cview = make_view(C, make_layout((M, N), (1, M)))
  let thr = tma.get_slice(t)
  let tAv = tma.partition_A(thr, Aview)
  let tBv = tma.partition_B(thr, Bview)
  var tCv = tma.partition_C(thr, Cview)
  var aFrag = make_fragment_A(tma.atom, tAv)
  simdgroupLoad(aFrag, Aview.data, uint32(M), 0'u32, true)
  var bFrag = make_fragment_B(tma.atom, tBv)
  simdgroupLoad(bFrag, Bview.data, uint32(N), 0'u32, false)
  var cFrag = make_fragment_C(tma.atom, tCv)
  cFrag.fillWith(1.0'f32)                        # nonzero accumulator input
  var dFrag = make_fragment_C(tma.atom, tCv)

  dFrag = cFrag
  gemm_atom(tma.atom, dFrag, aFrag, bFrag)   # dFrag = aFrag·bFrag + cFrag

  simdgroupStore(dFrag, Cview.data, uint32(M), 0'u32, true)

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

when isMainModule:
  runTest()

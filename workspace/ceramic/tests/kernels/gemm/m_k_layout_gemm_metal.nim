## gemm_tile on Metal: one APPLE_8x8x8_F32 simdgroup atom, C(8,8) = A(8,16) * B(8,16),
## per-lane register tiles (V, (MR, KR)) gathered through the atom layouts, 16 trials.
##
## Run:
##   nim c -r -d:metal --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/ceramic/tests/kernels/gemm/m_k_layout_gemm_metal.nim (from tattletale)

import std/[math, random, strformat]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/src/hardware/h_mma_configgen
import workspace/ceramic/src/hardware/h_mma_properties
import workspace/ceramic/src/kernels/k_layout_gemm
import workspace/ceramic/src/kernels/k_layout_fillwith_gpu

const atom = APPLE_8x8x8_F32

const kSlices = 2
const M = atom.getM()
const N = atom.getN()
const K = atom.getK()
const vA = atom.valuesPerThread(opA)
const vB = atom.valuesPerThread(opB)
const vC = atom.valuesPerThread(opC)
const blockElems = M * K          # one k slice of the A or B operand, 64 elements
const aLayout = atom.getLayoutA()
const bLayout = atom.getLayoutB()
const cLayout = atom.getLayoutC()

proc gemmTileFrag(C, A, B: ptr UncheckedArray[float32]) {.inline.} =
  ## One simdgroup: per-lane register tiles (V, (MR, KR)), gemm_tile accumulates,
  ## the fragments store back through the atom's layouts.
  let tid = thread_index_in_threadgroup
  var rA = make_tensor(float32, makeIntTuple((vA, (1, kSlices))))
  var rB = make_tensor(float32, makeIntTuple((vB, (1, kSlices))))
  var rC = make_tensor(float32, makeIntTuple((vC, (1, 1))))
  for v in 0 ..< vA:
    for s in 0 ..< kSlices:
      rA[v, (0, s)] = A[crd2idx(aLayout, (tid, v)) + s * blockElems]
      rB[v, (0, s)] = B[crd2idx(bLayout, (tid, v)) + s * blockElems]
  rC.fillWith(0.0'f32)
  atom.gemm_tile(rC, rA, rB)
  for v in 0 ..< vC:
    C[crd2idx(cLayout, (tid, v))] = rC[v, (0, 0)]

const kernelCode = metal:
  proc gemmTileKernel(C, A, B: ptr UncheckedArray[float32]) {.global.} =
    gemmTileFrag(C, A, B)

proc runTest() =
  var engine = bkMetal.init()
  engine.ingest(kernelCode)
  var rng = initRand(0xC0FFEE)
  var allOk = true
  for trial in 0 ..< 16:
    # host storage matches the kernel's operand blocks: A (M, kSlices*K):(1, M),
    # B (N, kSlices*K):(1, N), C (M, N):(1, M), all col-major
    var A = newSeq[float32](M * kSlices * K)
    var B = newSeq[float32](N * kSlices * K)
    for m in 0 ..< M:
      for k in 0 ..< kSlices * K:
        A[m + k * M] = rng.rand(-2.0 .. 2.0)
    for n in 0 ..< N:
      for k in 0 ..< kSlices * K:
        B[n + k * N] = rng.rand(-2.0 .. 2.0)
    var refC = newSeq[float32](M * N)
    for m in 0 ..< M:
      for n in 0 ..< N:
        var acc = 0.0'f32
        for k in 0 ..< kSlices * K:
          acc += A[m + k * M] * B[n + k * N]
        refC[m + n * M] = acc
    var gpuC = newSeq[float32](M * N)
    engine.run << (grid: (1, 1), blk: (32, 1)) >>
      ("gemmTileKernel", gpuC, (A, B))
    var worst = 0.0'f32
    for m in 0 ..< M:
      for n in 0 ..< N:
        worst = max(worst, abs(gpuC[m + n * M] - refC[m + n * M]))
    let pass = worst < 1e-5
    allOk = allOk and pass
    echo &"  gemm_tile Apple trial {trial}: ",
         (if pass: &"PASS worst |d| = {worst}" else: &"FAIL worst |d| = {worst}")
  if not allOk: quit 1

when isMainModule:
  runTest()

## Phase 1b GPU test — on-hardware fragment dump.
##
## A 2×2 tiled tf32 mma (128 threads) computes each thread's A/B/C fragment
## coords on the GPU and dumps them; the host compares against the host-side
## partition_A/B/C (atoms_mma_partitioning). Validates the get_slice
## decomposition + layout offsets + col-major decomposition as emitted
## through the DSL, on the actual hardware.
##
## Kernel constants are derived from SM80_16x8x8_F32TF32TF32F32_TN:
##   atom mnk (16, 8, 8), T=32, V = 4/2/4, thread tiling (ThrM, ThrN, ThrK)
##   = (2, 2, 1).
## Run with: nim cpp -r workspace/ceramic/experiments/wip_mma_gemm/test_fragment_dump_gpu.nim

import std/strformat
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_constructors
import workspace/ceramic/src/layout_indexing
import workspace/ceramic/src/atoms
import workspace/ceramic/src/atoms_nvidia
import workspace/ceramic/src/atoms_mma_partitioning
import workspace/crucible/src/codegen/nvrtc

const kernelCode = cuda:
  proc dumpFragments(
      outA, outB, outC: ptr UncheckedArray[int32]) {.global.} =
    ## Each thread writes its fragment (row, col) pairs.
    ## A: 4 vals, B: 2 vals, C: 4 vals — flat layout
    ##   outX[t * Vmax + v * 2 + 0] = row, + 1 = col
    let t = int(threadIdx.x)
    let
      T = 32          # threads per atom (from atom.threadCount)
      mAtom = 16      # atom.mnk.m
      nAtom = 8       # atom.mnk.n
      kAtom = 8       # atom.mnk.k
      thrM = 2
      thrN = 2
      thrK = 1

    # get_slice: flat = tv + T·(tm + ThrM·(tn + ThrN·tk))
    let atomIdx = t div T
    let tm = atomIdx mod thrM
    let tn = (atomIdx div thrM) mod thrN
    let tk = atomIdx div (thrM * thrN)
    let localT = t mod T

    # A fragment: atom tile over (tm, tk)
    let aLayout = make_layout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    for v in 0 .. 3:
      let off = crd2idx(aLayout, localT + T * v)
      let lm = off mod mAtom
      let lk = off div mAtom
      outA[t * 8 + v * 2 + 0] = int32(tm * mAtom + lm)
      outA[t * 8 + v * 2 + 1] = int32(tk * kAtom + lk)

    # B fragment: atom tile over (tn, tk)
    let bLayout = make_layout(((4, 8), 2), ((8, 1), 32))
    for v in 0 .. 1:
      let off = crd2idx(bLayout, localT + T * v)
      let ln = off mod nAtom
      let lk = off div nAtom
      outB[t * 4 + v * 2 + 0] = int32(tn * nAtom + ln)
      outB[t * 4 + v * 2 + 1] = int32(tk * kAtom + lk)

    # C fragment: atom tile over (tm, tn)
    let cLayout = make_layout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    for v in 0 .. 3:
      let off = crd2idx(cLayout, localT + T * v)
      let lm = off mod mAtom
      let ln = off div mAtom
      outC[t * 8 + v * 2 + 0] = int32(tm * mAtom + lm)
      outC[t * 8 + v * 2 + 1] = int32(tn * nAtom + ln)

when isMainModule:
  const atom = SM80_16x8x8_F32TF32TF32F32_TN
  const tiled2 = TiledMma[typeof(atom), typeof(make_layout((2, 2, 1)))](
    atom: atom, threadLayout: make_layout((2, 2, 1)))

  var bufA = newSeq[int32](128 * 8)
  var bufB = newSeq[int32](128 * 4)
  var bufC = newSeq[int32](128 * 8)
  var nv = initNvrtc(kernelCode)
  nv.compile()
  nv.getPtx()
  # res = all three buffers: the runtime copies back ONLY the res tuple
  # (inputs are copy-in-only — the dump kernel WRITES outB/outC, so they
  # must be res, not inputs)
  nv.execute("dumpFragments", dim3(1), dim3(128), (bufA, bufB, bufC), ())

  # host-side expectation
  for t in 0 ..< 128:
    let fA = tiled2.partitionA((32, 8), t)
    for v in 0 ..< 4:
      let gRow = bufA[t * 8 + v * 2 + 0]
      let gCol = bufA[t * 8 + v * 2 + 1]
      doAssert gRow == int32(fA[v].row), &"A t{t} v{v}: GPU row {gRow} != host {fA[v].row}"
      doAssert gCol == int32(fA[v].col), &"A t{t} v{v}: GPU col {gCol} != host {fA[v].col}"
    let fB = tiled2.partitionB((16, 8), t)
    for v in 0 ..< 2:
      doAssert bufB[t * 4 + v * 2 + 0] == int32(fB[v].row), &"B t{t} v{v}"
      doAssert bufB[t * 4 + v * 2 + 1] == int32(fB[v].col), &"B t{t} v{v}"
    let fC = tiled2.partitionC((32, 16), t)
    for v in 0 ..< 4:
      doAssert bufC[t * 8 + v * 2 + 0] == int32(fC[v].row), &"C t{t} v{v}"
      doAssert bufC[t * 8 + v * 2 + 1] == int32(fC[v].col), &"C t{t} v{v}"

  echo "  OK — on-GPU fragment dump matches host partition (2×2 tiled, 128 threads)"

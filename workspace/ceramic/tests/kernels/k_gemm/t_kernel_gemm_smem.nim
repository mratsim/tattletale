# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tests for k_tile_gemm_smem:
##
## - bank patterns of the stage store and of the mma fragment loads.
##   Pure functions, no GPU.
## - staged contents against a host reference, including zero-fill
##   for ragged bounds and the k-tail chunk's elementwise prefix copy
## - gemm_smem against the fp32 host reference over the highlight
##   and fuzz case lists, one operand pass per bf16 and f16
##
## Run:
##   nim c -r --outdir:build/tests \
##     workspace/ceramic/tests/kernels/k_gemm/t_kernel_gemm_smem.nim

import std/[random, times]
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/src/hardware/h_mma_properties
import workspace/ceramic/src/kernels/k_tile_gemm_smem
import workspace/crucible
import workspace/math
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/hardware/h_mma_configgen
import workspace/ceramic/src/hardware/h_mma_dispatch
import workspace/ceramic/src/hardware/h_copy_dispatch
import workspace/ceramic/src/kernels/k_layout_copy_gpu
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/tests/kernels/k_gemm/kernel_gemm_template
import workspace/ceramic/src/kernels/k_layout_fillwith_cpu
import std/[strformat, strutils]

# ═══════════════════════════════════════════════════════════════════════
#  Bank-pattern analysis (pure, host side)
# ═══════════════════════════════════════════════════════════════════════

const StageK = 64
const StageRows = 64

proc chunkCoord(tid, i: int): (int, int) =
  ## Staging chunk (kc, rc) that thread `tid` copies at round `i`,
  ## 64 staging threads, chunks flat-numbered c = tid + i*64 over
  ## the (chunksPerRow, stageRows) chunk grid
  let c = tid + i * 64
  (c mod stageChunksPerRow(StageK), c div stageChunksPerRow(StageK))

proc testStoreBanks() =
  ## One row's 8 staged chunks stay 16-byte aligned and span 4
  ## consecutive words each, tiling that row's 32 bank words with no
  ## overlap inside the phase that writes them
  for phase in 0 ..< 8:
    var firsts: array[8, int]
    for t in 0 ..< 8:
      let (kc, rc) = chunkCoord(phase * 8 + t, 0)
      firsts[t] = stageBankWord(rc, kc * 8, StageK)
    for a in 0 ..< 8:
      for b in a + 1 ..< 8:
        # one chunk spans 4 consecutive words, the 8 chunks of a phase
        # must tile the 32-word bank range without overlap
        doAssert firsts[a] div 4 != firsts[b] div 4 or
          (firsts[a] mod 4) + 4 <= firsts[b] mod 4 or
          (firsts[b] mod 4) + 4 <= firsts[a] mod 4,
          "store phase " & $phase & ": chunks " & $a & "," & $b &
          " share banks (" & $firsts[a] & " vs " & $firsts[b] & ")"
  echo "  OK (stage store bank pattern, " & $StageK & "x" & $StageRows & ")"

proc fragmentTableA(atomMap: Layout): array[32, array[16, (int, int)]] =
  ## (row, k) of every (lane, value) A-fragment slot of one k16 step.
  ## The atom map sends (lane, value) to a col-major offset inside
  ## the (16, 16) atom cell, and the M repetition adds 16 rows.
  for t0 in 0 ..< 4:
    for t1 in 0 ..< 8:
      for v0 in 0 ..< 2:
        for v1 in 0 ..< 2:
          for v2 in 0 ..< 2:
            for r0 in 0 ..< 2:
              let off = int(crd2idx(atomMap, ((t0, t1), (v0, v1, v2))))
              result[t0 + 4*t1][v0 + 2*v1 + 4*v2 + 8*r0] =
                (16*r0 + off mod 16, off div 16)

proc fragmentTableB(atomMap: Layout): array[32, array[16, (int, int)]] =
  ## (col, k) of every (lane, value) B-fragment slot of one k16 step.
  ## The atom map sends (lane, value) to a col-major offset inside
  ## the (8, 16) atom cell, and the N repetition adds 8 columns.
  for t0 in 0 ..< 4:
    for t1 in 0 ..< 8:
      for v0 in 0 ..< 2:
        for v1 in 0 ..< 2:
          for r0 in 0 ..< 4:
            let off = int(crd2idx(atomMap, ((t0, t1), (v0, v1))))
            result[t0 + 4*t1][v0 + 2*v1 + 4*r0] =
              (8*r0 + off mod 8, off div 8)

proc testFragmentLoadBanks() =
  ## For every fragment register, one per value pair with adjacent
  ## values sharing a 4-byte word, the 32 lanes read bank-distinct
  ## words or else the same word when the value broadcasts.
  ##
  ## The check covers both warp rows, every k16 step of the stage,
  ## and the padded layout.
  # Host-side analysis, no DSL target block: the warp-tile mma
  # resolves against the SM80 bf16 atom directly
  const atom = SM80_16x8x16_F32BF16BF16F32_TN
  const coordsA = atom.getLayoutA().fragmentTableA()
  const coordsB = atom.getLayoutB().fragmentTableB()
  for warpRow in 0 ..< 2:
    for warpCol in 0 ..< 2:
      for kStep in 0 ..< StageK div 16:
        # A: 16 values per lane per step = 8 register pairs
        for pair in 0 ..< 8:
          var words: array[32, int]
          for t in 0 ..< 32:
            let (m, k) = coordsA[t][pair * 2]
            words[t] = stageBankWord(warpRow * 32 + m, kStep * 16 + k, StageK)
            let (_, kHi) = coordsA[t][pair * 2 + 1]
            doAssert words[t] == stageBankWord(warpRow * 32 + m, kStep * 16 + kHi, StageK),
              "A pair " & $pair & " splits its 4-byte word"
          for a in 0 ..< 32:
            for b in a + 1 ..< 32:
              doAssert (words[a] mod 32) != (words[b] mod 32) or words[a] == words[b],
                "A load pair " & $pair & " bank conflict: lanes " & $a & "," & $b &
                " words " & $words[a] & "," & $words[b]
        # B: 16 values per lane per step = 8 register pairs
        for pair in 0 ..< 8:
          var words: array[32, int]
          for t in 0 ..< 32:
            let (n, k) = coordsB[t][pair * 2]
            words[t] = stageBankWord(warpCol * 32 + n, kStep * 16 + k, StageK)
          for a in 0 ..< 32:
            for b in a + 1 ..< 32:
              doAssert (words[a] mod 32) != (words[b] mod 32) or words[a] == words[b],
                "B load pair " & $pair & " bank conflict: lanes " & $a & "," & $b &
                " words " & $words[a] & "," & $words[b]
  echo "  OK (fragment load bank pattern, warp grid 2x2, all k steps)"

# ═══════════════════════════════════════════════════════════════════════
#  Staged contents vs host reference (CUDA)
# ═══════════════════════════════════════════════════════════════════════

const stageWords = 64 * stageRowStride(64) div 2

proc stageProbeKernel(
    dst: TensorView[uint32, tuple[a: int, b: int], tuple[a: int, b: int]],
    aSrc: TensorView[bfloat16, tuple[a: int, b: int], tuple[a: int, b: int]],
    bSrc: TensorView[bfloat16, tuple[a: int, b: int], tuple[a: int, b: int]]) =
  ## Stages (64, 64) tiles of `aSrc` and `bSrc` through the staging
  ## path of gemm_smem, then drains both stages back out as words
  ## into `dst`: the A stage at [0, stageWords), the B stage after it.
  var
    stageA {.smem.}: array[2 * stageWords + 8, uint32]
    stageB {.smem.}: array[2 * stageWords + 8, uint32]
  let thrID = getLocalThreadID()
  let aStage = cast[ptr UncheckedArray[uint32]](
    (cast[uint64](addr stageA[0]) + 15) and (not uint64(15)))
  let bStage = cast[ptr UncheckedArray[uint32]](
    (cast[uint64](addr stageB[0]) + 15) and (not uint64(15)))
  stageSmemTile(aStage, aSrc, thrID, 64, 64, 0, 0, 0,
                aSrc.shape[0], aSrc.shape[1])
  stageSmemTile(bStage, bSrc, thrID - 64, 64, 64, 0, 0, 0,
                bSrc.shape[0], bSrc.shape[1])
  commit_group(getCopyAsyncAtom(uint32))
  wait_group(getCopyAsyncAtom(uint32), 0)
  threadgroup_barrier()
  # Drain the aligned stage, the alignment slack stays behind.
  let slackA = int((cast[uint64](aStage) - cast[uint64](addr stageA[0])) div 4)
  let slackB = int((cast[uint64](bStage) - cast[uint64](addr stageB[0])) div 4)
  for i in 0 ..< stageWords div 128:
    let w = thrID + i * 128
    dst.data[w] = aStage[slackA + w]
    dst.data[stageWords + w] = bStage[slackB + w]

proc stageWordRef(src: HostTensor[bfloat16, tuple[a: int, b: int], tuple[a: int, b: int]], w: int): uint32 =
  ## Stage word `w` that the staging must produce: elements 2w, 2w+1 of the row-major padded stage, real data in bounds else zero
  let stageWidth = stageRowStride(64)
  var lo, hi: uint16
  for (off, bits) in [(0, addr lo), (1, addr hi)]:
    let elem = 2 * w + off
    let (row, k) = (elem div stageWidth, elem mod stageWidth)
    if row < src.layout.shape[0] and k < src.layout.shape[1]:
      bits[] = cast[uint16](src.data[src.layout.crd2idx((row, k))])
    else:
      bits[] = 0'u16
  uint32(lo) or (uint32(hi) shl 16)

proc checkStage[E: HwEngine](engine: E, aShape: (int, int), bShape: (int, int)) =
  var rng = initRand(2026_11_02)
  var
    aHost = rng.randomHostTensor(bfloat16, aShape, (aShape[1], 1))
    bHost = rng.randomHostTensor(bfloat16, bShape, (bShape[1], 1))
    # One flat word buffer, the A stage then the B stage, 2-D metadata keeps the tensor utilities happy.
    outHost = makeHostTensor(uint32, (2 * stageWords, 1), (1, 2 * stageWords))
  let
    aDev = engine.bridge(aHost)
    bDev = engine.bridge(bHost)
    outDev = engine.bridge(outHost)
  engine.launchDeviceFn stageProbeKernel(outDev, aDev, bDev) do:
    grid = (1, 1)
    blk = (128, 1)
  engine.copyToHostFromDevice(addr outHost.data[0], outDev.data, outHost.len)
  for w in 0 ..< stageWords:
    doAssert outHost.data[w] == stageWordRef(aHost, w),
      "A stage word " & $w & ": got " & toHex(outHost.data[w]) & " want " &
      toHex(stageWordRef(aHost, w)) & " (shape " & $aShape & ")"
    doAssert outHost.data[stageWords + w] == stageWordRef(bHost, w),
      "B stage word " & $w & ": got " & toHex(outHost.data[stageWords + w]) &
      " want " & toHex(stageWordRef(bHost, w)) & " (shape " & $bShape & ")"
  echo "  OK (staged contents, A " & $aShape & " B " & $bShape & ")"

proc testStaging[E: HwEngine](engine: E) =
  checkStage(engine, (64, 64), (64, 64))
  checkStage(engine, (50, 37), (45, 64)) # ragged rows, ragged k with tail chunk
  checkStage(engine, (1, 1), (64, 9))    # single row and two tail chunks

# ═══════════════════════════════════════════════════════════════════════
#  gemm_smem vs fp32 host reference (CUDA)
# ═══════════════════════════════════════════════════════════════════════

const GemmSmemHighlights = [
  # square in-bounds tiles, both stride orders, beta accumulation
  (alpha: 1.0'f32, A: ((64, 64), (64, 1)), B: ((64, 64), (64, 1)), beta: 0.0'f32, C: ((64, 64), (64, 1))),
  (alpha: 2.0'f32, A: ((64, 64), (1, 64)), B: ((64, 64), (1, 64)), beta: 1.0'f32, C: ((64, 64), (1, 64))),
  # multi-block tile grids, grid-stride covered
  (alpha: 1.0'f32, A: ((100, 200), (200, 1)), B: ((150, 200), (1, 150)), beta: 0.5'f32, C: ((100, 150), (150, 1))),
  (alpha: 2.0'f32, A: ((128, 128), (128, 1)), B: ((128, 128), (1, 128)), beta: 1.0'f32, C: ((128, 128), (128, 1))),
  # ragged bounds: block-tile and stage-tail predication, odd strides
  (alpha: 1.0'f32, A: ((65, 62), (62, 1)), B: ((63, 62), (63, 1)), beta: 2.0'f32, C: ((65, 63), (63, 1))),
  (alpha: 1.0'f32, A: ((17, 23), (23, 1)), B: ((19, 23), (1, 19)), beta: 1.0'f32, C: ((17, 19), (19, 1))),
  (alpha: 1.0'f32, A: ((1, 1), (1, 1)), B: ((1, 1), (1, 1)), beta: 0.0'f32, C: ((1, 1), (1, 1)))]

proc checkGemmSmem[E: HwEngine, TIn](
    rng: var Rand, engine: E, _: typedesc[TIn], cs: GemmCase) =
  let (aShape, aStrides) = cs.A
  let (bShape, bStrides) = cs.B
  let (cShape, cStrides) = cs.C
  let (M, N, K) = (aShape[0], bShape[0], aShape[1])
  var
    aHost = rng.randomHostTensor(TIn, aShape, aStrides)
    bHost = rng.randomHostTensor(TIn, bShape, bStrides)
    cHost = rng.randomHostTensor(float32, cShape, cStrides)
    dHost = makeHostTensor(float32, cShape, cStrides)
    refHost = makeHostTensor(float32, (M, N), (N, 1))
  var dView = dHost.view()
  fillWith_cpu(dView, -12345.0'f32) # every D cell must be written
  let
    aDev = engine.bridge(aHost)
    bDev = engine.bridge(bHost)
    cDev = engine.bridge(cHost)
    dDev = engine.bridge(dHost)
  # one (64, 64) block tile per launch block, 128 threads, the grid runs
  # 1..2x the tile grid to stress the grid-stride loop and the barrier
  # that guards the stage reuse between block tiles
  engine.launchDeviceFn gemm_smem(tiledMma(float32, TIn, TIn, 32, 32, 64),
      tiledMma(float32, TIn, TIn, 32, 32, 16),
      dDev, cs.alpha, aDev, bDev, cs.beta, cDev) do:
    grid = (rng.rand(1 .. 2 * ceil_div(M, 64)),
            rng.rand(1 .. 2 * ceil_div(N, 64)))
    blk = (128, 1)
  engine.copyToHostFromDevice(addr dHost.data[0], dDev.data, dHost.len)
  gemm_ref(refHost, cs.alpha, aHost, bHost, cs.beta, cHost)
  assertAllClose(dHost.view(), refHost.view(), rtol = 1e-5, abstol = 1e-3,
    msg = "gemm_smem " & $M & "x" & $N & "x" & $K & " strides " & $aStrides &
          " " & $bStrides & " " & $cStrides)

proc runGemmSmemChecks[E: HwEngine, TIn](
    rng: var Rand, engine: E, _: typedesc[TIn],
    cases: openArray[GemmCase], Iters: static int) =
  for iter in 0 ..< Iters:
    for cs in cases:
      rng.checkGemmSmem(engine, TIn, cs)

proc runGemmSmemFuzz[E: HwEngine, TIn](
    rng: var Rand, engine: E, _: typedesc[TIn], cases: openArray[GemmCase]) =
  for cs in cases:
    rng.checkGemmSmem(engine, TIn, cs)

# ═══════════════════════════════════════════════════════════════════════
#  The suite
# ═══════════════════════════════════════════════════════════════════════

proc main =
  echo "t_kernel_gemm_smem"
  testStoreBanks()
  testFragmentLoadBanks()

  var engine = newEngine(runCuda)
  testStaging(engine)

  var rng = initRand(0xC16D_02B7)
  rng.runGemmSmemChecks(engine, bfloat16, GemmSmemHighlights, Iters = 2)
  rng.runGemmSmemChecks(engine, float16, GemmSmemHighlights, Iters = 2)
  rng.runGemmSmemFuzz(engine, bfloat16, genGemmCases(
    (64, 64, 64), (100, 150, 200), (37, 61, 45), (128, 128, 128)))
  rng.runGemmSmemFuzz(engine, float16, genGemmCases(
    (64, 64, 64), (100, 150, 200), (37, 61, 45), (128, 128, 128)))

when isMainModule:
  main()

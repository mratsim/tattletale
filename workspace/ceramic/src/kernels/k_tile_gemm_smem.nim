# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Shared-memory staging for the block-tiled matmul.
##
## - A and B stage per block in shared memory, double-buffered over
##   the reduction: async copy chunks fill one stage while the warps
##   consume the other, with `commit_group` and `wait_group`
##   pipelining the transfers
## - stage rows keep k-contiguous elements padded by one 16-byte unit.
##   Every staged chunk stays 16-byte aligned, and the padded row
##   stride maps each warp's fragment load to its own bank group.
## - the mma fragments load from the stage through the atom
##   thread->value map. The register tiles share that map.
## - ragged M, N, K go through coordinate predication with zero-fill
##   for out-of-bounds chunks and an elementwise in-bounds prefix
##   copy for the k-tail chunk, so no edge tile needs
##   a scalar pass

import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/hardware/h_mma_configgen
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/src/hardware/h_mma_properties
import workspace/ceramic/src/hardware/h_mma_dispatch
import workspace/ceramic/src/kernels/k_layout_gemm
import workspace/ceramic/src/kernels/k_layout_copy_gpu
import workspace/ceramic/src/hardware/h_copy_dispatch
import workspace/ceramic/src/kernels/k_layout_fillwith_gpu

# ═══════════════════════════════════════════════════════════════════════
#  Warp grid
# ═══════════════════════════════════════════════════════════════════════

const
  threadsPerWarp* = 32
    ## Threads per warp
  warpsPerDim* = 2
    ## Warp grid dimension per block tile, one (BM, BN) tile is covered
    ## by a warpsPerDim x warpsPerDim grid of warps.
    ## Warp (row, col) takes the quadrant
    ## (warpsPerDim·mTile + row, warpsPerDim·nTile + col)
  warpsPerBlock* = warpsPerDim * warpsPerDim
    ## Warps per launch block
  stageThreads* = warpsPerBlock * threadsPerWarp div 2
    ## Staging threads per stageSmemTile call, the block's warps split
    ## in two halves, one per operand

# ═══════════════════════════════════════════════════════════════════════
#  Stage geometry
# ═══════════════════════════════════════════════════════════════════════

const stagePadElements* = 8
  ## Stage rows pad by one 16-byte unit, 8 bf16/f16 elements.
  ##
  ## - every staged chunk stays 16-byte aligned under the padded stride
  ## - the padded row stride maps each warp's fragment load to banks
  ##   of its own group per row
  ##
  ## The cp.async store side is a DMA write whose banking is not
  ## instruction-serialized, the fragment loads are.

func stageRowStride*(stageK: static int): static int {.inline.} =
  ## Elements between two stage rows, `stageK` padded by one 16-byte unit
  stageK + stagePadElements

func stageChunksPerRow*(stageK: static int): static int {.inline.} =
  ## 16-byte chunks along one stage row, `stageK` div 8
  stageK div 8

func stageBankWord*(row, k: int, stageK: static int): int {.inline.} =
  ## 4-byte word index of stage element (row, k). Bank analysis uses
  ## it directly: bank = word mod 32
  (row * stageRowStride(stageK) + k) div 2

# ═══════════════════════════════════════════════════════════════════════
#  Fragment loads
# ═══════════════════════════════════════════════════════════════════════

template loadFragTile*(dst: untyped, src: TensorView, tvm: static ThreadValueMap, threadIdx: int, tileID: (int, int)) =
  ## Loads one thread->value tile of `src` into the register tile
  ## `dst`, the tile.load fast path over a shared-memory stage.
  ##
  ## Contract:
  ## - the read is always in bounds. The staging covers every slice
  ##   coordinate with real data or with the zero the mma consumes.
  ##   The guarded fallback of tile.load has no work here
  ## - the source view points into threadgroup storage, so the loader
  ##   stays a template: as a proc, the `TensorView` parameter fixes
  ##   its memory space to `caDevice` and rejects the smem-backed view
  ##
  ## Pure layout math and field reads only, every step inlines.
  let lay = src.getLayout()
  let tiled = lay.logical_divide(tvm.layout)
  let thrCrd = idx2crd(tvm.getThreadPartitioning(), threadIdx)
  let thrSel = (((thrCrd, _), _), tileID)
  let srcReg = tiled.slice(thrSel)
  for i in 0 ..< srcReg.size():
    dst[i] = src.data[crd2idx(srcReg, i)]

# ═══════════════════════════════════════════════════════════════════════
#  Staging
# ═══════════════════════════════════════════════════════════════════════

template stageSmemTile*(stageWords: untyped, src: TensorView, tid: int, stageRows: static int, stageK: static int, wordBase, rowOrigin, kOrigin, rowBound, kBound: int) =
  ## Stages one (stageRows, stageK) tile of `src` into the word-backed
  ## shared-memory `stageWords` array at word offset `wordBase`.
  ##
  ## Args:
  ## - stageRows, stageK: the tile shape, 64 staging threads copy it,
  ##   8 chunks of 8 elements per staging thread
  ## - rowOrigin, kOrigin: the global coordinates of the tile origin
  ## - rowBound, kBound: the tensor bounds.
  ##   Out-of-bounds chunks zero-fill through the src-size operand
  ##   that the async copy applies
  ##
  ## Stage element (row, k) reads `src` at global coordinate
  ## (rowOrigin + row, kOrigin + k) through the operand's layout.
  ##
  ## - a fully in-bounds, k-contiguous (k stride 1) and 16-byte-aligned
  ##   chunk moves as one 16-byte async copy
  ## - a k-strided or misaligned chunk copies elementwise via crd2idx
  const
    chunkElems = 8
    chunkWords = 4
    chunksPerRow = stageChunksPerRow(stageK)
    chunksPerThread = (stageRows * chunksPerRow) div stageThreads
  static:
    assert sizeof(src.data[0]) == 2,
      "stageSmemTile: the chunk geometry fixes 2-byte elements (" &
      $chunkElems & " per 16-byte chunk), got " &
      $sizeof(src.data[0]) & "-byte elements"
    doAssert stageK mod chunkElems == 0,
      "stageSmemTile: stageK " & $stageK & " does not divide into whole " &
      $chunkElems & "-element chunks, the row tails would never stage"
    doAssert (stageRows * chunksPerRow) mod stageThreads == 0,
      "stageSmemTile: " & $(stageRows * chunksPerRow) & " chunks do not " &
      "divide evenly over " & $stageThreads & " staging threads, the tail " &
      "chunks would drop"
  let
    elems = cast[ptr UncheckedArray[typeof(src.data[0])]](stageWords)
    zeroVal = zero(typeof(src.data[0]))
    lay = src.getLayout()
    kStride = lay.stride[1]
  const atom = getCopyAsyncAtom(uint32)
  # Participation guard: both staging calls run on all warpsPerBlock threads,
  # one call with the thread id and one with the id shifted down
  # stageThreads lanes, so exactly one call matches a staging thread
  # id and the other skips whole warps (no intra-warp divergence).
  if tid >= 0 and tid < stageThreads:
    for i in 0 ..< chunksPerThread:
      let c = tid + i * stageThreads
      let kc = c mod chunksPerRow
      let rc = c div chunksPerRow
      let elemOff = rc * stageRowStride(stageK) + kc * chunkElems
      let kIn = max(min(kBound - (kOrigin + kc * chunkElems), chunkElems), 0)
      let rowIn = rowOrigin + rc < rowBound
      let dstChunk = make_view(addr stageWords[wordBase + elemOff div 2],
                               make_layout((chunkWords,)))
      let srcFlat = lay.crd2idx((rowOrigin + rc, kOrigin + kc * chunkElems))
      if not rowIn:
        # outside the row bounds: zero-fill, no global read
        copyIf(atom, dstChunk, dstChunk, false, chunkWords)
      elif kIn == chunkElems and kStride == 1 and
          srcFlat mod 8 == 0:
        # fully in bounds, k-contiguous and the chunk 16-byte aligned:
        # one async chunk, the gmem side re-viewed as words
        let srcChunk = make_view(
          cast[ptr UncheckedArray[uint32]](addr src.data[srcFlat]),
          make_layout((chunkWords,)))
        copyIf(atom, dstChunk, srcChunk, true, chunkWords)
      elif kIn > 0:
        # K-tail chunk, or a full chunk that is k-strided or misaligned:
        # elementwise copy of the in-bounds prefix through the layout,
        # zero-fill for the rest
        for j in 0 ..< kIn:
          elems[wordBase * 2 + elemOff + j] = src.data[lay.crd2idx(
            (rowOrigin + rc, kOrigin + kc * chunkElems + j))]
        for j in kIn ..< chunkElems:
          elems[wordBase * 2 + elemOff + j] = zeroVal
      else:
        copyIf(atom, dstChunk, dstChunk, false, chunkWords)

proc gemm_smem*[TAcc, TA, TB, MT, NT, KT](
    mma: static TiledMma[TAcc, TA, TB, MT, NT, KT],
    mmaStep: static TiledMma[TAcc, TA, TB, MT, NT, 16],
    D: distinct TensorView, alpha: float32, A, B: distinct TensorView,
    beta: float32, C: distinct TensorView) =
  ## D = alpha·A·B + beta·C over (BM, BN) block tiles, each one staged
  ## in shared memory. Beta = 0 skips the C read.
  ## D may stay uninitialized.
  ##
  ## Staging takes the 16-byte async-copy chunk only where the operand
  ## is k-contiguous and the chunk 16-byte aligned. Operands that are
  ## k-strided or misaligned stage elementwise through their layout.
  ##
  ## Args:
  ## - mma: the warp tile schedule. One block tile measures
  ##   BM = warpsPerDim·MT rows by BN = warpsPerDim·NT columns over
  ##   a warpsPerDim x warpsPerDim warp grid
  ## - mmaStep: the same schedule narrowed to one atom cell along k
  ##   (KT = 16), the fragment loads and the accumulate step work one
  ##   k16 step at a time
  ## - alpha, beta: the epilogue scales, beta = 0 skips the C read
  ##
  ## Pipeline:
  ## - the reduction runs over BK = KT k-elements in two stage buffers
  ## - the wait drains the current stage and the barrier orders
  ##   previous consumers off the next buffer
  ## - the next stage issues, then the warps consume the current one
  ##   while its copy flies in
  ##
  ## The warps load their mma fragments straight out of the stage
  ## through the atom thread->value map, one k16 step at a time.
  ## D accumulates in registers across the whole reduction.
  ##
  ## Refusal conditions, the kernel returns with D untouched:
  ##
  ## - the operand k sizes differ, a mismatched reduction length would
  ##   zero-fill the shorter operand's tail into the result
  ## - the block runs any other thread count, a mis-sized block reads
  ##   and writes off the stage and the block tile
  if A.shape[1] != B.shape[1]:
    return
  if threads_per_threadgroup.x * threads_per_threadgroup.y *
      threads_per_threadgroup.z != warpsPerBlock * threadsPerWarp:
    return
  static:
    assert mmaStep.MT == mma.MT and mmaStep.NT == mma.NT,
      "gemm_smem: mmaStep must narrow mma to KT = 16 at the same MT/NT, " &
      "got " & $mmaStep.MT & "x" & $mmaStep.NT & " vs " &
      $mma.MT & "x" & $mma.NT
  const
    BM = warpsPerDim * mma.MT
    BN = warpsPerDim * mma.NT
    BK = mma.KT
    stageKPad = stageRowStride(BK)
    kSteps = BK div 16
    aStageWords = BM * stageKPad div 2
    bStageWords = BN * stageKPad div 2
    tvmAStep = mmaStep.getThreadValueMap(opA)
    tvmBStep = mmaStep.getThreadValueMap(opB)
    tvmC = mma.getThreadValueMap(opC)
    regAStep = mmaStep.getMmaLayout(opA)
    regBStep = mmaStep.getMmaLayout(opB)
    regC = mma.getMmaLayout(opC)

  var
    # +8 words of headroom for aligning the stage base up to 16 bytes,
    # slack words behind the aligned base stay unread
    stageA {.smem.}: array[2 * aStageWords + 8, uint32]
    stageB {.smem.}: array[2 * bStageWords + 8, uint32]

  let
    thrID = getLocalThreadID()
    warpID = thrID div 32
    laneID = thrID mod 32
    warpRow = warpID div warpsPerDim
    warpCol = warpID mod warpsPerDim
    aStage = cast[ptr UncheckedArray[uint32]](
      (cast[uint64](addr stageA[0]) + 15) and (not uint64(15)))
    bStage = cast[ptr UncheckedArray[uint32]](
      (cast[uint64](addr stageB[0]) + 15) and (not uint64(15)))
    aElems = cast[ptr UncheckedArray[A.T]](aStage)
    bElems = cast[ptr UncheckedArray[B.T]](bStage)
    cGeom = D.tileWith((BM, BN))
    cGrid = cGeom.getTileGrid()
    (_, kTiles) = A.tileWith((BM, BK)).getTileGrid()
    gridStride = getGridSize()

  for blockTile in countup(cGrid.getLocalTileID(), cGrid.size() - 1, gridStride):
    # Grid-stride block-tile reuse: the k-loop's top barrier orders
    # each kTile's consumers before the next restage, an odd kTiles
    # count leaves the final fragment reads with no trailing barrier.
    # This one orders every warp off the double buffers before the prologue
    # restages buffer 0.
    threadgroup_barrier()
    let (mTile, nTile) = idx2crd(cGrid, blockTile)
    let blockM0 = mTile * BM
    let blockN0 = nTile * BN

    # prologue: stage the first k tile into buffer 0
    stageSmemTile(aStage, A, thrID, BM, BK, 0, blockM0, 0,
                  A.shape[0], A.shape[1])
    stageSmemTile(bStage, B, thrID - stageThreads, BN, BK, 0, blockN0, 0,
                  B.shape[0], B.shape[1])
    commit_group(getCopyAsyncAtom(uint32))

    var rD: tile(D.T, regC)
    rD.fillWith(zero(D.T))
    var cur = 0
    for kTile in 0 ..< kTiles:
      # drain everything: the current stage is fully copied
      wait_group(getCopyAsyncAtom(uint32), 0)
      threadgroup_barrier()
      let kNext = (kTile + 1) * BK
      if kNext < kTiles * BK:
        # Next stage issues into the other buffer and overlaps the compute below.
        stageSmemTile(aStage, A, thrID, BM, BK, (1 - cur) * aStageWords,
                      blockM0, kNext, A.shape[0], A.shape[1])
        stageSmemTile(bStage, B, thrID - stageThreads, BN, BK,
                      (1 - cur) * bStageWords, blockN0, kNext,
                      B.shape[0], B.shape[1])
        commit_group(getCopyAsyncAtom(uint32))
      # Fragment consumption, one k16 step at a time: the warp's row
      # slice covers all steps and the tile id picks the k16 step,
      # with (0, kStep) · tileShape = kStep·16 k-elements per read.
      # Runtime-int slice layout, matched against the runtime layouts
      # inside the tile load/store machinery.
      let elemBase = cur * BM * stageKPad
      let bElemBase = cur * BN * stageKPad
      let rows = int(mma.MT)
      let cols = int(mma.NT)
      let stageCols = int(BK)
      let pad = int(stageKPad)
      let aSlice = make_view(
        addr aElems[elemBase + warpRow * mma.MT * stageKPad],
        make_layout((rows, stageCols), (pad, 1)))
      let bSlice = make_view(
        addr bElems[bElemBase + warpCol * mma.NT * stageKPad],
        make_layout((cols, stageCols), (pad, 1)))
      for kStep in 0 ..< kSteps:
        var rA: tile(A.T, regAStep)
        var rB: tile(B.T, regBStep)
        loadFragTile(rA, aSlice, tvmAStep, laneID, (0, kStep))
        loadFragTile(rB, bSlice, tvmBStep, laneID, (0, kStep))
        mmaStep.atom.gemm_tile(rD, rA, rB, rD)
      cur = 1 - cur

    # epilogue: scale, accumulate C, store the warp's D tile
    if alpha != 1.0'f32:
      for i in 0 ..< regC.size():
        rD[i] = alpha * rD[i]
    let dRow = warpsPerDim * mTile + warpRow
    let dCol = warpsPerDim * nTile + warpCol
    if beta != 0.0'f32:
      let rC = tile(C.T, regC).load(C, tvmC, laneID, (dRow, dCol))
      for i in 0 ..< regC.size():
        rD[i] += beta * rC[i]
    D.store(rD, tvmC, laneID, (dRow, dCol))

proc matmul_smem*[TAcc, TA, TB, MT, NT, KT](
    mma: static TiledMma[TAcc, TA, TB, MT, NT, KT],
    mmaStep: static TiledMma[TAcc, TA, TB, MT, NT, 16],
    D, A, B: distinct TensorView) =
  ## D = A·B, same staging and launch mapping as gemm_smem:
  ##
  ## - D may stay uninitialized, equivalent to the alpha = 1, beta = 0
  ##   gemm_smem call over C = D
  gemm_smem(mma, mmaStep, D, 1.0'f32, A, B, 0.0'f32, D)

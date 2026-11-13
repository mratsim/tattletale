# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Grouped GEMV and GEMM kernels: every line contracts against one (K, N) weight matrix per group, b (G, K, N).
## The variants are named by their input form.
## - The offsets form: the lines of all groups sit concatenated group-major,
##   a (G,) int32 tensor of cumulative END offsets marks each group's line range,
##   each group owns one contiguous range.
## - The expert-ids form: the ids tensor stores one expert id per line, the kernel
##   reads it at runtime, so groups may interleave inside an output tile,
##   an id below 0 pads or skips the line.
##
## Public kernels:
## - grouped_gemv_via_offsets: dst[m, :] = a[m, :] · b_g[m], one threadgroup per (line, n-block)
## - grouped_gemm_via_offsets: dst = a_g · b_g over (MT, NT) output tiles
## - grouped_gemv_via_expertIds: dst(t, :) = sum over i of w(t, i) · x(t, :) · b_ids(t,i),
##   one launch folds the routing combine, grouped_gemv_splitk + grouped_gemv_reduce
##   split K for small t·n
## - grouped_gemm_via_expertIds: dst = a_g · b_g over per-line ids

import workspace/crucible/src/crucible_gpu_builtins
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/kernels/k_layout_gemm
import workspace/positron/src/kernels/k_moe_router {.all.}

# ═══════════════════════════════════════════════════════════════════════
# Shared device helpers
# ═══════════════════════════════════════════════════════════════════════

template getOwningGroup(offs: untyped, gCount, target: int): int =
  ## Returns the group owning line `target`: the smallest g with offs[g] > target.
  var
    lo = 0
    hi = gCount - 1
  while lo < hi:
    let mid = (lo + hi) div 2
    if offs[mid] > target:
      hi = mid
    else:
      lo = mid + 1
  lo

template lineGroupMask(mask: var Tile, lineLo, lineHi, m0: int) =
  ## Marks the lines of `mask` inside [lineLo, lineHi), absolute lines minus m0, closes over regCrd and crdBase.
  forEach idx, var c in mask:
    let (lineLocal, _) = crdBase + regCrd.crd2idx(idx)
    c = lineLocal >= lineLo - m0 and lineLocal < lineHi - m0

# ═══════════════════════════════════════════════════════════════════════
# Dispatch rules
# ═══════════════════════════════════════════════════════════════════════

type GroupedTier = enum
  gtGemv
  gtGemm

proc pickGroupedTier(rowsPerExpert: int): GroupedTier =
  if rowsPerExpert < 8:
    gtGemv
  else:
    gtGemm

type GroupedGemvForm = enum
  ggfSingleLaunch
  ggfSplitK

const GroupedGemvSplitMaxOutputs = 4096
  ## Split-K dispatch boundary in outputs t·n.

const GroupedGemvSplitKSplits = 32

proc groupedGemvForm(t, n: int): tuple[form: GroupedGemvForm, kSplits: int] =
  if t * n <= GroupedGemvSplitMaxOutputs:
    result = (ggfSplitK, GroupedGemvSplitKSplits)
  else:
    result = (ggfSingleLaunch, 1)

# ═══════════════════════════════════════════════════════════════════════
# grouped_gemv_via_offsets
# ═══════════════════════════════════════════════════════════════════════

const GroupedGemvOutputs = 4
  ## Each grouped_gemv lane owns GroupedGemvOutputs consecutive outputs: dst[m, n0 .. n0+3].
  ## Per k step a lane reads GroupedGemvOutputs consecutive b elements, so the simdgroup's
  ## reads cover getLaneWidth()·GroupedGemvOutputs contiguous b elements per step.

proc grouped_gemv_via_offsets*(dst, a, b, offs: distinct TensorView) =
  ## Decode grouped GEMV for the offsets form: every line contracts
  ## against its group's weight matrix.
  ##
  ## ```
  ## a (6, K) group-major   offs = [2 5 6]              b (3, K, N)
  ## lines 0..1  group 0    dst[0, :] = a[0, :] · b_0[:]
  ## lines 2..4  group 1    dst[3, :] = a[3, :] · b_1[:]
  ## lines 5..5  group 2
  ## ```
  ##
  ## - a (totalM, K), offs (G,) int32 cumulative END offsets, b (G, K, N),
  ##   dst (totalM, N)
  ## - dst[m, :] = a[m, :] · b_g[m], g the group owning line m
  let
    totalM = a.shape[0]
    gCount = b.shape[0]
    k = a.shape[1]
    n = dst.shape[1]
    nBlocks = n.ceil_div(getLaneWidth() * GroupedGemvOutputs)
    totalTiles = totalM * nBlocks
  for tileID in countup(getLocalTileID(totalTiles), totalTiles - 1, getGridSize()):
    let
      m = tileID div nBlocks
      nBlock = tileID mod nBlocks
      n0 = nBlock * (getLaneWidth() * GroupedGemvOutputs) + getLocalThreadID() * GroupedGemvOutputs
    if n0 < n:
      let g = offs.getOwningGroup(gCount, m)
      var accs: array[GroupedGemvOutputs, float32]
      for e in 0 ..< GroupedGemvOutputs:
        accs[e] = 0
      for kk in 0 ..< k:
        let av = to(a[m, kk], float32)
        for e in 0 ..< GroupedGemvOutputs:
          # Ragged N edge: in-range elements load unguarded
          if n0 + e < n:
            accs[e] += av * to(b[g, kk, n0 + e], float32)
      for e in 0 ..< GroupedGemvOutputs:
        if n0 + e < n:
          dst[m, n0 + e] = to(accs[e], dst.T)

# ═══════════════════════════════════════════════════════════════════════
# grouped_gemv_via_expertIds
# ═══════════════════════════════════════════════════════════════════════

proc grouped_gemv_via_expertIds*(dst, x, b, ids, weights: distinct TensorView, topk: int) =
  ## MoE decode expert GEMV with the routing combine folded in:
  ## each token contracts against its top-k experts, the router weights the sums.
  ##
  ## ```
  ## x (2, K)   ids        weights      dst (2, N)
  ## t0: x0     [2 0 -]    [w0 w1 -]    dst[t0, :] = w0·b_2·x0 + w1·b_0·x0
  ## t1: x1     [0 1 -]    [w0 w1 -]    dst[t1, :] = w0·b_0·x1 + w1·b_1·x1
  ## ```
  ##
  ## - x (T, K), ids (T, topk) int32, weights (T, topk), b (E, K, N), dst (T, N)
  ## - dst[t, :] = sum over i of weights[t, i] · x[t, :] · b[ids[t, i]]
  ## - an id below 0 marks an unused top-k slot and contributes nothing
  let
    tokens = x.shape[0]
    k = x.shape[1]
    n = dst.shape[1]
    nBlocks = n.ceil_div(getLaneWidth() * GroupedGemvOutputs)
    totalTiles = tokens * nBlocks
  for tileID in countup(getLocalTileID(totalTiles), totalTiles - 1, getGridSize()):
    let
      t = tileID div nBlocks
      nBlock = tileID mod nBlocks
      n0 = nBlock * (getLaneWidth() * GroupedGemvOutputs) + getLocalThreadID() * GroupedGemvOutputs
    if n0 < n:
      var sums: array[GroupedGemvOutputs, float32]
      var accs: array[GroupedGemvOutputs, float32]
      for e in 0 ..< GroupedGemvOutputs:
        sums[e] = 0
      for i in 0 ..< topk:
        let gid: int = ids[t, i]
        if gid >= 0:
          let w = to(weights[t, i], float32)
          for e in 0 ..< GroupedGemvOutputs:
            accs[e] = 0
          for kk in 0 ..< k:
            let xv = to(x[t, kk], float32)
            for e in 0 ..< GroupedGemvOutputs:
              # Ragged N edge: in-range elements load unguarded
              if n0 + e < n:
                accs[e] += xv * to(b[gid, kk, n0 + e], float32)
          for e in 0 ..< GroupedGemvOutputs:
            sums[e] += w * accs[e]
      for e in 0 ..< GroupedGemvOutputs:
        if n0 + e < n:
          dst[t, n0 + e] = to(sums[e], dst.T)

proc grouped_gemv_splitk(dPartials, x, b, ids, weights: distinct TensorView, topk, kSplits: int) =
  ## Split-K stage 1: per-slot partial sums into the (T, kSplits, N) fp32 dPartials, one threadgroup per (line, n-block, k-slice).
  let
    tokens = x.shape[0]
    k = x.shape[1]
    n = dPartials.shape[2]
    nBlocks = n.ceil_div(getLaneWidth() * GroupedGemvOutputs)
    kQuot = k div kSplits
    kRem = k mod kSplits
    totalSlots = tokens * nBlocks * kSplits
  for slot in countup(getLocalTileID(totalSlots), totalSlots - 1, getGridSize()):
    let
      t = slot div (nBlocks * kSplits)
      slotX = slot mod (nBlocks * kSplits)
      kSlice = slotX mod kSplits
      n0 = (slotX div kSplits) * (getLaneWidth() * GroupedGemvOutputs) + getLocalThreadID() * GroupedGemvOutputs
      k0 = kSlice * kQuot + min(kSlice, kRem)
      kLen = kQuot + (if kSlice < kRem: 1 else: 0)
    if n0 < n:
      var sums: array[GroupedGemvOutputs, float32]
      var accs: array[GroupedGemvOutputs, float32]
      for e in 0 ..< GroupedGemvOutputs:
        sums[e] = 0
      for i in 0 ..< topk:
        let gid: int = ids[t, i]
        if gid >= 0:
          let w = to(weights[t, i], float32)
          for e in 0 ..< GroupedGemvOutputs:
            accs[e] = 0
          for kk in k0 ..< k0 + kLen:
            let xv = to(x[t, kk], float32)
            for e in 0 ..< GroupedGemvOutputs:
              # Ragged N edge: in-range elements load unguarded
              if n0 + e < n:
                accs[e] += xv * to(b[gid, kk, n0 + e], float32)
          for e in 0 ..< GroupedGemvOutputs:
            sums[e] += w * accs[e]
      for e in 0 ..< GroupedGemvOutputs:
        if n0 + e < n:
          dPartials[t, kSlice, n0 + e] = sums[e]

proc grouped_gemv_reduce(dst, dPartials: distinct TensorView, kSplits: int) =
  ## Split-K stage 2: sums dPartials into dst over ascending k slices in fp32,
  ## one threadgroup per (line, n-block) over stage 1's kSplits.
  let
    tokens = dst.shape[0]
    n = dst.shape[1]
    nBlocks = n.ceil_div(getLaneWidth() * GroupedGemvOutputs)
    totalTiles = tokens * nBlocks
  for tileID in countup(getLocalTileID(totalTiles), totalTiles - 1, getGridSize()):
    let
      t = tileID div nBlocks
      nBlock = tileID mod nBlocks
      n0 = nBlock * (getLaneWidth() * GroupedGemvOutputs) + getLocalThreadID() * GroupedGemvOutputs
    if n0 < n:
      var sums: array[GroupedGemvOutputs, float32]
      for e in 0 ..< GroupedGemvOutputs:
        sums[e] = 0
      for s in 0 ..< kSplits:
        for e in 0 ..< GroupedGemvOutputs:
          if n0 + e < n:
            sums[e] += dPartials[t, s, n0 + e]
      for e in 0 ..< GroupedGemvOutputs:
        if n0 + e < n:
          dst[t, n0 + e] = to(sums[e], dst.T)

# ═══════════════════════════════════════════════════════════════════════
# grouped_gemm
# ═══════════════════════════════════════════════════════════════════════

func groupedGemmTiles(T: typedesc): tuple[MT, NT, KT: int] {.compileTime.} =
  when T is float32 or T is float16:
    (32, 32, 16)
  else:
    {.error: "grouped_gemm has no tile schedule for " & $T &
        ", Metal simdgroup atoms cover fp32/fp16".}

func countGroupedGemmTiles(T: typedesc, totalM, n: int): int =
  const (MT, NT, _) = groupedGemmTiles(T)
  result = totalM.ceil_div(MT) * n.ceil_div(NT)

proc grouped_gemm_via_offsets*(dst, a, b, offs: distinct TensorView) =
  ## Prefill grouped GEMM for the offsets form: each group's lines
  ## contract against the group's weight matrix.
  ##
  ## ```
  ## a (6, K) group-major   offs = [2 5 6]
  ## group 0: lines 0..1  ─►  dst[0:2, :] = a[0:2, :] · b_0
  ## group 1: lines 2..4  ─►  dst[2:5, :] = a[2:5, :] · b_1
  ## group 2: lines 5..5  ─►  dst[5:6, :] = a[5:6, :] · b_2
  ## ```
  ##
  ## - a (totalM, K), offs (G,) int32 cumulative END offsets, b (G, K, N),
  ##   dst (totalM, N)
  ## - the tiled (MT, NT, KT) schedule is groupedGemmTiles(a.T)
  const (MT, NT, KT) = groupedGemmTiles(a.T)
  const mma = tiledMma(float32, a.T, b.T, MT, NT, KT)
  static: doAssert mma.atom.getThreadCount() == getLaneWidth(),
    "grouped_gemm_via_offsets: one simdgroup per threadgroup, the atom must cover the lanes"
  const
    tvmA = mma.getThreadValueMap(opA)
    tvmB = mma.getThreadValueMap(opB)
    tvmC = mma.getThreadValueMap(opC)
    aLayout = mma.getMmaLayout(opA)
    bLayout = mma.getMmaLayout(opB)
    cLayout = mma.getMmaLayout(opC)

  let thrID = getLocalThreadID()
  if thrID < getLaneWidth():
    let
      totalM = a.shape[0]
      gCount = b.shape[0]
      nTiles = b.shape[2].ceil_div(NT)
      totalTiles = totalM.ceil_div(MT) * nTiles
      (_, kTiles) = a.tileWith((MT, KT)).getTileGrid()
      thrCrd = idx2crd(tvmA.getThreadPartitioning(), thrID)
      regCrd = tvmA.layout.slice(((thrCrd, _), _))
      crdBase = tvmA.layout.crd2idx(((thrCrd, _), _))
    var rAm: tile(a.T, aLayout)
    for tileID in countup(getLocalTileID(totalTiles), totalTiles - 1, getGridSize()):
      let
        mTile = tileID div nTiles
        nTile = tileID mod nTiles
        m0 = mTile * MT
        mEnd = min(m0 + MT, totalM)
        gLo = offs.getOwningGroup(gCount, m0)
        gHi = offs.getOwningGroup(gCount, mEnd - 1)
      var rD = zeros(tile(float32, cLayout))
      for g in gLo .. gHi:
        let
          gEnd: int = offs[g]
          gStart: int = if g == 0: 0 else: offs[g - 1]
          lineLo = max(m0, gStart)
          lineHi = min(m0 + MT, gEnd)
        if lineLo < lineHi:
          var mask: tile(bool, aLayout)
          mask.lineGroupMask(lineLo, lineHi, m0)
          let bT = b(g, _, _).transpose(0, 1)
          for kTile in 0 ..< kTiles:
            var rA: tile(a.T, aLayout)
            rA.load(a, tvmA, thrID, (mTile, kTile))
            rAm.where(mask, rA, zero(a.T))
            var rB: tile(b.T, bLayout)
            rB.load(bT, tvmB, thrID, (nTile, kTile))
            mma.atom.gemm_tile(rD, rAm, rB, rD)
      dst.store(rD.downcast(dst.T), tvmC, thrID, (mTile, nTile))

proc grouped_gemm_via_expertIds*(dst, a, b, expertIds: distinct TensorView) =
  ## MoE prefill expert GEMM over per-line expert ids:
  ## each packed line contracts against its expert's weight matrix over the shared K and N dimensions.
  ##
  ## ```
  ## a (7, K), expertIds (7,)
  ## ids: [g0 g0 g0 -1 g1 g1 -1]  ─►   dst[m, :] = a[m, :] · b_ids[m]
  ##                                   for the valid lines of the example:
  ##                                   lines 0..2 against b_0, lines 4..5
  ##                                   against b_1
  ## ```
  ##
  ## - a (totalM, K) expert-major with -1 padding line tails, expertIds
  ##   (totalM,) int32, b (E, K, N), dst (totalM, N)
  ## - the tiled (MT, NT, KT) schedule is groupedGemmTiles(a.T)
  const (MT, NT, KT) = groupedGemmTiles(a.T)
  const mma = tiledMma(float32, a.T, b.T, MT, NT, KT)
  static: doAssert mma.atom.getThreadCount() == getLaneWidth(),
    "grouped_gemm_via_expertIds: one simdgroup per threadgroup, the atom must cover the lanes"
  const
    tvmA = mma.getThreadValueMap(opA)
    tvmB = mma.getThreadValueMap(opB)
    tvmC = mma.getThreadValueMap(opC)
    aLayout = mma.getMmaLayout(opA)
    bLayout = mma.getMmaLayout(opB)
    cLayout = mma.getMmaLayout(opC)

  let thrID = getLocalThreadID()
  if thrID < getLaneWidth():
    let
      totalM = a.shape[0]
      nTiles = b.shape[2].ceil_div(NT)
      totalTiles = totalM.ceil_div(MT) * nTiles
      (_, kTiles) = a.tileWith((MT, KT)).getTileGrid()
      thrCrd = idx2crd(tvmA.getThreadPartitioning(), thrID)
      regCrd = tvmA.layout.slice(((thrCrd, _), _))
      crdBase = tvmA.layout.crd2idx(((thrCrd, _), _))
    var rAm: tile(a.T, aLayout)
    for tileID in countup(getLocalTileID(totalTiles), totalTiles - 1, getGridSize()):
      let
        mTile = tileID div nTiles
        nTile = tileID mod nTiles
        m0 = mTile * MT
        mLines = min(m0 + MT, totalM) - m0
      var rD = zeros(tile(float32, cLayout))
      var tileHasValidLine = false
      var j = 0
      while j < mLines:
        let gid: int = expertIds[m0 + j]
        if gid < 0:
          j = j + 1
        else:
          tileHasValidLine = true
          var jEnd = j + 1
          while jEnd < mLines and expertIds[m0 + jEnd] == gid:
            jEnd = jEnd + 1
          var mask: tile(bool, aLayout)
          mask.lineGroupMask(m0 + j, m0 + jEnd, m0)
          let bT = b(gid, _, _).transpose(0, 1)
          for kTile in 0 ..< kTiles:
            var rA: tile(a.T, aLayout)
            rA.load(a, tvmA, thrID, (mTile, kTile))
            rAm.where(mask, rA, zero(a.T))
            var rB: tile(b.T, bLayout)
            rB.load(bT, tvmB, thrID, (nTile, kTile))
            mma.atom.gemm_tile(rD, rAm, rB, rD)
          j = jEnd
      if tileHasValidLine:
        dst.store(rD.downcast(dst.T), tvmC, thrID, (mTile, nTile))

# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Grouped contraction kernels: every line contracts against one (K, N) weight matrix per group, b (G, K, N).
## The variants are named by their input form.
## - The offsets form: the lines of all groups sit concatenated group-major,
##   a (G,) int32 tensor of cumulative END offsets marks each group's line range,
##   each group owns one contiguous range.
## - The expert-ids form: the ids tensor stores one expert id per line, the kernel
##   reads it at runtime, so groups may interleave inside an output tile,
##   an id below 0 pads or skips the line.
##
## Public kernels:
## - grouped_gemv_via_offsets: d[m, :] = a[m, :] · b_g[m], one threadgroup per (line, n-block)
## - grouped_gemm_via_offsets: d = a_g · b_g over (MT, NT) output tiles
## - grouped_gemv_via_expertIds: d(t, :) = sum over i of w(t, i) · x(t, :) · b_ids(t,i),
##   one launch folds the routing combine, grouped_gemv_splitk + grouped_gemv_reduce
##   split K for small t·n
## - grouped_gemm_via_expertIds: d = a_g · b_g over per-line ids

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
    if offs.data[mid] > target:
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
  ## Each grouped_gemv lane owns GroupedGemvOutputs consecutive outputs: d[m, n0 .. n0+3].
  ## Per k step a lane reads GroupedGemvOutputs consecutive b elements, so the simdgroup's
  ## reads cover getLaneWidth()·GroupedGemvOutputs contiguous b elements per step.

proc grouped_gemv_via_offsets*(d, a, b, offs: distinct TensorView) =
  ## Per-line contraction `d[m, :] = a[m, :] · b_g[m]`, one threadgroup per (line, n-block).
  let
    totalM: int = a.shape[0]
    gCount: int = b.shape[0]
    k: int = a.shape[1]
    n: int = d.shape[1]
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
          d[m, n0 + e] = to(accs[e], d.T)

# ═══════════════════════════════════════════════════════════════════════
# grouped_gemv_via_expertIds
# ═══════════════════════════════════════════════════════════════════════

proc grouped_gemv_via_expertIds*(d, x, b, ids, weights: distinct TensorView, topk: int) =
  ## Folded-combine grouped GEMV, one launch, one threadgroup per (line, n-block).
  let
    tokens: int = x.shape[0]
    k: int = x.shape[1]
    n: int = d.shape[1]
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
          d[t, n0 + e] = to(sums[e], d.T)

proc grouped_gemv_splitk*(dPartials, x, b, ids, weights: distinct TensorView, topk, kSplits: int) =
  ## Split-K stage 1: per-slot partial sums into the (T, kSplits, N) fp32 dPartials, one threadgroup per (line, n-block, k-slice).
  let
    tokens: int = x.shape[0]
    k: int = x.shape[1]
    n: int = dPartials.shape[2]
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

proc grouped_gemv_reduce*(d, dPartials: distinct TensorView, kSplits: int) =
  ## Split-K stage 2: sums dPartials into d over ascending k slices in fp32,
  ## one threadgroup per (line, n-block) over stage 1's kSplits.
  let
    tokens: int = d.shape[0]
    n: int = d.shape[1]
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
          d[t, n0 + e] = to(sums[e], d.T)

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

proc grouped_gemm_via_offsets*(d, a, b, offs: distinct TensorView) =
  ## Per-group contraction `d = a_g · b_g`, one simdgroup per (MT, NT) output tile, countGroupedGemmTiles(T, totalM, n) threadgroups.
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
      totalM: int = a.shape[0]
      gCount: int = b.shape[0]
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
          gEnd: int = offs.data[g]
          gStart: int = if g == 0: 0 else: offs.data[g - 1]
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
      d.store(rD.downcast(d.T), tvmC, thrID, (mTile, nTile))

proc grouped_gemm_via_expertIds*(d, a, b, expertIds: distinct TensorView) =
  ## Per-expert contraction `d = a_g · b_g`, countGroupedGemmTiles(a.T, totalM, n) threadgroups, per-tile run-walk over expertIds.
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
      totalM: int = a.shape[0]
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
        let gid: int = expertIds.data[m0 + j]
        if gid < 0:
          j = j + 1
        else:
          tileHasValidLine = true
          var jEnd = j + 1
          while jEnd < mLines and expertIds.data[m0 + jEnd] == gid:
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
        d.store(rD.downcast(d.T), tvmC, thrID, (mTile, nTile))

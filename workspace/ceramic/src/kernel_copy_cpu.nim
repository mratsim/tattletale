## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

##
## Codegen flow (ASCII) — <compile-time> vs [runtime]:
##
##   effR = effective rank = number of dims after removing compile-time size-1 dims
##   bs = blockSize parameter (static int, default -1).
##        When > 0, genNestedCopy tiles loops into bs-sized blocks for cache efficiency.
##        genCopyMemLoops ignores bs (single copyMem already optimal).
##
##   <copySameShapeImpl> ──► <buildStrideSortedArrays>
##   <copyPermutedImpl>          │
##                               │ skip size-1 dims, reorder by stride
##                               ▼
##                          effR == 0? ──yes──► dst[0] = src[0]  [single elem copy]
##                               │
##                              no
##                               │
##                               ▼
##                          <genContiguityCode>
##                          walk innermost→outermost
##                          check srcSt[d]==innerProd && dstSt[d]==innerProd
##                               │
##                               ▼
##                       copyDimStart < effR?
##                        (contiguous suffix found?)
##                         /                    \
##                       yes                    no
##                        │                      │
##                        ▼                      ▼
##   <genCopyMemLoops>              <genNestedCopy>
##   (outerR, copyCount)            (effR, bs)
##        │                              │
##        │ wrap outerR for-loops        │ if bs>0: tile into bs×bs blocks
##        ▼                              ▼
##   [for d in 0..<outerR:          [for all dims:
##      copyMem(copyCount)]           elem-by-elem copy]
import std/[macros, algorithm]
import ./int_tuples
import ./layouts
import ./tensors


# ── compile-time helpers ────────────────────────────────────────
proc flatTupleLen(t: NimNode): int {.compileTime.} =
  ## Count leaf elements in a (possibly nested) tuple type.
  if t.kind == nnkTupleConstr:
    for i in 0 ..< t.len:
      result += flatTupleLen(t[i])
  else:
    result = 1

# ── Helper: seq[int] → nnkBracket (compile-time array literal) ──

proc toBracket(vals: seq[int]): NimNode {.compileTime.} =
  result = nnkBracket.newTree()
  for v in vals:
    result.add newLit(v)

# ── Helper: flatten + index into a layout field ─────────────────

proc flattenElem(lay, field: NimNode; idx: NimNode; totalRank: int): NimNode {.compileTime.} =
  ## Generates `int(flatten(lay.field)[idx])` (multi-mode) or `int(flatten(lay.field))` (rank-1).
  if totalRank <= 1:
    result = quote do:
      int(flatten(`lay`.`field`))
  else:
    result = quote do:
      int(flatten(`lay`.`field`)[`idx`])


proc buildStrideSortedArrays(
    R: int;
    rawSrcSh, rawSrcSt, rawDstSh, rawDstSt: seq[int];
    srcLay, dstLay: NimNode;
    permDtoS: seq[int]
): tuple[
    shLit, dstStLit, srcStLit: NimNode;
    shapeVals, srcStVals, dstStVals: seq[int];
    effR, lastOk: int
] {.compileTime.} =
  ## Build stride-sorted arrays from compile-time-static shape/stride info.
  ##
  ## `effR` = effective rank = number of dims after filtering out size-1 dims.
  ##
  ## Flow:
  ##   1. Walk dims 0..R-1.
  ##   2. Skip compile-time size-1 dims (trivial loops that would
  ##      iterate once — they clutter the loop nest for no benefit).
  ##   3. For each kept dim, emit shape/dstSt/srcSt into parallel arrays.
  ##      Unknown (runtime) values get DynamicSentinel in `Vals` seq
  ##      and a `flattenElem()` call in the `Lit` NimNode (codegen).
  ##   4. Post-loop: reorder so unknown-stride dims come FIRST
  ##      (outermost loops — fewer iterations, conservative placement)
  ##      and known-stride dims are stride-sorted descending (innermost).
  ##
  ## Returns:
  ##   shLit/dstStLit/srcStLit: nnkBracket literals for let-bindings in generated code.
  ##     Mix of `newLit(Int[N])` for compile-time-known and `flattenElem(...)` for runtime.
  ##   shapeVals/srcStVals/dstStVals: seq[int] of compile-time-known values,
  ##     DynamicSentinel for runtime-unknown (for compile-time analysis like contiguity).
  ##   effR: effective rank after size-1 filtering.
  ##   lastOk: original dim index of the innermost (last) entry after reordering.
  ##     Used by caller for innermost-stride checks.
  result.shLit = nnkBracket.newTree()
  result.dstStLit = nnkBracket.newTree()
  result.srcStLit = nnkBracket.newTree()
  result.effR = 0
  result.lastOk = -1
  var okVals: seq[int]
  for k in 0 ..< R:

    let shVal = if k < rawDstSh.len: rawDstSh[k] else: DynamicSentinel
    # Skip compile-time-known size-1 dims (trivial loops)
    if shVal != DynamicSentinel and shVal == 1:
      continue
    okVals.add k
    result.effR += 1
    # Shape (always from dst — we write to dst)
    if k < rawDstSh.len and rawDstSh[k] != DynamicSentinel:
      result.shLit.add newLit(rawDstSh[k])
      result.shapeVals.add rawDstSh[k]
    else:
      result.shLit.add flattenElem(dstLay, ident"shape", newLit(k), R)
      result.shapeVals.add DynamicSentinel
    # Dst stride
    if k < rawDstSt.len and rawDstSt[k] != DynamicSentinel:
      result.dstStLit.add newLit(rawDstSt[k])
      result.dstStVals.add rawDstSt[k]
    else:
      result.dstStLit.add flattenElem(dstLay, ident"stride", newLit(k), R)
      result.dstStVals.add DynamicSentinel
    # Src stride (with optional permutation)
    let srcPos = if permDtoS.len > 0: permDtoS[k] else: k
    if srcPos < rawSrcSt.len and rawSrcSt[srcPos] != DynamicSentinel:
      result.srcStLit.add newLit(rawSrcSt[srcPos])
      result.srcStVals.add rawSrcSt[srcPos]
    else:
      result.srcStLit.add flattenElem(srcLay, ident"stride", newLit(srcPos), R)
      result.srcStVals.add DynamicSentinel
  # Reorder: unknown-stride dims outermost, sorted known-stride dims inner
  if result.effR > 1:
    var known, unknown: seq[int]
    for d in 0 ..< result.effR:
      if result.dstStVals[d] == DynamicSentinel: unknown.add d
      else: known.add d
    # Always sort known strides by descending dst stride even when there
    # are no unknown strides (innermost loop = smallest stride)
    if known.len > 0:
      let dstVals = result.dstStVals
      known.sort do (a, b: int) -> int:
        cmp(dstVals[b], dstVals[a])
      let newOrder = unknown & known
      let oldSh = result.shLit
      let oldDstSt = result.dstStLit
      let oldSrcSt = result.srcStLit
      let oldShV = result.shapeVals
      let oldDstV = result.dstStVals
      let oldSrcV = result.srcStVals
      result.shLit = nnkBracket.newTree()
      result.dstStLit = nnkBracket.newTree()
      result.srcStLit = nnkBracket.newTree()
      let oldOkV = okVals
      okVals = @[]
      result.shapeVals = @[]
      result.dstStVals = @[]
      result.srcStVals = @[]
      for d in newOrder:
        result.shLit.add oldSh[d]
        result.shapeVals.add oldShV[d]
        result.dstStLit.add oldDstSt[d]
        result.dstStVals.add oldDstV[d]
        result.srcStLit.add oldSrcSt[d]
        result.srcStVals.add oldSrcV[d]
        okVals.add oldOkV[d]
      if known.len > 0:
        result.lastOk = okVals[^1]
      else:
        result.lastOk = -1

proc genCopyMemLoops(
    dstData, srcData, shSym, dstStSym, srcStSym: NimNode;
    loopCount: int; copyCountExpr: NimNode
): NimNode {.compileTime.} =
  var baseDst, baseSrc: NimNode = newLit(0)
  let idxSyms = nnkBracket.newTree()
  for d in 0 ..< loopCount:
    idxSyms.add ident("mm" & $d & "_")
    let iSym = idxSyms[d]
    let ds = newLit(d)
    baseDst = quote do: `baseDst` + `iSym` * `dstStSym`[`ds`]
    baseSrc = quote do: `baseSrc` + `iSym` * `srcStSym`[`ds`]
  result = quote do:
    copyMem(addr `dstData`[`baseDst`], addr `srcData`[`baseSrc`],
            `copyCountExpr` * sizeof(typeof(`dstData`[0])))
  for d in countdown(loopCount - 1, 0):
    let iSym = idxSyms[d]
    let sd = newLit(d)
    result = quote do:
      for `iSym` in 0 ..< `shSym`[`sd`]:
        `result`

proc genNestedCopy(
    R, blockSize: int;
    shape, srcStrides, dstStrides: NimNode;
    dstPtr, srcPtr: NimNode
): NimNode =
  ## Generate nested for-loops for strided copy.
  #
  # Rank-0 guard: no dims → nothing to copy
  if R == 0:
    return newEmptyNode()

  # Generate loop variable symbols
  let indices = nnkBracket.newTree()
  for d in 0 ..< R:
    indices.add genSym(nskForVar, "i" & $d & "_")

  # ── Rank-1: single dimension (fast path, no offset accumulation) ──
  if R == 1:
    let i0 = indices[0]
    if blockSize <= 0:
      return quote do:
        for `i0` in 0 ..< `shape`[0]:
          `dstPtr`[`i0` * `dstStrides`[0]] = `srcPtr`[`i0` * `srcStrides`[0]]
    let i0b = genSym(nskForVar, "i0_block_")
    return quote do:
      for `i0b` in countup(0, `shape`[0] - 1, `blockSize`):
        for `i0` in `i0b` ..< min(`i0b` + `blockSize`, `shape`[0]):
          `dstPtr`[`i0` * `dstStrides`[0]] = `srcPtr`[`i0` * `srcStrides`[0]]

  # ── Rank > 1: build offset = sum(idx_i * stride_i) ──
  var dstOff, srcOff: NimNode = newLit(0)
  for d in 0 ..< R:
    let idx = indices[d]
    let ds = newLit(d)
    dstOff = quote do:
      `dstOff` + `idx` * `dstStrides`[`ds`]
    srcOff = quote do:
      `srcOff` + `idx` * `srcStrides`[`ds`]
  let body = quote do:
    `dstPtr`[`dstOff`] = `srcPtr`[`srcOff`]

  # ── No blocking: wrap body with full-range loops (innermost→outermost) ──
  if blockSize <= 0:
    result = body
    for i in countdown(R - 1, 0):
      let idx = indices[i]
      let si = newLit(i)
      result = quote do:
        for `idx` in 0 ..< `shape`[`si`]:
          `result`

  # ── Blocking: full tiling on ALL dims (no degenerate case) ──
  else:
    var blockIdx: seq[NimNode]
    for i in 0 ..< R:
      blockIdx.add genSym(nskForVar, "i" & $i & "_block_")
    # Step 1: inner tiled loops (each dim is a bs-wide tile)
    result = body
    for i in countdown(R - 1, 0):
      let idx = indices[i]
      let bIdx = blockIdx[i]
      let si = newLit(i)
      result = quote do:
        for `idx` in `bIdx` ..< min(`bIdx` + `blockSize`, `shape`[`si`]):
          `result`
    # Step 2: outer blocking loops (iterate tiles across each dim)
    for i in countdown(R - 1, 0):
      let bIdx = blockIdx[i]
      let si = newLit(i)
      result = quote do:
        for `bIdx` in countup(0, `shape`[`si`] - 1, `blockSize`):
          `result`


# ── Runtime fallback: build flat array args from layout fields ──
# ── (precomputed into local let-bindings to avoid flatten() in loops) ──

proc runtimeCopyArgs(
    dst, src: NimNode; R, bs: int;
    dstData, srcData: NimNode
): NimNode {.compileTime.} =
  ## Build genNestedCopy call using runtime layout.shape/stride.
  ## Extracts flat arrays ONCE into local variables before the nested loops.
  let srcLay = newTree(nnkDotExpr, src, ident"layout")
  let dstLay = newTree(nnkDotExpr, dst, ident"layout")
  if R == 1:
    let sh = flattenElem(srcLay, ident"shape", newLit(0), R)
    let sst = flattenElem(srcLay, ident"stride", newLit(0), R)
    let dst = flattenElem(dstLay, ident"stride", newLit(0), R)
    # rank-1: shape/strides are scalars, inline loops directly
    if bs <= 0:
      let i0 = ident("rc0_")
      return quote do:
        for `i0` in 0 ..< `sh`:
          `dstData`[`i0` * `dst`] = `srcData`[`i0` * `sst`]
    else:
      let i0b = genSym(nskForVar, "rc0b_")
      let i0 = ident("rc0_")
      return quote do:
        for `i0b` in countup(0, `sh` - 1, `bs`):
          for `i0` in `i0b` ..< min(`i0b` + `bs`, `sh`):
            `dstData`[`i0` * `dst`] = `srcData`[`i0` * `sst`]
  var stmts = newStmtList()
  let shSym = genSym(nskLet, "sh")
  let srcStSym = genSym(nskLet, "srcSt")
  let dstStSym = genSym(nskLet, "dstSt")
  var shapeLit = nnkBracket.newTree()
  var srcStLit = nnkBracket.newTree()
  var dstStLit = nnkBracket.newTree()
  for i in 0 ..< R:
    let li = newLit(i)
    shapeLit.add flattenElem(srcLay, ident"shape", li, R)
    srcStLit.add flattenElem(srcLay, ident"stride", li, R)
    dstStLit.add flattenElem(dstLay, ident"stride", li, R)
  stmts.add newLetStmt(shSym, shapeLit)
  stmts.add newLetStmt(srcStSym, srcStLit)
  stmts.add newLetStmt(dstStSym, dstStLit)
  stmts.add genNestedCopy(R, bs, shSym, srcStSym, dstStSym, dstData, srcData)
  result = stmts

proc genContiguityCode(
    effR: int;
    shapeVals, srcStVals, dstStVals: seq[int];
    shSym, srcStSym, dstStSym: NimNode;
    dstData, srcData: NimNode;
    bs: int
): NimNode {.compileTime.} =
  ## Detect contiguous suffix and fuse into a single copyMem.
  #
  # Walk dims innermost-first. While both src and dst strides match
  # the cumulative inner product (innermost has stride-1), extend the
  # contiguous suffix window inward. Dynamic shapes force a break.
  var copyDimStart = effR
  var innerProd = 1
  for d in countdown(effR - 1, 0):
    if shapeVals[d] == DynamicSentinel:
      break  # dynamic shape → can't verify contiguity at compile time
    if srcStVals[d] == innerProd and dstStVals[d] == innerProd:
      innerProd *= shapeVals[d]
      copyDimStart = d
    else:
      break

  # ── Some suffix was contiguous: fuse into copyMem ──
  if copyDimStart < effR:
    let outerR = copyDimStart
    var copyCount = newLit(1)
    for d in copyDimStart ..< effR:
      let cd = newLit(d)
      copyCount = quote do: `copyCount` * `shSym`[`cd`]
    result = genCopyMemLoops(dstData, srcData, shSym, dstStSym, srcStSym,
                             outerR, copyCount)

  # ── No contiguous suffix: fully strided loops ──
  else:
    result = genNestedCopy(effR, bs, shSym, srcStSym, dstStSym, dstData, srcData)
macro copySameShapeImpl(dst: typed; src: typed; blockSize: static int): untyped =
  let tvDst = dst.getTypeInst()
  let tvSrc = src.getTypeInst()
  let rawSrcSh = toSeqStaticInts(tvSrc[2])
  let rawSrcSt = toSeqStaticInts(tvSrc[3])
  let rawDstSh = toSeqStaticInts(tvDst[2])
  let rawDstSt = toSeqStaticInts(tvDst[3])
  let R = max(flatTupleLen(tvSrc[2]), flatTupleLen(tvDst[2]))
  let bs = blockSize

  # ── Extract data pointers and layout references ──
  let dstData = newTree(nnkDotExpr, dst, ident"data")
  let srcData = newTree(nnkDotExpr, src, ident"data")
  let srcLay = newTree(nnkDotExpr, src, ident"layout")
  let dstLay = newTree(nnkDotExpr, dst, ident"layout")

  # ── Build stride-sorted shape/strides arrays ──
  let a = buildStrideSortedArrays(
    R, rawSrcSh, rawSrcSt, rawDstSh, rawDstSt,
    srcLay, dstLay, permDtoS = @[])
  if a.effR == 0:
    # All dims are size-1 → single element copy
    result = newStmtList()
    result.add quote do:
      `dstData`[0] = `srcData`[0]
    return
  # ── Bind local arrays, then generate loops via contiguity code ──
  let shSym = genSym(nskLet, "sh")
  let srcStSym = genSym(nskLet, "srcSt")
  let dstStSym = genSym(nskLet, "dstSt")
  var stmts = newStmtList()
  stmts.add newLetStmt(shSym, a.shLit)
  stmts.add newLetStmt(srcStSym, a.srcStLit)
  stmts.add newLetStmt(dstStSym, a.dstStLit)

  stmts.add genContiguityCode(a.effR, a.shapeVals, a.srcStVals, a.dstStVals,
                              shSym, srcStSym, dstStSym, dstData, srcData, bs)
  result = stmts

# ── copyPermutedImpl: private bridge macro ──

macro copyPermutedImpl[Rank: static int](
    dst, src: typed;
    permDtoS: static array[Rank, int];
    blockSize: static int
): untyped =
  let tvDst = dst.getTypeInst()
  let tvSrc = src.getTypeInst()
  let rawDstSh = toSeqStaticInts(tvDst[2])
  let rawDstSt = toSeqStaticInts(tvDst[3])
  let rawSrcSh = toSeqStaticInts(tvSrc[2])
  let rawSrcSt = toSeqStaticInts(tvSrc[3])
  let R = Rank
  let bs = blockSize
  let dstData = newTree(nnkDotExpr, dst, ident"data")
  let srcData = newTree(nnkDotExpr, src, ident"data")
  let dstLay = newTree(nnkDotExpr, dst, ident"layout")
  let srcLay = newTree(nnkDotExpr, src, ident"layout")


  # Convert static perm to seq for shared helper
  var permSeq = newSeq[int](R)
  for i in 0 ..< R: permSeq[i] = permDtoS[i]

  let a = buildStrideSortedArrays(
    R, rawSrcSh, rawSrcSt, rawDstSh, rawDstSt,
    srcLay, dstLay, permDtoS = permSeq)
  if a.effR == 0:
    # All dims are size-1 → single element copy
    result = quote do:
      `dstData`[0] = `srcData`[0]
    return
  let shSym = genSym(nskLet, "sh")
  let dstStSym = genSym(nskLet, "dstSt")
  let srcStSym = genSym(nskLet, "srcSt")
  var stmts = newStmtList()
  stmts.add newLetStmt(shSym, a.shLit)
  stmts.add newLetStmt(dstStSym, a.dstStLit)
  stmts.add newLetStmt(srcStSym, a.srcStLit)

  stmts.add genContiguityCode(a.effR, a.shapeVals, a.srcStVals, a.dstStVals,
                              shSym, srcStSym, dstStSym, dstData, srcData, bs)
  result = stmts
#
#                    Public functions
#
#################################################################

# ── copySameShape_cpu: public func ─────────────────────────

func copySameShape_cpu*[D, S: TensorView](
    dst: D, src: S, blockSize: static int = -1
) =
  copySameShapeImpl(dst, src, blockSize)

# ── copyPermuted_cpu: public func ──────────────────────────────────

func copyPermuted_cpu*[Rank: static int, D, S: TensorView](
    dst: D, src: S;
    permDtoS: static array[Rank, int]; blockSize: static int = -1
) =
  copyPermutedImpl(dst, src, permDtoS, blockSize)

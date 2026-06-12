import std/[macros, algorithm]
import ../../src/int_tuples
import ../../src/layouts
import ../../src/tensors

{.experimental: "callOperator".}

# ── compile-time helpers ────────────────────────────────────────
proc strideSortDesc(dstSt: seq[int]): seq[int] =
  ## Sort mode indices by dstStride descending.
  ## Innermost loop = smallest dst stride (cache-friendly writes).
  for i in 0 ..< dstSt.len: result.add i
  for i in 0 ..< result.len:
    for j in i + 1 ..< result.len:
      if dstSt[result[i]] < dstSt[result[j]]:
        swap result[i], result[j]

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

proc flattenElem(lay, field: NimNode; idx: NimNode): NimNode {.compileTime.} =
  ## Generates `int(flatten(lay.field)[idx])`.
  result = quote do:
    int(flatten(`lay`.`field`)[`idx`])

proc hasAllNonZeroStrides(rawStrides: seq[int]): bool {.compileTime.} =
  ## True when all strides are compile-time non-zero.
  if rawStrides.len == 0: return false
  for v in rawStrides:
    if v == DynamicSentinel: return false
  return true

proc buildStrideSortedArrays(
    R: int; order: seq[int];
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
  ## Flow:
  ##   1. Walk dims in `order` (stride-sorted by dst stride).
  ##   2. Skip compile-time size-1 dims (trivial loops that would
  ##      iterate once — they clutter the loop nest for no benefit).
  ##   3. For each kept dim, emit shape/dstSt/srcSt into parallel arrays.
  ##      Unknown (runtime) values get DynamicSentinel in `Vals` seq
  ##      and a `flattenElem()` call in the `Lit` NimNode (codegen).
  ##   4. Post-loop: reorder so unknown-stride dims come FIRST
  ##      (outermost loops — fewer iterations, conservative placement)
  ##      and known-stride dims are stride-sorted descending (innermost).
  ##
  ## Parameters:
  ##   R: rank (number of dimensions)
  ##   order: dim indices in processing order (strideSortDesc output,
  ##          or identity [0..R-1] when strides are unknown)
  ##   rawSrcSh/rawSrcSt: compile-time shape/stride values from `toSeqStaticInts`,
  ##          or empty seq if fully dynamic.
  ##   rawDstSh/rawDstSt: same for dst.
  ##   srcLay/dstLay: NimNode for `src.layout` / `dst.layout` (runtime fallback).
  ##   permDtoS: permutation for src strides ([1,0,2,3] = swap dims 0,1).
  ##          Empty = identity (no permutation).
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
    let ok = order[k]
    let shVal = if ok < rawDstSh.len: rawDstSh[ok] else: DynamicSentinel
    # Skip compile-time-known size-1 dims (trivial loops)
    if shVal != DynamicSentinel and shVal == 1:
      continue
    okVals.add ok
    result.effR += 1
    # Shape (always from dst — we write to dst)
    if ok < rawDstSh.len and rawDstSh[ok] != DynamicSentinel:
      result.shLit.add newLit(rawDstSh[ok])
      result.shapeVals.add rawDstSh[ok]
    else:
      result.shLit.add flattenElem(dstLay, ident"shape", newLit(ok))
      result.shapeVals.add DynamicSentinel
    # Dst stride
    if ok < rawDstSt.len and rawDstSt[ok] != DynamicSentinel:
      result.dstStLit.add newLit(rawDstSt[ok])
      result.dstStVals.add rawDstSt[ok]
    else:
      result.dstStLit.add flattenElem(dstLay, ident"stride", newLit(ok))
      result.dstStVals.add DynamicSentinel
    # Src stride (with optional permutation)
    let srcPos = if permDtoS.len > 0: permDtoS[ok] else: ok
    if srcPos < rawSrcSt.len and rawSrcSt[srcPos] != DynamicSentinel:
      result.srcStLit.add newLit(rawSrcSt[srcPos])
      result.srcStVals.add rawSrcSt[srcPos]
    else:
      result.srcStLit.add flattenElem(srcLay, ident"stride", newLit(srcPos))
      result.srcStVals.add DynamicSentinel
  # Reorder: unknown-stride dims outermost, sorted known-stride dims inner
  if result.effR > 1:
    var known, unknown: seq[int]
    for d in 0 ..< result.effR:
      if result.dstStVals[d] == DynamicSentinel: unknown.add d
      else: known.add d
    if unknown.len > 0 and known.len > 0:
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
  if R == 0:
    return newEmptyNode()
  let indices = nnkBracket.newTree()
  for d in 0 ..< R:
    indices.add genSym(nskForVar, "i" & $d & "_")
  if R == 1:
    let i0 = indices[0]
    if blockSize <= 0:
      return quote do:
        for `i0` in 0 ..< `shape`:
          `dstPtr`[`i0` * `dstStrides`] = `srcPtr`[`i0` * `srcStrides`]
    let i0b = genSym(nskForVar, "i0_block_")
    return quote do:
      for `i0b` in countup(0, `shape` - 1, `blockSize`):
        for `i0` in `i0b` ..< min(`i0b` + `blockSize`, `shape`):
          `dstPtr`[`i0` * `dstStrides`] = `srcPtr`[`i0` * `srcStrides`]
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
  if blockSize <= 0:
    result = body
    for i in countdown(R - 1, 0):
      let idx = indices[i]
      let si = newLit(i)
      result = quote do:
        for `idx` in 0 ..< `shape`[`si`]:
          `result`
  else:
    var blockIdx: seq[NimNode]
    for i in 0 ..< R:
      blockIdx.add genSym(nskForVar, "i" & $i & "_block_")
    result = body
    for i in countdown(R - 1, 0):
      let idx = indices[i]
      let bIdx = blockIdx[i]
      let si = newLit(i)
      result = quote do:
        for `idx` in `bIdx` ..< min(`bIdx` + `blockSize`, `shape`[`si`]):
          `result`
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
    let sh = flattenElem(srcLay, ident"shape", newLit(0))
    let sst = flattenElem(srcLay, ident"stride", newLit(0))
    let dst = flattenElem(dstLay, ident"stride", newLit(0))
    return genNestedCopy(R, bs, sh, sst, dst, dstData, srcData)
  var stmts = newStmtList()
  let shSym = genSym(nskLet, "sh")
  let srcStSym = genSym(nskLet, "srcSt")
  let dstStSym = genSym(nskLet, "dstSt")
  var shapeLit = nnkBracket.newTree()
  var srcStLit = nnkBracket.newTree()
  var dstStLit = nnkBracket.newTree()
  for i in 0 ..< R:
    let li = newLit(i)
    shapeLit.add flattenElem(srcLay, ident"shape", li)
    srcStLit.add flattenElem(srcLay, ident"stride", li)
    dstStLit.add flattenElem(dstLay, ident"stride", li)
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
  var copyDimStart = effR
  var innerProd = 1
  for d in countdown(effR - 1, 0):
    if srcStVals[d] == innerProd and dstStVals[d] == innerProd:
      innerProd *= shapeVals[d]
      copyDimStart = d
    else:
      break
  if copyDimStart < effR:
    let outerR = copyDimStart
    var copyCount = newLit(1)
    for d in copyDimStart ..< effR:
      let cd = newLit(d)
      copyCount = quote do: `copyCount` * `shSym`[`cd`]
    result = genCopyMemLoops(dstData, srcData, shSym, dstStSym, srcStSym,
                             outerR, copyCount)
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
  let dstData = newTree(nnkDotExpr, dst, ident"data")
  let srcData = newTree(nnkDotExpr, src, ident"data")

  let hasStrides = hasAllNonZeroStrides(rawDstSt)
  var order: seq[int]
  if hasStrides:
    order = strideSortDesc(rawDstSt)
  else:
    order = newSeq[int](R)
    for i in 0 ..< R: order[i] = i
  let srcLay = newTree(nnkDotExpr, src, ident"layout")
  let dstLay = newTree(nnkDotExpr, dst, ident"layout")
  let a = buildStrideSortedArrays(
    R, order, rawSrcSh, rawSrcSt, rawDstSh, rawDstSt,
    srcLay, dstLay, permDtoS = @[])
  if a.effR == 0:
    result = newEmptyNode()
    return
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

  let hasStrides = hasAllNonZeroStrides(rawDstSt)
  var order: seq[int]
  if hasStrides:
    order = strideSortDesc(rawDstSt)
  else:
    order = newSeq[int](R)
    for i in 0 ..< R: order[i] = i

  # Convert static perm to seq for shared helper
  var permSeq = newSeq[int](R)
  for i in 0 ..< R: permSeq[i] = permDtoS[i]

  let a = buildStrideSortedArrays(
    R, order, rawSrcSh, rawSrcSt, rawDstSh, rawDstSt,
    srcLay, dstLay, permDtoS = permSeq)
  if a.effR == 0:
    result = newEmptyNode()
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

# ── copySameShape: public func ─────────────────────────

func copySameShape*[D, S: TensorView](
    dst: D, src: S, blockSize: static int = -1
) =
  copySameShapeImpl(dst, src, blockSize)

# ── copyPermuted: public func ──────────────────────────────────

func copyPermuted*[Rank: static int, D, S: TensorView](
    dst: D, src: S;
    permDtoS: static array[Rank, int]; blockSize: static int = -1
) =
  copyPermutedImpl(dst, src, permDtoS, blockSize)

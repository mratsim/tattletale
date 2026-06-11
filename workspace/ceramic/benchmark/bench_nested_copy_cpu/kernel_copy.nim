import std/[macros]
import ../../src/int_tuples
import ../../src/layouts
import ../../src/layout_algebra
import ../../src/tensors

{.experimental: "callOperator".}

# ── compile-time helpers ────────────────────────────────────────

proc filterOnesSeq(vals: seq[int]): seq[int] =
  ## Filter out size-1 dimensions; return @[1] if all were 1.
  for v in vals:
    if v == 0: return @[]
    if v > 1: result.add v
  if result.len == 0: result.add 1

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

# ── loop generation (compile-time NimNode builder) ──────────────

proc genNestedCopy(
    R, blockSize: int;
    shape, srcStrides, dstStrides: NimNode;
    dstPtr, srcPtr: NimNode
): NimNode =
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

proc runtimeCopyArgs(
    dst, src: NimNode; R, bs: int;
    dstData, srcData: NimNode
): NimNode {.compileTime.} =
  ## Build genNestedCopy call using runtime layout.shape/stride.
  let srcLay = newTree(nnkDotExpr, src, ident"layout")
  let dstLay = newTree(nnkDotExpr, dst, ident"layout")
  let shapeArr = nnkBracket.newTree()
  let srcStArr = nnkBracket.newTree()
  let dstStArr = nnkBracket.newTree()
  for i in 0 ..< R:
    let li = newLit(i)
    shapeArr.add flattenElem(srcLay, ident"shape", li)
    srcStArr.add flattenElem(srcLay, ident"stride", li)
    dstStArr.add flattenElem(dstLay, ident"stride", li)
  result = genNestedCopy(R, bs, shapeArr, srcStArr, dstStArr, dstData, srcData)

# ── copySameShapeImpl: private bridge macro ─────────────────────

macro copySameShapeImpl(dst: typed; src: typed; blockSize: static int): untyped =
  let tvDst = dst.getTypeInst()
  let tvSrc = src.getTypeInst()
  let rawSrcSh = toSeqStaticInts(tvSrc[2])
  let rawSrcSt = toSeqStaticInts(tvSrc[3])
  let rawDstSh = toSeqStaticInts(tvDst[2])
  let rawDstSt = toSeqStaticInts(tvDst[3])
  let srcCompactSh = filterOnesSeq(rawSrcSh)
  let srcCompactSt = filterOnesSeq(rawSrcSt)
  let dstCompactSh = filterOnesSeq(rawDstSh)
  let dstCompactSt = filterOnesSeq(rawDstSt)
  let allStatic = srcCompactSh.len > 0 and dstCompactSh.len > 0
  let R = if allStatic: srcCompactSh.len else: max(flatTupleLen(tvSrc[2]), flatTupleLen(tvDst[2]))
  let bs = blockSize
  let dstData = newTree(nnkDotExpr, dst, ident"data")
  let srcData = newTree(nnkDotExpr, src, ident"data")
  if allStatic and R > 0 and dstCompactSh.len == R:
    # Same-shape: dim N ↔ dim N by position (identity permutation).
    let order = strideSortDesc(dstCompactSt)
    var shapeSeq, srcStSeq, dstStSeq: seq[int]
    for k in 0 ..< R:
      let ok = order[k]
      shapeSeq.add srcCompactSh[ok]
      srcStSeq.add srcCompactSt[ok]
      dstStSeq.add dstCompactSt[ok]
    result = genNestedCopy(
      R, bs,
      toBracket(shapeSeq), toBracket(srcStSeq), toBracket(dstStSeq),
      dstData, srcData)
  else:
    result = runtimeCopyArgs(dst, src, R, bs, dstData, srcData)

macro copyPermutedImpl[Rank: static int](
    dst, src: typed;
    permDtoS: static array[Rank, int];
    blockSize: static int
): untyped =
  let tvDst = dst.getTypeInst()
  let tvSrc = src.getTypeInst()
  let rawDstSh = toSeqStaticInts(tvDst[2])
  let rawDstSt = toSeqStaticInts(tvDst[3])
  let rawSrcSt = toSeqStaticInts(tvSrc[3])
  let dstCompactSh = filterOnesSeq(rawDstSh)
  let dstCompactSt = filterOnesSeq(rawDstSt)
  let srcCompactSt = filterOnesSeq(rawSrcSt)
  let allStatic = dstCompactSh.len == Rank and srcCompactSt.len == Rank
  let R = Rank
  let bs = blockSize
  let dstData = newTree(nnkDotExpr, dst, ident"data")
  let srcData = newTree(nnkDotExpr, src, ident"data")
  if allStatic:
    var srcStPerm = newSeq[int](R)
    for d in 0 ..< R:
      srcStPerm[d] = srcCompactSt[permDtoS[d]]
    let order = strideSortDesc(dstCompactSt)
    var shapeSeq, dstStSeq, srcStSeq: seq[int]
    for k in 0 ..< R:
      let ok = order[k]
      shapeSeq.add dstCompactSh[ok]
      dstStSeq.add dstCompactSt[ok]
      srcStSeq.add srcStPerm[ok]
    result = genNestedCopy(
      R, bs,
      toBracket(shapeSeq), toBracket(srcStSeq), toBracket(dstStSeq),
      dstData, srcData)
  else:
    let dstLay = newTree(nnkDotExpr, dst, ident"layout")
    let srcLay = newTree(nnkDotExpr, src, ident"layout")
    let dstShArr = nnkBracket.newTree()
    let dstStArr = nnkBracket.newTree()
    for i in 0 ..< R:
      let li = newLit(i)
      dstShArr.add flattenElem(dstLay, ident"shape", li)
      dstStArr.add flattenElem(dstLay, ident"stride", li)
    var srcStPerm = nnkBracket.newTree()
    for d in 0 ..< R:
      srcStPerm.add flattenElem(srcLay, ident"stride", newLit(permDtoS[d]))
    result = genNestedCopy(R, bs, dstShArr, srcStPerm, dstStArr, dstData, srcData)

#################################################################
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

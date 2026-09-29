# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layout algebra: coalesce, filter_zeros, filter, sort.

import std/macros
import std/sequtils
import std/algorithm
import std/typetraits
import workspace/ceramic/src/int_tuples
import ./layouts

# ═══════════════════════════════════════════════════════════════
#  getIndicesSortedByStride — sort permutation by stride
# ═══════════════════════════════════════════════════════════════

proc getIndicesSortedByStride(strides: seq[int]): seq[int] {.compileTime.} =
  ## Return indices sorted by stride ascending.
  ## Data stays in original arrays — just iterate this permutation.
  result = newSeq[int](strides.len)
  for i in 0 ..< result.len:
    result[i] = i
  for i in 0 ..< result.len:
    for j in i + 1 ..< result.len:
      if strides[result[i]] > strides[result[j]]:
        swap result[i], result[j]

#  coalesce — merge contiguous dimensions where stride matches
# ═══════════════════════════════════════════════════════════════

macro coalesceBackward(layoutShape, layoutStride: typed; preserveTrailing: static bool = false): untyped =
  var shLeaves, shTypes, stLeaves, stTypes: seq[NimNode]
  for (leaf, ty) in flatLeavesRev(layoutShape):
    shLeaves.add leaf
    shTypes.add ty
  for (leaf, ty) in flatLeavesRev(layoutStride):
    stLeaves.add leaf
    stTypes.add ty

  if shLeaves.len == 1 and stLeaves.len == 1:
    if isStaticOne(shTypes[0]):
      result = newCall(bindSym"make_layout", newLit(1), newLit(0))
    else:
      result = newCall(bindSym"make_layout", shLeaves[0], stLeaves[0])
    return

  var resShapes: seq[NimNode] = @[]
  var resStrides: seq[NimNode] = @[]
  var resSTypes: seq[NimNode] = @[]
  var resSTypes2: seq[NimNode] = @[]

  resShapes.add shLeaves[0]
  resStrides.add stLeaves[0]
  resSTypes.add shTypes[0]
  resSTypes2.add stTypes[0]

  if preserveTrailing:
    if isStaticOne(shTypes[0]):
      resShapes[0] = IntCT(low(int))
      resSTypes[0] = newNimNode(nnkBracketExpr).add(ident"Int", newLit(low(int)))

  for k in 1 ..< shLeaves.len:
    let curST = shTypes[k]
    let curST2 = stTypes[k]

    if isStaticOne(curST):
      continue

    if isStaticOne(resSTypes[0]):
      resShapes[0] = shLeaves[k]
      resStrides[0] = stLeaves[k]
      resSTypes[0] = curST
      resSTypes2[0] = curST2
      continue

    if isStaticInt(curST) and isStaticInt(curST2) and
       isStaticInt(resSTypes2[0]) and isStaticInt(resSTypes[0]):
      let curProd = getStaticInt(curST) * getStaticInt(curST2)
      if curProd == getStaticInt(resSTypes2[0]):
        let mergedVal = getStaticInt(curST) * getStaticInt(resSTypes[0])
        let mergedNode = IntCT(mergedVal)
        resShapes[0] = mergedNode
        resStrides[0] = stLeaves[k]
        resSTypes[0] = newNimNode(nnkBracketExpr).add(ident"Int", newLit(mergedVal))
        resSTypes2[0] = curST2
        continue

    resShapes.insert(shLeaves[k], 0)
    resStrides.insert(stLeaves[k], 0)
    resSTypes.insert(curST, 0)
    resSTypes2.insert(curST2, 0)

  if not preserveTrailing:
    while resShapes.len > 0 and isStaticOne(resSTypes[^1]):
      discard resShapes.pop()
      discard resStrides.pop()
      discard resSTypes.pop()
      discard resSTypes2.pop()

  if resShapes.len == 0:
    result = newCall(bindSym"make_layout", IntCT(1), newLit(0))
    return

  var rShape = newNimNode(nnkTupleConstr)
  var rStride = newNimNode(nnkTupleConstr)
  for idx in 0 ..< resShapes.len:
    rShape.add resShapes[idx]
    rStride.add resStrides[idx]
  if rShape.len == 1:
    rShape = rShape[0]
    rStride = rStride[0]

  result = newCall(bindSym"make_layout", rShape, rStride)

func coalesce*(layout: Layout): auto {.inline, noInit.} =
  ## Merge contiguous dimensions.
  coalesceBackward(layout.shape, layout.stride)

func coalesce_preserve_trailing(layout: Layout): auto {.inline, noInit.} =
  ## Like `coalesce` but preserves trailing size-1 dimensions (e.g. stride-0 broadcasts).
  coalesceBackward(layout.shape, layout.stride, preserveTrailing = true)

# ═══════════════════════════════════════════════════════════════
#  filter_inactive — remove stride-0 and size-1 dimensions
# ═══════════════════════════════════════════════════════════════

func filter_inactive*(layout: Layout): auto {.inline.} =
  ## Remove stride-0 and size-1 dimensions
  coalesce(filter_zeros(layout))

# ═══════════════════════════════════════════════════════════════
#  complement
# ═══════════════════════════════════════════════════════════════
#
#  ## Flow
#  ##
#  ##   complement(sh, st, cosizeBound) [sh/st = flattened shape/stride]
#  ##        │
#  ##        ├─ scalar (rank-1) ───────────────────────────────────┐
#  ##        │    │                                                │
#  ##        │    ├─ st == Int[0] (broadcast) ──► Layout(bound, 1) │
#  ##        │    │                                                │
#  ##        │    └─ gap = max(1, st)                              │
#  ##        │       prd = st * sh                                 │
#  ##        │       rem = ceil_div(bound, prd)                    │
#  ##        │       result = Layout((gap, rem), (1, st))          │
#  ##        │                                                     │
#  ##        └─ multi-dimension                                         │
#  ##             │                                                │
#  ##             ├─ allStridesStatic? ── must be (doAssert)       │
#  ##             │                                                │
#  ##             ├─ filterIt(shVal ≠ 1 and stVal ≠ 0)             │
#  ##             ├─ sort by (stVal, shVal)                        │
#  ##             │                                                │
#  ##             ├─ scan:                                         │
#  ##             │    gap = stVal / cur  (emit if > 1)            │
#  ##             │    cur *= shVal                                │
#  ##             │                                                │
#  ##             ├─ allShapesStatic?                              │
#  ##             │    YES ──► rem = ceil_div(bound, cur)          │
#  ##             │              emit (rem, cur)                   │
#  ##             │    NO  ──► halt at first dynamic shape         │
#  ##             │              emit rem = ceil_div(bound, cur)   │
#  ##             │              as runtime expression, then break │
#  ##             │                                                │
#  ##             └─ coalesce result                               │
#  ##
#  ## Static/dynamic rules:
#  ##   - All-static: full compile-time computation
#  ##   - Dynamic strides: compile-time error (CuTe: static_assert)
#  ##   - Dynamic shapes + static strides: partial runtime
#  ##   - Rank-1 dynamic: runtime via func instantiation
#  ##
#  ## cosizeBound type:
#  ##   In the multi-dimension path, bound may be Int[N] (static) or int (dynamic).
#  ##   For Int[N] bounds, ceil_div is computed at compile time.
#  ##   For int bounds, a runtime ceil_div expression is emitted.
#  ##   Shape-tuple bounds (e.g. (32,4,4)) are converted to size(product).
# ═══════════════════════════════════════════════════════════════

proc complementScalar(sh, st, boundExpr: NimNode): NimNode {.compileTime.} =
  ## Standard CuTe scalar complement formula.
  ##   gap = max(1, st);  prd = st * sh;  rem = ceil_div(bound, prd)
  ##   result = coalesce(Layout((gap, rem), (1, prd)))
  let stTyp = st.getTypeInst()

  if stTyp.kind == nnkBracketExpr and $stTyp[0] == "Int" and stTyp[1].intVal == 0:
    return newCall(bindSym"make_layout", boundExpr, newLit(1))

  template leafV(typ, node: NimNode): int =
    ## Static leaf value from an Int[V] type or an int literal, -1 marks runtime
    if typ.kind == nnkBracketExpr and $typ[0] == "Int":
      typ[1].intVal.int
    elif node.kind == nnkIntLit:
      node.intVal.int
    else:
      -1

  let shTyp = sh.getTypeInst()
  let stV = leafV(stTyp, st)
  let shV = leafV(shTyp, sh)
  let boundV =
    if boundExpr.kind == nnkIntLit: boundExpr.intVal.int
    elif boundExpr.kind == nnkSym and boundExpr.getTypeInst.kind == nnkBracketExpr and
        $boundExpr.getTypeInst[0] == "Int":
      boundExpr.getTypeInst[1].intVal.int
    elif boundExpr.kind == nnkCall and boundExpr[0].kind == nnkBracketExpr and
        $boundExpr[0][0] == "Int":
      boundExpr[0][1].intVal.int
    else:
      -1
  if stV >= 1 and shV >= 1:
    let gapV = max(1, stV)
    let prdV = stV * shV
    if boundV >= 1:
      let remV = (boundV + prdV - 1) div prdV
      if remV == 1 and gapV == 1:
        return newCall(bindSym"make_layout", newLit(1), newLit(0))
      elif remV == 1:
        return newCall(bindSym"make_layout", newLit(gapV), newLit(1))
      elif gapV == 1:
          return newCall(bindSym"make_layout", newLit(remV), newLit(prdV))
      elif prdV == gapV:
          return newCall(bindSym"make_layout", newLit(gapV * remV), newLit(1))
      else:
        return newCall(bindSym"make_layout",
          newTree(nnkTupleConstr, newLit(gapV), newLit(remV)),
          newTree(nnkTupleConstr, newLit(1), newLit(prdV)))
    let remExpr = newCall(bindSym"ceil_div", boundExpr, newLit(prdV))
    if gapV == 1:
      return newCall(bindSym"make_layout", remExpr, newLit(prdV))
    elif prdV == gapV:
      return newCall(bindSym"make_layout",
        newCall(bindSym"*", newLit(gapV), remExpr), newLit(1))
    else:
      return newCall(bindSym"make_layout",
        newTree(nnkTupleConstr, newLit(gapV), remExpr),
        newTree(nnkTupleConstr, newLit(1), newLit(prdV)))

  let gap = newCall(bindSym"max", newLit(1), st)
  let prd = newCall(bindSym"*", st, sh)
  let rem = newCall(bindSym"ceil_div", boundExpr, prd)
  newCall(bindSym"coalesce",
    newCall(bindSym"make_layout",
      newTree(nnkTupleConstr, gap, rem),
      newTree(nnkTupleConstr, newLit(1), prd)))

# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
#  Pure seq-based helpers (no NimNode manipulation)
# ═══════════════════════════════════════════════════════════════

proc complementGaps(
    strides, shapes: seq[int]; shNode, boundExpr: NimNode): LayoutCT {.compileTime.} =
  ## Build full complement LayoutCT (gap dimensions + remainder), folding over
  ## dimensions in ascending-stride order: each dimension contributes a gap dimension
  ## (stride div cur, cur) and advances cur = stride * shape.
  ## Runtime shapes advance cur with a runtime expression, the fold
  ## continues past them.
  ## Statically-1 dimensions are skipped, runtime dimensions are appended unconditionally.
  var cur = 1
  var curNode: NimNode = IntCT(1)
  var curStatic = true
  result = LayoutCT()
  for idx in getIndicesSortedByStride(strides):
    if curStatic:
      let gap = if strides[idx] > cur: strides[idx] div cur else: 1
      if gap > 1:
        result.append(IntCT(gap), IntCT(cur))
    else:
      # cur is a runtime expression — the gap cannot be proven statically,
      # so it is emitted unconditionally.
      result.append(newCall(bindSym"div", IntCT(strides[idx]), curNode), curNode)
    if shapes[idx] == DynamicSentinel:
      # runtime shape: cur becomes a runtime expression
      curNode = newCall(bindSym"*", IntCT(strides[idx]),
                        newTree(nnkBracketExpr, shNode, newLit(idx)))
      curStatic = false
    else:
      # static shape: cur folds back to Int[N]
      cur = strides[idx] * shapes[idx]
      curNode = IntCT(cur)
      curStatic = true
  let rem = newCall(bindSym"ceil_div", boundExpr, curNode)
  result.append(rem, curNode)

proc complementMulti(sh, st, boundExpr: NimNode): NimNode {.compileTime.} =
  ## Multi-dimension complement: sort by stride, fold to fill gaps.
  ## All strides must be static Int[N] (compile-time check).
  let stTyp = st.getTypeInst()
  let shTyp = sh.getTypeInst()

  doAssert stTyp.kind == nnkTupleConstr,
    "complementMulti: expected tuple type for strides"
  for i in 0 ..< stTyp.len:
    let stNode = stTyp[i]
    doAssert stNode.kind == nnkBracketExpr and $stNode[0] == "Int",
      "complement: multi-dimension with dynamic strides not supported at index " & $i

  let strides = toSeqStaticInts(stTyp)
  let shapes  = toSeqStaticInts(shTyp)
  let acc = complementGaps(strides, shapes, sh, boundExpr)
  newCall(bindSym"coalesce", acc.emit())

macro complementImpl(sh, st, cosizeBound: typed): untyped =
  ## Dispatch to scalar or multi-dimension complement.
  let boundExpr =
    if cosizeBound.getTypeInst().kind == nnkTupleConstr:
      newCall(bindSym"product", cosizeBound)
    else:
      cosizeBound
  if sh.getTypeInst().kind != nnkTupleConstr:
    complementScalar(sh, st, boundExpr)
  else:
    complementMulti(sh, st, boundExpr)

func complement*(layout: Layout; cosizeBound: Int or int): auto =
  ## Compute complement: fills stride gaps up to cosizeBound.
  ## Filters inactive dimensions first (matches CuTe's filter-before-complement).
  let f = filter_inactive(layout)
  complementImpl(flatten(f.shape), flatten(f.stride), cosizeBound)

func complement*(layout: Layout; cosizeBound: static int): auto =
  ## Compile-time int overload: wrap in Int[N] to preserve constness.
  complement(layout, Int[cosizeBound]())

func complement*(layout: Layout): auto =
  ## Compute complement with default bound = cosize(filtered layout).
  let f = filter_inactive(layout)
  complementImpl(flatten(f.shape), flatten(f.stride), cosize(f))

func complement*(layout: Layout; cosizeBound: tuple): auto =
  ## Compute complement with a shape-tuple bound (size converted to product).
  let f = filter_inactive(layout)
  complementImpl(flatten(f.shape), flatten(f.stride), cosizeBound)

# ═══════════════════════════════════════════════════════════════
#  compose — layout composition
# ═══════════════════════════════════════════════════════════════

##
## `compose(A, B)` produces a layout `R` such that `R(i) = A(B(i))`
## for all `i` in `0..cosize(B)-1`.
##
## Algorithm (mirrors CuTe C++ `composition_impl`):
## ```
##         ┌──────────────────────────────────────────────┐
##         │           compose(A, B)                      │
##         │              R(i) = A(B(i))                  │
##         └──────────────────────┬───────────────────────┘
##                                │
##                    ┌───────────┴───────────┐
##                    │                       │
##               scalar LHS             tuple LHS
##                    │                       │
##                    ▼                       ▼
##         make_layout(              fold over dimensions
##         B.shape,                   0..R-2 of A:
##         B.stride ×                 ┌──────────────┐
##         A.stride)                  │ currShape    │
##                                   │ currStride   │
##                                   │ absRemStride │
##                                   │ nextShape    │
##                                   │ clampedShape │
##                                   │              │
##                                   │ rSh.append   │
##                                   │ rSt.append   │
##                                   │ remShape /=  │
##                                   │ remStride *= │
##                                   └──────┬───────┘
##                                          │
##                                   ┌──────┴───────┐
##                                   │ Last dimension    │
##                                   │ (R-1):       │
##                                   │ append       │
##                                   │ remShape     │
##                                   │ remStride ×  │
##                                   │ lastStride   │
##                                   └──────┬───────┘
##                                          │
##                                          ▼
##                               make_layout(rSh, rSt)
## ```
##
##            (fold(make_seq<R-1>{}, ...) + append remainder)

template divisibilityCheck(remainingShape, clampedShape: untyped) =
  ## Python tensor-layouts compatible divisibility check.
  ## Static leaves assert at compile time, runtime shapes are not asserted,
  ## device code carries no doAssert, correctness is by construction.
  when clampedShape is Int:
    when typeof(clampedShape).V == 1:
      discard  # shape 1 is trivially divisor
    elif remainingShape is Int:
      static: doAssert typeof(remainingShape).V mod typeof(clampedShape).V == 0,
        "compose: shape " & $typeof(remainingShape).V & " and consumed shape " & $typeof(clampedShape).V & " are not divisible"

macro composeImpl(remainingShape, remainingStride: untyped; lhsShapes, lhsStrides: typed): untyped =
  ## Fold over LHS dimensions with a 2-state accumulator, the remaining
  ## shape and stride, emitting one (shape, stride) dimension pair per
  ## unconsumed LHS dimension.
  ##
  ## Per-dimension flow for dimension d of R, where
  ## nextSh = ceil_div(lhsSh[d], |remSt|) and
  ## consumed = (nextSh is Int[1] or remSh is Int[1]).
  ## - consumed → carry (remSh, remSt ← nextSt), nothing emitted
  ## - otherwise emit (min(nextSh, remSh), remSt * lhsSt[d]),
  ##   then remSh ← remSh div clamped, remSt ← nextSt
  ##
  ## Returns the composed layout as an untyped node.
  ##
  ## - The value parameters are untyped. Typed macro parameters arrive
  ##   nil for arguments whose types are still being computed

  var lhsShLeaves, lhsStLeaves: seq[NimNode]
  for (leaf, _) in flatLeaves(lhsShapes):
    lhsShLeaves.add leaf
  for (leaf, _) in flatLeaves(lhsStrides):
    lhsStLeaves.add leaf
  let R = lhsShLeaves.len

  template consumeStep(currShLeaf, currStLeaf, remSh, remSt,
                       currShape, currStride, absRem, nextSh, nextSt,
                       clamped, remSh2, scaled, skipBody, elseBody) =
    let currShape = currShLeaf
    let currStride = currStLeaf
    let absRem = abs(remSt)
    let nextSh = ceil_div(currShape, absRem)
    when nextSh is Int and typeof(nextSh) is Int[1] or
        remSh is Int and typeof(remSh) is Int[1]:
      skipBody
    else:
      elseBody

  template consumeSkip(nextSt, absRem, currShape, remSt, tail) =
    let nextSt = ceil_div(absRem, currShape) * sign(remSt)
    tail

  template consumeElse(clamped, nextSh, remSh, remSh2, nextSt,
                       absRem, currShape, remSt, scaled, tail) =
    let clamped = min(nextSh, remSh)
    divisibilityCheck(remSh, clamped)
    let remSh2 = remSh div clamped
    let nextSt = ceil_div(absRem, currShape) * sign(remSt)
    tail

  template consumeLastShared(remSh, accShN, accStN, fullSh, fullSt) =
    when remSh is Int and typeof(remSh) is Int[1]:
      make_layout(unwrap(accShN), unwrap(accStN))
    else:
      make_layout(unwrap(fullSh), unwrap(fullSt))

  proc emitStep(dimIdx: int; remSh, remSt: NimNode;
                accSh, accSt: seq[NimNode]): NimNode =
    ## Emit the fold step for LHS dimension `dimIdx`, nesting the next
    ## dimension's step, consuming `remSh`/`remSt` as it goes.
    if dimIdx >= R - 1:
      let scaled = nnkInfix.newTree(ident"*", remSt, lhsStLeaves[dimIdx])
      if accSh.len == 0:
        return bindSym"make_layout".newCall(
          bindSym"unwrap".newCall(nnkTupleConstr.newTree(remSh)),
          bindSym"unwrap".newCall(nnkTupleConstr.newTree(scaled)))
      let accShN = nnkTupleConstr.newTree(accSh)
      let accStN = nnkTupleConstr.newTree(accSt)
      let fullSh = nnkTupleConstr.newTree(accSh & @[remSh])
      let fullSt = nnkTupleConstr.newTree(accSt & @[scaled])
      return getAst(consumeLastShared(remSh, accShN, accStN, fullSh, fullSt))
    else:
      let currShape  = genSym(nskLet, "currShape")
      let currStride = genSym(nskLet, "currStride")
      let absRem     = genSym(nskLet, "absRem")
      let nextSh     = genSym(nskLet, "nextShape")
      let nextSt     = genSym(nskLet, "nextStride")
      let clamped    = genSym(nskLet, "clampedShape")
      let remSh2     = genSym(nskLet, "remainingShape")
      let scaled     = nnkInfix.newTree(ident"*", remSt, currStride)
      let skipTail = emitStep(dimIdx + 1, remSh, nextSt, accSh, accSt)
      let skipBody = getAst(consumeSkip(nextSt, absRem, currShape, remSt, skipTail))
      let elseTail = emitStep(dimIdx + 1, remSh2, nextSt,
                              accSh & @[clamped], accSt & @[scaled])
      let elseBody = getAst(consumeElse(clamped, nextSh, remSh, remSh2, nextSt,
                                        absRem, currShape, remSt, scaled, elseTail))
      return getAst(consumeStep(lhsShLeaves[dimIdx], lhsStLeaves[dimIdx],
                                remSh, remSt, currShape, currStride, absRem,
                                nextSh, nextSt, clamped, remSh2, scaled,
                                skipBody, elseBody))

  template strideZeroEntry(remSh, remSt, fold) =
    when remSt is Int and typeof(remSt) is Int[0]:
      # Static stride-0 RHS dimension, every coordinate maps to offset 0,
      # the composed dimension is the RHS dimension itself
      # (CuTe composition_impl is_constant<0, RStride> shortcut).
      make_layout(remSh, remSt)
    else:
      fold

  let remSh0 = genSym(nskLet, "remainingShape")
  let remSt0 = genSym(nskLet, "remainingStride")
  let firstStep = emitStep(0, remSh0, remSt0, @[], @[])
  result = nnkStmtListExpr.newTree(
    nnkLetSection.newTree(
      nnkIdentDefs.newTree(remSh0, newEmptyNode(), remainingShape),
      nnkIdentDefs.newTree(remSt0, newEmptyNode(), remainingStride)),
    getAst(strideZeroEntry(remSh0, remSt0, firstStep)))


func composeDistribute(lhsShapes, lhsStrides: tuple; rhsShapes, rhsStrides: tuple): auto =
  ## Layer RHS dimensions one by one over the FULL coalesced LHS via mapDimensionsWith.
  ## Nested RHS dimensions are handled by recursive composeDistribute calls;
  ## scalar dimensions go directly to composeImpl.
  mapDimensionsWith(make_layout(rhsShapes, rhsStrides)):
    when it.shape is tuple:
      composeDistribute(lhsShapes, lhsStrides, it.shape, it.stride)
    else:
      composeImpl(it.shape, it.stride, lhsShapes, lhsStrides)


func compose*[A, B: Layout](a: A, b: B): auto =
  ## Layout composition.
  ##
  ## Returns a layout `R` such that `R(i) = A(B(i))` for all
  ## `i` in `0 ..< cosize(B)`.
  ##
  ## Divisibility of the consumed shape is a caller precondition:
  ## static leaves assert at compile time, runtime shapes are unchecked
  ## (device code carries no doAssert).
  when a.shape isnot tuple:
    when b.stride is tuple:
      when countLeaves(b.shape) != rank(b.shape):
        composeDistribute((a.shape,), (a.stride,), b.shape, b.stride)
      else:
        make_layout(b.shape, flatMapLeaves(b.stride, it * a.stride))
    else:
      make_layout(flatMapLeaves(b.shape, it), b.stride * a.stride)
  elif b.shape isnot tuple:
    # CuTe: coalesce LHS first (preserving trailing stride-0 dimensions), then compose with scalar RHS
    # Uses coalesce_preserve_trailing to match CuTe's coalesce_x in composition.
    let flatA = coalesce_preserve_trailing(a)
    when flatA.shape isnot tuple:
      # flatA is rank-1 scalar: RHS shape is result, strides = b.stride * flatA.stride
      make_layout(b.shape, b.stride.scaleBy(flatA.stride))
    else:
      composeImpl(b.shape, b.stride, flatA.shape, flatA.stride)
  else:
    # CuTe: coalesce LHS first (preserving trailing stride-0 dimensions), then compose with tuple RHS
    let flatA = coalesce_preserve_trailing(a)
    when flatA.shape isnot tuple:
      # flatA is rank-1 scalar: preserve B's nesting, scale strides by flatA.stride
      make_layout(b.shape, b.stride.scaleBy(flatA.stride))
    else:
      composeDistribute(flatA.shape, flatA.stride, b.shape, b.stride)

# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
#  logical_divide — tile a layout into (tile, rest)
# ═══════════════════════════════════════════════════════════════
#
#  CuTe formula:  logical_divide(A, B) = compose(A, Layout(B, complement(B, shape(coalesce(A)))))
#  Tuple tiler → per-dimension divide
#  int / Int:      make_layout(tiler) then CuTe formula
#

# ═══════════════════════════════════════════════════════════════

func logical_divide_impl[A, B: Layout](layout: A; tiler: B): auto =
  ## Core CuTe formula: complement + concat + compose.
  let comp = complement(tiler, size(coalesce(layout)))
  let combined = make_layout((tiler.shape, comp.shape), (tiler.stride, comp.stride))
  compose(layout, combined)

func logical_divide*[L, T: Layout](layout: L; tiler: T): auto =
  ## Layout tiler → CuTe formula directly.
  logical_divide_impl(layout, tiler)

func logical_divide*[L: Layout](layout: L; tiler: int): auto {.inline.} =
  ## Dynamic int tiler → wrap in Layout → CuTe formula.
  when layout.shape isnot tuple:
    make_layout((tiler, ceil_div(layout.shape, tiler)),
                (layout.stride, layout.stride * tiler))
  else:
    logical_divide_impl(layout, make_layout(tiler))

func logical_divide*[L: Layout; V: static int](layout: L; tiler: Int[V]): auto {.inline.} =
  ## Static int tiler (Int[N]) → wrap in Layout → CuTe formula.
  when layout.shape isnot tuple:
    make_layout((tiler, ceil_div(layout.shape, tiler)),
                (layout.stride, layout.stride * tiler))
  else:
    logical_divide_impl(layout, make_layout(tiler))

func logical_divide*[L: Layout](layout: L; tiler: static int): auto {.inline.} =
  ## Compile-time int tiler (const) → preserve via Int[N] wrap → CuTe formula.
  logical_divide_impl(layout, make_layout(Int[tiler]()))

macro logical_divide*(layout: Layout; tiler: tuple): untyped =
  ## Tuple tiler → per-dimension divide.
  ## Each tiler element applies to the corresponding layout dimension.
  ##
  ## - Scalar dimensions with static scalar tilers emit the closed form inline
  ##   per dimension, (T):(d) and ((s + T - 1) div T):(d*T), zero runtime funcs
  ## - Other dimensions delegate to logical_divide per dimension
  ## - Dimensions beyond len(tiler) pass through unchanged
  let lyt = genSym(nskLet, "lyt")
  let tlr = genSym(nskLet, "tlr")
  let (shTy, stTy) = layoutTypeArgs(layout)
  let shapeType = shTy
  let R = if shapeType.kind in {nnkTupleConstr, nnkTupleTy}:
            shapeType.len
          else:
            1
  let tilerRank = tiler.getTypeInst().len
  doAssert tilerRank <= R,
    "logical_divide: tiler has more dimensions (" & $tilerRank &
    ") than layout (" & $R & ")"


  template dimDivided(d, lyt, tlr, idx) =
    let d = logical_divide(dimension(lyt, idx), tlr[idx])

  proc leafIsScalar(t: NimNode): bool =
    ## int or Int[N] leaf type
    (t.kind == nnkSym and $t == "int") or
    (t.kind == nnkBracketExpr and $t[0] == "Int")

  proc closedFormRest(sh, st, tiler: NimNode; shTyLeaf: NimNode;
                      tilerV: int): tuple[sh, st: NimNode] =
    ## Rest (shape, stride) of (s):(d) divided by static scalar T:
    ## shape (s + T - 1) div T, stride d * T.
    ## Int-typed shapes fold to Int literals, genBinOp folds static strides.
    let shV =
      if shTyLeaf.kind == nnkBracketExpr and $shTyLeaf[0] == "Int":
        int(shTyLeaf[1].intVal)
      else:
        -1
    if shV >= 0:
      result.sh = IntCT((shV + tilerV - 1) div tilerV)
    else:
      result.sh = nnkInfix.newTree(ident"div",
        nnkPar.newTree(nnkInfix.newTree(ident"+", sh, newLit(tilerV - 1))),
        newLit(tilerV))
    result.st = nnkInfix.newTree(ident"*", st, tiler)

  let tilerTy = tiler.getTypeInst()
  var accSh, accSt: seq[NimNode]
  var stmts: seq[NimNode]
  for idx in 0 ..< R:
    if idx < tilerRank:
      let d = genSym(nskLet, "d")
      # tiler element, the literal child for tuple constructors,
      # a static bracket access otherwise
      let elemNode =
        if tiler.kind in {nnkTupleConstr, nnkPar}:
          tiler[idx]
        else:
          nnkBracketExpr.newTree(tiler, newLit(idx))
      # static tiler value from the element type node or literal value
      let elemTy = tilerTy[idx]
      let elemV =
        if elemTy.kind == nnkBracketExpr and $elemTy[0] == "Int":
          int(elemTy[1].intVal)
        elif elemTy.kind in {nnkIntLit, nnkInt64Lit}:
          int(elemTy.intVal)
        elif elemNode.kind in {nnkIntLit, nnkInt64Lit}:
          int(elemNode.intVal)
        else:
          -1
      if elemV >= 1 and leafIsScalar(shTy[idx]) and leafIsScalar(stTy[idx]):
        # Closed-form emission, the tiler leaf is wrapped in Int[V] like
        # the scalar tiler overloads do
        let tilerLeaf =
          if elemNode.kind in {nnkIntLit, nnkInt64Lit}:
            newNimNode(nnkObjConstr).add(
              newNimNode(nnkBracketExpr).add(ident"Int", elemNode))
          else:
            elemNode
        let rf = closedFormRest(
          nnkBracketExpr.newTree(nnkDotExpr.newTree(lyt, ident"shape"), newLit(idx)),
          nnkBracketExpr.newTree(nnkDotExpr.newTree(lyt, ident"stride"), newLit(idx)),
          tilerLeaf, shTy[idx], elemV)
        stmts.add nnkLetSection.newTree(
          nnkIdentDefs.newTree(d, newEmptyNode(),
            newCall(bindSym"make_layout",
              nnkTupleConstr.newTree(tilerLeaf, rf.sh),
              nnkTupleConstr.newTree(
                nnkBracketExpr.newTree(nnkDotExpr.newTree(lyt, ident"stride"), newLit(idx)),
                rf.st))))
      else:
        stmts.add getAst(dimDivided(d, lyt, tlr, newLit(idx)))
      accSh.add d.newDotExpr(ident"shape")
      accSt.add d.newDotExpr(ident"stride")
    else:
      let m = genSym(nskLet, "m")
      stmts.add nnkLetSection.newTree(
        nnkIdentDefs.newTree(m, newEmptyNode(), bindSym"dimension".newCall(lyt, newLit(idx))))
      accSh.add m.newDotExpr(ident"shape")
      accSt.add m.newDotExpr(ident"stride")

  let shapeTuple = newTree(nnkTupleConstr, accSh)
  let strideTuple = newTree(nnkTupleConstr, accSt)
  result = nnkStmtListExpr.newTree(
    nnkLetSection.newTree(
      nnkIdentDefs.newTree(lyt, newEmptyNode(), layout),
      nnkIdentDefs.newTree(tlr, newEmptyNode(), tiler)))
  for st in stmts:
    result.add st
  result.add bindSym"make_layout".newCall(shapeTuple, strideTuple)

# ═══════════════════════════════════════════════════════════════
#  tile_unzip — unzip a logical_divide/product result into tiles+rest
# ═══════════════════════════════════════════════════════════════

template tile_unzip*[L: Layout, T](layout: L; tiler: T): auto =
  ## Unzip a logical_divide/logical_product result according to a tiler.
  ## Returns a rank-2 Layout: ((tile_modes), (rest_modes)).
  block:
    evalOnceAs(lyt, layout)
    evalOnceAs(tlr, tiler)
    when tiler is Layout:
      make_layout(
        zip2_by(lyt.shape, tlr.shape),
        zip2_by(lyt.stride, tlr.shape))
    else:
      make_layout(
        zip2_by(lyt.shape, tlr),
        zip2_by(lyt.stride, tlr))

# ═══════════════════════════════════════════════════════════════
func zipped_divide*[L: Layout, T](layout: L, tiler: T): auto {.inline.} =
  ## Divide layout by tiler and zip tile/rest dimensions into rank-2 result.
  ##
  ## CuTe: zipped_divide =
  ##   - Layout tiler: logical_divide(layout, tiler)
  ##   - tuple/int tiler: tile_unzip(logical_divide(layout, tiler), tiler)
  when T is Layout:
    logical_divide(layout, tiler)
  elif T is int or T is Int:
    logical_divide(layout, tiler)
  else:
    tile_unzip(logical_divide(layout, tiler), tiler)

template tiled_divide*(layout: Layout; tiler: auto): auto =
  ## Like zipped_divide but unpack the second dimension into individual dimensions.
  ## Keeps dimension-0 grouped (the tile).
  block:
    evalOnceAs(lyt, layout)
    evalOnceAs(tlr, tiler)
    evalOnceAs(zd, zipped_divide(lyt, tlr))
    make_layout(
      concatFlat(
        (dimension(zd, 0).shape,),
        dimension(zd, 1).shape
      ),
      concatFlat(
        (dimension(zd, 0).stride,),
        dimension(zd, 1).stride
      )
    )

template flat_divide*(layout: Layout; tiler: auto): auto =
  ## Like zipped_divide but unpack BOTH dimensions into a flat layout.
  ## Difference from tiled_divide: tile dimensions are also unpacked.
  block:
    evalOnceAs(lyt, layout)
    evalOnceAs(tlr, tiler)
    evalOnceAs(zd, zipped_divide(lyt, tlr))
    make_layout(
      concatFlat(
        dimension(zd, 0).shape,
        dimension(zd, 1).shape,
      ),
      concatFlat(
        dimension(zd, 0).stride,
        dimension(zd, 1).stride,
      ),
    )

# ═══════════════════════════════════════════════════════════════
#  right_inverse — quasi-inverse sorted by stride
# ═══════════════════════════════════════════════════════════════

proc rightInverseChain(
    strides, shapes, prefixProd: seq[int]; shNode: NimNode): LayoutCT {.compileTime.} =
  ## Return right-inverse dimensions as LayoutCT (empty if no chain found).
  result = LayoutCT()
  var curr = 1
  for idx in getIndicesSortedByStride(strides):
    if strides[idx] == curr:
      result.append(
        newTree(nnkBracketExpr, shNode, newLit(idx)),
        IntCT(prefixProd[idx]))
      if shapes[idx] != DynamicSentinel:
        curr = strides[idx] * shapes[idx]
      else:
        break

macro rightInverseImpl(sh, st: typed): untyped =
  ## right_inverse on flattened (shape, stride).
  let stTyp = st.getTypeInst()
  let shTyp = sh.getTypeInst()

  # Scalar: no sorting needed
  if shTyp.kind != nnkTupleConstr:
    let stNode = stTyp
    if stNode.kind == nnkBracketExpr and $stNode[0] == "Int" and stNode[1].intVal == 1:
      result = newCall(bindSym"make_layout", sh, st)
    else:
      result = newCall(bindSym"make_layout", IntCT(1), newLit(0))
    return

  # Multi-dimension: extract values, fill LayoutCT via helper
  let strides = toSeqStaticInts(stTyp)
  let shapes  = toSeqStaticInts(shTyp)
  let prefixProd = prefixProduct(shapes)
  let acc = rightInverseChain(strides, shapes, prefixProd, sh)
  if acc.shape.len == 0:
    result = newCall(bindSym"make_layout", IntCT(1), newLit(0))
  else:
    result = newCall(bindSym"coalesce", acc.emit())

func right_inverse*(layout: Layout): auto =
  ## Quasi-inverse: L(R(i)) == i for all i < size(R).
  ## Sorts dimensions by stride, finds max contiguous chain.
  let c = coalesce(layout)
  rightInverseImpl(flatten(c.shape), flatten(c.stride))

# ═══════════════════════════════════════════════════════════════
#  left_inverse — left inverse (injective layouts only)
# ═══════════════════════════════════════════════════════════════

proc leftInverseDimensions*(
    strides, shapes, prefixProd: seq[int]; shNode: NimNode): LayoutCT {.compileTime.} =
  ## Return left-inverse dimensions as a LayoutCT.
  ## Builds from stride ratios:
  ##   result_shape[i] = stride / size_so_far
  ##   result_prefix[i] = prefixProd[prev_idx]
  result = LayoutCT()
  var sizeSoFar = 1
  var prevIdx = -1
  var prevPrefix = 0
  for idx in getIndicesSortedByStride(strides):
    if strides[idx] == 0:
      continue
    doAssert strides[idx] mod sizeSoFar == 0,
      "left_inverse: stride " & $strides[idx] & " not divisible by " & $sizeSoFar
    if prevIdx == -1:
      # First dimension: computed shape, zero stride
      result.append(IntCT(strides[idx] div sizeSoFar), IntCT(0))
    else:
      # Intermediate dimension: computed shape, previous prefix as stride
      result.append(IntCT(strides[idx] div sizeSoFar), IntCT(prevPrefix))
    sizeSoFar = strides[idx]
    prevIdx = idx
    prevPrefix = prefixProd[idx]
  # Last dimension from original layout
  result.append(newTree(nnkBracketExpr, shNode, newLit(prevIdx)), IntCT(prevPrefix))

macro leftInverseImpl(sh, st: typed): untyped =
  ## left_inverse on flattened (shape, stride). All strides must be static.
  let stTyp = st.getTypeInst()
  let shTyp = sh.getTypeInst()

  if shTyp.kind != nnkTupleConstr:
    let stNode = stTyp
    if stNode.kind == nnkBracketExpr and $stNode[0] == "Int" and stNode[1].intVal == 1:
      result = newCall(bindSym"make_layout", sh, st)
    elif stNode.kind == nnkBracketExpr and $stNode[0] == "Int" and stNode[1].intVal == 0:
      result = newCall(bindSym"make_layout", IntCT(1), newLit(0))
    else:
      # Non-unit, non-zero stride: build left_inverse from stride ratios
      let strideVal = stNode[1].intVal
      var acc = LayoutCT()
      acc.append(IntCT(strideVal), IntCT(0))
      acc.append(sh, IntCT(1))
      result = newCall(bindSym"coalesce", acc.emit())
    return

  let strides = toSeqStaticInts(stTyp)
  let shapes  = toSeqStaticInts(shTyp)
  let prefixProd = prefixProduct(shapes)
  let acc = leftInverseDimensions(strides, shapes, prefixProd, sh)
  if acc.shape.len == 0:
    result = newCall(bindSym"make_layout", IntCT(1), newLit(0))
  else:
    result = newCall(bindSym"coalesce", acc.emit())

func left_inverse*(layout: Layout): auto =
  ## Left inverse: Li(L(i)) == i for injective layouts.
  ## Requires all-static strides. Builds from stride ratios.
  let c = coalesce(layout)
  leftInverseImpl(flatten(c.shape), flatten(c.stride))


# ═══════════════════════════════════════════════════════════════
#  logical_product — reproduce a block over a tiler
# ═══════════════════════════════════════════════════════════════

func logical_product*[A, B: Layout](a: A; tiler: B): auto =
  ## Reproduce block over tiler: rank-2 result ((BLOCK), (TILE)).
  ## Inverse of logical_divide.
  let rest = compose(complement(a, size(a) * cosize(tiler)), tiler)
  make_layout((a.shape, rest.shape), (a.stride, rest.stride))


func nested_product*[A, B: Layout](a: A; b: B): auto =
  ## Categorical product of two layouts, preserving each argument's dimension grouping.
  ##
  ## Given:
  ##   A: (a0, a1, ...):(sa0, sa1, ...)
  ##   B: (b0, b1, ...):(sb0, sb1, ...)
  ## Returns:
  ##   ((a0, a1, ...), (b0, b1, ...)) : ((sa0, sa1, ...), (sb0, sb1, ...))
  make_layout((a.shape, b.shape), (a.stride, b.stride))


# ── zipped_product / tiled_product / flat_product ──

template zipped_product*(blk: Layout; tiler: auto): auto =
  ## Reproduce block over tiler, zipped into rank-2 result.
  ##
  ## CuTe: zipped_product = tile_unzip(logical_product(block, tiler), tiler)
  block:
    evalOnceAs(bk, blk)
    evalOnceAs(tlr, tiler)
    when tiler is Layout:
      logical_product(bk, tlr)
    else:
      tile_unzip(logical_product(bk, tlr), tlr)

template tiled_product*(blk: Layout; tiler: auto): auto =
  ## Like zipped_product but unpack the second dimension.
  ## Keeps dimension-0 grouped (the block).
  block:
    evalOnceAs(bk, blk)
    evalOnceAs(tlr, tiler)
    evalOnceAs(zp, zipped_product(bk, tlr))
    make_layout(
      concatFlat(
        (dimension(zp, 0).shape,),
        dimension(zp, 1).shape,
      ),
      concatFlat(
        (dimension(zp, 0).stride,),
        dimension(zp, 1).stride,
      ),
    )

template flat_product*(blk: Layout; tiler: auto): auto =
  ## Like zipped_product but unpack BOTH dimensions into a flat layout.
  ## Difference from tiled_product: block dimensions are also unpacked.
  block:
    evalOnceAs(bk, blk)
    evalOnceAs(tlr, tiler)
    evalOnceAs(zp, zipped_product(bk, tlr))
    make_layout(
      concatFlat(
        dimension(zp, 0).shape,
        dimension(zp, 1).shape,
      ),
      concatFlat(
        dimension(zp, 0).stride,
        dimension(zp, 1).stride,
      ),
    )

# ═══════════════════════════════════════════════════════════════
#  blocked_product — blocks laid out contiguously
# ═══════════════════════════════════════════════════════════════
#
#  blocked_product(block, tiler):
#    1. Append both to rank R = max(rank(block), rank(tiler))
#    2. result = logical_product(padded_block, padded_layout)
#    3. return zipDimensions(result[0], result[1])

func blocked_product*[A, B: Layout](blk: A; tiler: B): auto =
  ## Repeat block over tiler grid, each block contiguous.
  ## Results in ((BLK_A, TILER_A), (BLK_B, TILER_B), ...).
  const mxR = max(rank(type(blk)), rank(type(tiler)))
  let lp = logical_product(padRight(blk, mxR), padRight(tiler, mxR))
  let m0 = dimension(lp, 0)
  let m1 = dimension(lp, 1)
  zipDimensions(m0, m1)

# ═══════════════════════════════════════════════════════════════
#
#  raked_product(block, tiler):
#    1. Same logical_product as blocked_product
#    2. return zipDimensions(result[1], result[0])  (swapped order)

func raked_product*[A, B: Layout](blk: A; tiler: B): auto =
  ## Repeat block over tiler grid, blocks interleaved.
  ## Results in ((TILER_A, BLK_A), (TILER_B, BLK_B), ...).
  const mxR = max(rank(type(blk)), rank(type(tiler)))
  let lp = logical_product(padRight(blk, mxR), padRight(tiler, mxR))
  let m0 = dimension(lp, 0)
  let m1 = dimension(lp, 1)
  zipDimensions(m1, m0)

# ═══════════════════════════════════════════════════════════════
#  tile_to_shape — repeat block layout to fill target shape
# ═══════════════════════════════════════════════════════════════

template tile_to_shape*(blk: Layout; target_shape: typed; ord_shape: static StrideOrder = LayoutLeft): auto =
  ## Recipe:
  ##   1. Pad block to rank R
  ##   2. Compute block_shape = product_each(block.shape)   — per-dimension products
  ##   3. Compute target_shape_flat = product_each(target_shape)    — per-dimension products
  ##   4. product_shape = ceil_div(target_shape, block_shape) — repeats per dimension
  ##   5. tiler = make_layout(product_shape, ord_shape)
  ##   6. result = blocked_product(padded_block, tiler)
  ##
  ## Example:
  ##   let tile = tile_to_shape(make_layout((2,3), (1,2)), (6, 12))
  ##   # block (2,3) repeated to fill (6,12) in 3 columns:
  ##   # ((2,3),3):((1,2),6)
  const R = static(rank(target_shape))
  block:
    evalOnceAs(bk, blk)
    evalOnceAs(ts, target_shape)
    let padded_blk = padRight(bk, R)
    let blk_shape = product_each(padded_blk.shape)
    let trg_flat = product_each(ts)
    let product_shape = zipDimensionsWith(trg_flat, blk_shape): ceil_div(it_a, it_b)
    let tiler = make_layout(product_shape, ord_shape)
    blocked_product(padded_blk, tiler)

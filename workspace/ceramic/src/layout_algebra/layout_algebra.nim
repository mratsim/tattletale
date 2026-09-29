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

  # Scalar guard for a single leaf (plain scalar or 1-element tuple)
  if shLeaves.len == 1 and stLeaves.len == 1:
    # Check if scalar shape is size-1 (inactive dimension)
    if isStaticOne(shTypes[0]):
      result = newCall(bindSym"make_layout", newLit(1), newLit(0))
    else:
      result = newCall(bindSym"make_layout", shLeaves[0], stLeaves[0])
    return

  var resShapes: seq[NimNode] = @[]
  var resStrides: seq[NimNode] = @[]
  var resSTypes: seq[NimNode] = @[]
  var resSTypes2: seq[NimNode] = @[]

  # Seed with the last dimension, the leaf streams are in reverse order
  resShapes.add shLeaves[0]
  resStrides.add stLeaves[0]
  resSTypes.add shTypes[0]
  resSTypes2.add stTypes[0]

  if preserveTrailing:
    # When preserving trailing size-1 dimensions, seed with `low(int)` (non-1 sentinel)
    # to prevent the post-loop discard from removing the last dimension.
    # Mirrors CuTe's coalesce_x which seeds bw_coalesce with Int<2>{} sentinel.
    if isStaticOne(shTypes[0]):
      resShapes[0] = IntCT(low(int))
      resSTypes[0] = newNimNode(nnkBracketExpr).add(ident"Int", newLit(low(int)))

  for k in 1 ..< shLeaves.len:
    let curST = shTypes[k]
    let curST2 = stTypes[k]

    if isStaticOne(curST):
      continue

    # CuTe branch 3: when seed (resSTypes[0]) is size-1, replace seed with current
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

  # Post-loop: discard trailing size-1 dimensions (the seed might be size-1)
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
  ## Mirrors CuTe's `coalesce_x`: seeds backward coalescence with `low(int)`
  ## sentinel instead of `Int[1]`, preventing the post-loop discard from
  ## removing the last dimension.
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

  # Broadcast (stride-0): complement is a single dimension with shape=bound, stride=1
  if stTyp.kind == nnkBracketExpr and $stTyp[0] == "Int" and stTyp[1].intVal == 0:
    return newCall(bindSym"make_layout", boundExpr, newLit(1))

  # General case
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
  ## (stride div cur, cur) and advances cur = stride * shape. Runtime
  ## shapes advance cur with a runtime expression — the fold continues
  ## past them. Statically-1 dimensions are skipped; runtime dimensions are
  ## appended unconditionally.
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

func unwrap(t: tuple): auto {.inline.} =
  ## CuTe's `unwrap`: collapse a single-element tuple to its scalar so a
  ## composed single-dimension result stays scalar (CuTe's composition_impl
  ## does `Layout{unwrap(result_shape), unwrap(result_stride)}`);
  ## multi-element tuples pass through unchanged. Without this, every
  ## composed leaf comes out as a rank-1 1-tuple `(4,)` and dimension
  ## collection nests them as `((4,), (8,))` instead of CuTe's flat
  ## `(4, 8)`.
  when rank(t) == 1:
    t[0]
  else:
    t

macro buildStride*(t: tuple; s: typed): untyped =
  ## Broadcast helper, multiply each element of tuple t by scalar s.
  ##
  ## - Emits a single flat tuple construction (flatMapLeaves)
  ## - Intermediate tuple types and per-element concat chains do not appear
  ## Returns the multiplied tuple as an untyped node.
  result = newCall(bindSym"flatMapLeaves", t, newTree(nnkInfix, ident"*", ident"it", s))

func buildStride[T, S: int or Int](t: T; s: S, idx: static int = 0): auto {.inline.} =
  static: doAssert idx == 0
  ((t * s))

template divisibilityCheck(remainingShape, clampedShape: untyped) =
  ## Python tensor-layouts compatible divisibility check.
  when clampedShape is Int:
    when typeof(clampedShape).V == 1:
      discard  # shape 1 is trivially divisor
    elif remainingShape is Int: # Compile time assert
      static: doAssert typeof(remainingShape).V mod typeof(clampedShape).V == 0,
        "compose: shape " & $typeof(remainingShape).V & " and consumed shape " & $typeof(clampedShape).V & " are not divisible"
    else:
      doAssert remainingShape mod clampedShape == 0,
        "compose: shape " & $remainingShape & " and consumed shape " & $clampedShape & " are not divisible"
  else:
    doAssert remainingShape mod clampedShape == 0,
      "compose: shape " & $remainingShape & " and consumed shape " & $clampedShape & " are not divisible"

macro composeImpl*(remainingShape, remainingStride: untyped; lhsShapes, lhsStrides: typed): untyped =
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

  proc pack(nodes: seq[NimNode]): NimNode =
    nnkTupleConstr.newTree(nodes)

  proc whenInt1(x: NimNode): NimNode =
    ## The consume check of the original fold, `x is Int and typeof(x) is
    ## Int[1]`, on an emitted expression.
    let isInt = newTree(nnkInfix, ident"is", x, ident"Int")
    let isInt1 = newTree(nnkInfix, ident"is",
      newCall(ident"typeof", x),
      nnkBracketExpr.newTree(ident"Int", newLit(1)))
    newTree(nnkInfix, ident"and", isInt, isInt1)

  proc letBind(sym: NimNode; expr: NimNode): NimNode =
    newTree(nnkLetSection, newTree(nnkIdentDefs, sym, newEmptyNode(), expr))

  proc emitStep(dimIdx: int; remSh, remSt: NimNode;
                lhsShLeaves, lhsStLeaves: seq[NimNode];
                accSh, accSt: seq[NimNode]): NimNode =
    if dimIdx >= R - 1:
      # Emits:
      #   when remSh is Int[1] and typeof(remSh) is Int[1]:
      #     make_layout(unwrap((accSh...)), unwrap((accSt...)))  # accumulated only
      #   else:
      #     make_layout(unwrap((accSh..., remSh)), unwrap((accSt..., remSt * lhsSt[R-1])))
      let lastSt = lhsStLeaves[dimIdx]
      let scaled = newTree(nnkInfix, ident"*", remSt, lastSt)
      if accSh.len == 0:
        return newCall(bindSym("make_layout"),
          newCall(bindSym("unwrap"), pack(@[remSh])),
          newCall(bindSym("unwrap"), pack(@[scaled])))
      let yes = nnkStmtListExpr.newTree(
        newCall(bindSym("make_layout"),
          newCall(bindSym("unwrap"), pack(accSh)),
          newCall(bindSym("unwrap"), pack(accSt))))
      let no = nnkStmtListExpr.newTree(
        newCall(bindSym("make_layout"),
          newCall(bindSym("unwrap"), pack(accSh & @[remSh])),
          newCall(bindSym("unwrap"), pack(accSt & @[scaled]))))
      return nnkWhenStmt.newTree(
        nnkElifBranch.newTree(whenInt1(remSh), yes),
        nnkElse.newTree(no))
    else:
      # Emits:
      #   let currShape = lhsSh[d]
      #   let currStride = lhsSt[d]
      #   let absRem = abs(remSt)
      #   let nextShape = ceil_div(currShape, absRem)
      #   when (nextShape is Int and typeof(nextShape) is Int[1])
      #     or (remSh is Int and typeof(remSh) is Int[1]):
      #     let nextStride = sign(remSt) * ceil_div(absRem, currShape)
      #     <emitStep(d+1, remSh, nextStride)>
      #   else:
      #     let clampedShape = min(nextShape, remSh)
      #     divisibilityCheck(remSh, clampedShape)  # compile-time doAssert
      #     let remainingShape = remSh div clampedShape
      #     let nextStride = sign(remSt) * ceil_div(absRem, currShape)
      #     <emitStep(d+1, remainingShape, nextStride)>
      let currShape  = genSym(nskLet, "currShape")
      let currStride = genSym(nskLet, "currStride")
      let absRem     = genSym(nskLet, "absRem")
      let nextSh     = genSym(nskLet, "nextShape")
      let nextSt     = genSym(nskLet, "nextStride")
      let clamped    = genSym(nskLet, "clampedShape")
      let remSh2     = genSym(nskLet, "remainingShape")
      let scaled     = newTree(nnkInfix, ident"*", remSt, currStride)

      let skipBody = nnkStmtListExpr.newTree(
        letBind(nextSt,
          newTree(nnkInfix, ident"*",
            newCall(bindSym("ceil_div"), absRem, currShape),
            newCall(bindSym("sign"), remSt))),
        emitStep(dimIdx + 1, remSh, nextSt, lhsShLeaves, lhsStLeaves, accSh, accSt))

      let elseBody = nnkStmtListExpr.newTree(
        letBind(clamped, newCall(bindSym("min"), nextSh, remSh)),
        newCall(bindSym("divisibilityCheck"), remSh, clamped),
        letBind(remSh2, newTree(nnkInfix, ident"div", remSh, clamped)),
        letBind(nextSt,
          newTree(nnkInfix, ident"*",
            newCall(bindSym("ceil_div"), absRem, currShape),
            newCall(bindSym("sign"), remSt))),
        emitStep(dimIdx + 1, remSh2, nextSt, lhsShLeaves, lhsStLeaves,
                 accSh & @[clamped], accSt & @[scaled]))

      return nnkStmtListExpr.newTree(
        letBind(currShape, lhsShLeaves[dimIdx]),
        letBind(currStride, lhsStLeaves[dimIdx]),
        letBind(absRem, newCall(bindSym("abs"), remSt)),
        letBind(nextSh, newCall(bindSym("ceil_div"), currShape, absRem)),
        nnkWhenStmt.newTree(
          nnkElifBranch.newTree(
            newTree(nnkInfix, ident"or", whenInt1(nextSh), whenInt1(remSh)),
            skipBody),
          nnkElse.newTree(elseBody)))

  let remSh0 = genSym(nskLet, "remainingShape")
  let remSt0 = genSym(nskLet, "remainingStride")
  result = nnkStmtListExpr.newTree(
    letBind(remSh0, remainingShape),
    letBind(remSt0, remainingStride))
  result.add emitStep(0, remSh0, remSt0, lhsShLeaves, lhsStLeaves, @[], @[])

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
  when a.shape isnot tuple:
    when b.stride is tuple:
      when countLeaves(b.shape) != rank(b.shape):
        composeDistribute((a.shape,), (a.stride,), b.shape, b.stride)
      else:
        make_layout(b.shape, buildStride(b.stride, a.stride))
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
#  Tuple tiler:    transform_layout(logical_divide, layout, tiler)  — per-dimension divide
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
  logical_divide_impl(layout, make_layout(tiler))

func logical_divide*[L: Layout; V: static int](layout: L; tiler: Int[V]): auto {.inline.} =
  ## Static int tiler (Int[N]) → wrap in Layout → CuTe formula.
  logical_divide_impl(layout, make_layout(tiler))

func logical_divide*[L: Layout](layout: L; tiler: static int): auto {.inline.} =
  ## Compile-time int tiler (const) → preserve via Int[N] wrap → CuTe formula.
  logical_divide_impl(layout, make_layout(Int[tiler]()))

macro logical_divide_builder*(layout, tiler: typed; LayoutRank: static int): untyped =
  ## Build helper for logical_divide over a tuple tiler, one pass over
  ## the layout dimensions.
  ##
  ## - Each divided dimension packs into the final (shape, stride) tuple
  ##   construction directly, with no per-dimension `concat` accumulator
  ##   and no intermediate tuple types
  ## Returns the divided layout as an untyped node.
  let lyt = genSym(nskLet, "lyt")
  let tlr = genSym(nskLet, "tlr")
  let tilerRank = tiler.getTypeInst().len
  var accSh, accSt: seq[NimNode]
  var stmts: seq[NimNode]

  proc pack(nodes: seq[NimNode]): NimNode =
    nnkTupleConstr.newTree(nodes)

  for idx in 0 ..< LayoutRank:
    let dimExpr = newCall(bindSym"dimension", lyt, newLit(idx))
    if idx < tilerRank:
      let d = genSym(nskLet, "d")
      stmts.add newTree(nnkLetSection, newTree(nnkIdentDefs, d,
        newEmptyNode(),
        newCall(bindSym"logical_divide", dimExpr, newTree(nnkBracketExpr, tlr, newLit(idx)))))
      accSh.add newDotExpr(d, ident"shape")
      accSt.add newDotExpr(d, ident"stride")
    else:
      let m = genSym(nskLet, "m")
      stmts.add newTree(nnkLetSection, newTree(nnkIdentDefs, m,
        newEmptyNode(), dimExpr))
      accSh.add newDotExpr(m, ident"shape")
      accSt.add newDotExpr(m, ident"stride")

  result = nnkStmtListExpr.newTree(
    newTree(nnkLetSection, newTree(nnkIdentDefs, lyt, newEmptyNode(), layout)),
    newTree(nnkLetSection, newTree(nnkIdentDefs, tlr, newEmptyNode(), tiler)))
  for st in stmts: result.add st
  result.add newCall(bindSym"make_layout", pack(accSh), pack(accSt))

func logical_divide*(layout: Layout; tiler: tuple): auto {.inline.} =
  ## Tuple tiler → per-dimension divide (transform_layout).
  ## Each tiler element applies to the corresponding layout dimension.
  ## Dimensions beyond len(tiler) pass through unchanged.
  const R = static(rank(layout))
  static: doAssert rank(tiler) <= R,
    "logical_divide: tiler has more dimensions (" & $rank(tiler) &
    ") than layout (" & $R & ")"
  logical_divide_builder(layout, tiler, R)

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
#  zipped_divide_builder — one-pass build for tuple tiler
# ═══════════════════════════════════════════════════════════════

macro zipped_divide_builder*(layout, tiler: typed; LayoutRank: static int): untyped =
  ## Build helper for zipped_divide over a tuple tiler, one pass over
  ## the layout dimensions building the (tile, rest) groups directly.
  ##
  ## - The final tuple constructions are packed in one shot, no
  ##   per-dimension `concat` accumulators and no intermediate tuple types
  ## - Intermediate tuple types trigger the Nim tuple hash collision
  ##   (nim-lang/Nim issue 25883), so they must not appear
  ## Returns the zipped layout as an untyped node.
  let lyt = genSym(nskLet, "lyt")
  let tlr = genSym(nskLet, "tlr")
  let tilerRank = tiler.getTypeInst().len
  var tileSh, tileSt, restSh, restSt: seq[NimNode]

  proc pack(nodes: seq[NimNode]): NimNode =
    nnkTupleConstr.newTree(nodes)

  var stmts: seq[NimNode]
  for idx in 0 ..< LayoutRank:
    let dimExpr = newCall(bindSym"dimension", lyt, newLit(idx))
    if idx < tilerRank:
      let d = genSym(nskLet, "d")
      stmts.add newTree(nnkLetSection, newTree(nnkIdentDefs, d,
        newEmptyNode(),
        newCall(bindSym"logical_divide", dimExpr, newTree(nnkBracketExpr, tlr, newLit(idx)))))
      for part in [0, 1]:
        let dd = newCall(bindSym"dimension", d, newLit(part))
        if part == 0:
          tileSh.add newDotExpr(dd, ident"shape")
          tileSt.add newDotExpr(dd, ident"stride")
        else:
          restSh.add newDotExpr(dd, ident"shape")
          restSt.add newDotExpr(dd, ident"stride")
    else:
      let m = genSym(nskLet, "m")
      stmts.add newTree(nnkLetSection, newTree(nnkIdentDefs, m,
        newEmptyNode(), dimExpr))
      restSh.add newDotExpr(m, ident"shape")
      restSt.add newDotExpr(m, ident"stride")

  result = nnkStmtListExpr.newTree(
    newTree(nnkLetSection, newTree(nnkIdentDefs, lyt, newEmptyNode(), layout)),
    newTree(nnkLetSection, newTree(nnkIdentDefs, tlr, newEmptyNode(), tiler)))
  for st in stmts: result.add st
  result.add newCall(bindSym"make_layout",
    nnkPar.newTree(pack(tileSh), pack(restSh)),
    nnkPar.newTree(pack(tileSt), pack(restSt)))

func zipped_divide*[LayoutT: Layout, TilerT](layout: LayoutT; tiler: TilerT): auto {.inline.} =
  ## Divide layout by tiler and zip tile/rest dimensions into rank-2 result.
  ##
  ## CuTe: zipped_divide =
  ##   - Layout tiler: logical_divide(layout, tiler)
  ##   - tuple/int tiler: tile_unzip(logical_divide(layout, tiler), tiler)
  when TilerT is Layout:
    logical_divide(layout, tiler)
  elif TilerT is int or TilerT is Int:
    # Scalar tiler
    logical_divide(layout, tiler)
  else:
    # Tuple tiler: one-pass builder avoids intermediate concat types
    # that trigger Nim C++ backend struct hash collision
    # (see https://github.com/nim-lang/Nim/issues/25883#issuecomment-4658908569)
    const R = static(rank(layout))
    const Tr = static(rank(tiler))
    static: doAssert Tr <= R,
      "zipped_divide: tiler has more dimensions (" & $Tr & ") than layout (" & $R & ")"
    zipped_divide_builder(layout, tiler, R)

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

proc rightInverseChain*(
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


template max_common_layout*(a, b: typed): untyped =
  ## Return a Layout for the maximum contiguous elements common to both.
  ## a(R(i)) == i and b(R(i)) == i for all i < size(result).
  block:
    evalOnceAs(va, a)
    evalOnceAs(vb, b)
    let inv_b = right_inverse(vb)
    let common = coalesce(compose(va, inv_b))
    type StrideT = typeof(common.stride)
    when StrideT is tuple:
      type FirstStride = typeof(common.stride[0])
      const s0 = FirstStride.V
      when s0 == 1:
        type FirstShape = typeof(common.shape[0])
        coalesce(compose(inv_b, make_layout(FirstShape.V, 1)))
      else:
        make_layout(1, 0)
    else:
      const s = StrideT.V
      when s == 1:
        type Sh = typeof(common.shape)
        coalesce(compose(inv_b, make_layout(Sh.V, 1)))
      else:
        make_layout(1, 0)

template max_common_vector*(a, b: typed): int =
  ## Return N: for 0 <= i < N, a(R(i)) == i and b(R(i)) == i.
  block:
    evalOnceAs(va, a)
    evalOnceAs(vb, b)
    let common = coalesce(compose(va, right_inverse(vb)))
    type StrideT = typeof(common.stride)
    when StrideT is tuple:
      type FirstStride = typeof(common.stride[0])
      const s0 = FirstStride.V
      when s0 == 1:
        type FirstShape = typeof(common.shape[0])
        FirstShape.V
      else:
        1
    else:
      const s = StrideT.V
      when s == 1:
        type Sh = typeof(common.shape)
        Sh.V
      else:
        1

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

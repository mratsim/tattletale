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
import ./int_tuples
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

#  coalesce — merge contiguous modes where stride matches
# ═══════════════════════════════════════════════════════════════

macro coalesceBackward(csShape, csStride: typed; preserveTrailing: static bool = false): untyped =
  template at(n, i: untyped): untyped = newTree(nnkBracketExpr, n, newLit(i))
  let stype = csShape.getTypeInst()
  let stype2 = csStride.getTypeInst()

  # Scalar guard: single-mode Layout
  if stype.kind != nnkTupleConstr and stype2.kind != nnkTupleConstr:
    # Check if scalar shape is size-1 (inactive mode)
    if (stype.kind == nnkBracketExpr and $stype[0] == "Int" and stype[1].intVal == 1) or
       (stype.kind == nnkIntLit and stype.intVal == 1):
      result = newCall(bindSym"make_layout", newLit(1), newLit(0))
    else:
      result = newCall(bindSym"make_layout", csShape, csStride)
    return

  let N = stype.len

  var resShapes: seq[NimNode] = @[]
  var resStrides: seq[NimNode] = @[]
  var resSTypes: seq[NimNode] = @[]
  var resSTypes2: seq[NimNode] = @[]

  # Seed: add last mode
  resShapes.add csShape.at(N - 1)
  resStrides.add csStride.at(N - 1)
  resSTypes.add stype[N - 1]
  resSTypes2.add stype2[N - 1]

  if preserveTrailing:
    # When preserving trailing size-1 modes, seed with `low(int)` (non-1 sentinel)
    # to prevent the post-loop discard from removing the last mode.
    # Mirrors CuTe's coalesce_x which seeds bw_coalesce with Int<2>{} sentinel.
    let lastST = stype[N - 1]
    if (lastST.kind == nnkBracketExpr and $lastST[0] == "Int" and lastST[1].intVal == 1) or
       (lastST.kind == nnkIntLit and lastST.intVal == 1):
      resShapes[0] = IntCT(low(int))
      resSTypes[0] = newNimNode(nnkBracketExpr).add(ident"Int", newLit(low(int)))

  for i in countdown(N - 2, 0):
    let curST = stype[i]
    let curST2 = stype2[i]

    if isStaticOne(curST):
      continue

    # CuTe branch 3: when seed (resSTypes[0]) is size-1, replace seed with current
    if isStaticOne(resSTypes[0]):
      resShapes[0] = csShape.at(i)
      resStrides[0] = csStride.at(i)
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
        resStrides[0] = csStride.at(i)
        resSTypes[0] = newNimNode(nnkBracketExpr).add(ident"Int", newLit(mergedVal))
        resSTypes2[0] = curST2
        continue

    resShapes.insert(csShape.at(i), 0)
    resStrides.insert(csStride.at(i), 0)
    resSTypes.insert(curST, 0)
    resSTypes2.insert(curST2, 0)

  # Post-loop: discard trailing size-1 modes (the seed might be size-1)
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
  ## Merge contiguous modes. Flatten preserves Int[N] types with getTypeInst.
  coalesceBackward(
    flatten(layout.shape),
    flatten(layout.stride)
  )

func coalesce_preserve_trailing(layout: Layout): auto {.inline, noInit.} =
  ## Like `coalesce` but preserves trailing size-1 modes (e.g. stride-0 broadcasts).
  ## Mirrors CuTe's `coalesce_x`: seeds backward coalescence with `low(int)`
  ## sentinel instead of `Int[1]`, preventing the post-loop discard from
  ## removing the last mode.
  coalesceBackward(
    flatten(layout.shape),
    flatten(layout.stride),
    preserveTrailing = true
  )

# ═══════════════════════════════════════════════════════════════
#  filter_inactive — remove stride-0 and size-1 modes
# ═══════════════════════════════════════════════════════════════

func filter_inactive*(layout: Layout): auto {.inline.} =
  ## Remove stride-0 and size-1 modes
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
#  ##        └─ multi-mode                                         │
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
#  ##   In the multi-mode path, bound may be Int[N] (static) or int (dynamic).
#  ##   For Int[N] bounds, ceil_div is computed at compile time.
#  ##   For int bounds, a runtime ceil_div expression is emitted.
#  ##   Shape-tuple bounds (e.g. (32,4,4)) are converted to size(product).
# ═══════════════════════════════════════════════════════════════

proc complementScalar(sh, st, boundExpr: NimNode): NimNode {.compileTime.} =
  ## Standard CuTe scalar complement formula.
  ##   gap = max(1, st);  prd = st * sh;  rem = ceil_div(bound, prd)
  ##   result = coalesce(Layout((gap, rem), (1, prd)))
  let stTyp = st.getTypeInst()

  # Broadcast (stride-0): complement is a single mode with shape=bound, stride=1
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
  ## Build full complement LayoutCT (gap modes + remainder), folding over
  ## modes in ascending-stride order: each mode contributes a gap mode
  ## (stride div cur, cur) and advances cur = stride * shape. Runtime
  ## shapes advance cur with a runtime expression — the fold continues
  ## past them. Statically-1 modes are skipped; runtime modes are
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
  ## Multi-mode complement: sort by stride, fold to fill gaps.
  ## All strides must be static Int[N] (compile-time check).
  let stTyp = st.getTypeInst()
  let shTyp = sh.getTypeInst()

  doAssert stTyp.kind == nnkTupleConstr,
    "complementMulti: expected tuple type for strides"
  for i in 0 ..< stTyp.len:
    let stNode = stTyp[i]
    doAssert stNode.kind == nnkBracketExpr and $stNode[0] == "Int",
      "complement: multi-mode with dynamic strides not supported at index " & $i

  let strides = toSeqStaticInts(stTyp)
  let shapes  = toSeqStaticInts(shTyp)
  let acc = complementGaps(strides, shapes, sh, boundExpr)
  newCall(bindSym"coalesce", acc.emit())

macro complementImpl(sh, st, cosizeBound: typed): untyped =
  ## Dispatch to scalar or multi-mode complement.
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
  ## Filters inactive modes first (matches CuTe's filter-before-complement).
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
##         make_layout(              fold over modes
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
##                                   │ Last mode    │
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
  ## composed single-mode result stays scalar (CuTe's composition_impl
  ## does `Layout{unwrap(result_shape), unwrap(result_stride)}`);
  ## multi-element tuples pass through unchanged. Without this, every
  ## composed leaf comes out as a rank-1 1-tuple `(4,)` and mode
  ## collection nests them as `((4,), (8,))` instead of CuTe's flat
  ## `(4, 8)`.
  when rank(t) == 1:
    t[0]
  else:
    t

func buildStride(t: tuple; s: int or Int, idx: static int = 0): auto {.inline.} =
  # Broadcast helper: multiply each element of tuple t by scalar s
  # Builds concat(t[0]*s, concat(t[1]*s, ... ())) at compile time
  # via recursive template. No heap, no seq, no macro type introspection.
  when idx == rank(t) - 1:
    concat(t[idx] * s, ())
  else:
    concat(t[idx] * s, buildStride(t, s, idx + 1))

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
        "compose: shape " & $typeof(remainingShape).V & " and consumed extent " & $typeof(clampedShape).V & " are not divisible"
    else:
      doAssert remainingShape mod clampedShape == 0,
        "compose: shape " & $remainingShape & " and consumed extent " & $clampedShape & " are not divisible"
  else:
    doAssert remainingShape mod clampedShape == 0,
      "compose: shape " & $remainingShape & " and consumed extent " & $clampedShape & " are not divisible"

func composeImpl(
    modeIdx:             static int;
    accShapes,
    accStrides,
    remainingShape,
    remainingStride:     auto;
    lhsShapes,
    lhsStrides:          tuple): auto {.inline.} =
  ## Fold over LHS modes with a 4-state accumulator
  ## (accShapes, accStrides, remainingShape, remainingStride).
  ## Uses recursion because the accumulator types change each iteration
  ## (shape/stride tuples grow via concat).
  when modeIdx >= rank(lhsShapes) - 1:
    ## Last mode (R-1): append remaining RHS as final mode,
    ## but skip when RHS was fully consumed (remaining is an Int[1] artifact).
    const skipLast =
      when remainingShape is Int and typeof(remainingShape) is Int[1] and rank(accShapes) != 0: true
      else: false
    when skipLast:
      make_layout(unwrap(accShapes), unwrap(accStrides))
    else:
      make_layout(unwrap(concat(accShapes, remainingShape)),
                  unwrap(concat(accStrides, remainingStride * lhsStrides[modeIdx])))
  else:
    ## Fold step for mode `modeIdx` (0 ≤ modeIdx < R-1).
    let currShape  = lhsShapes[modeIdx]
    let currStride = lhsStrides[modeIdx]
    let absRemainingStride = abs(remainingStride)
    let nextShape          = ceil_div(currShape, absRemainingStride)
    const doSkip =
      when nextShape is Int and typeof(nextShape) is Int[1]: true
      elif remainingShape is Int and typeof(remainingShape) is Int[1]: true
      else: false
    when doSkip:
      let nextStride = ceil_div(absRemainingStride, currShape) * sign(remainingStride)
      composeImpl(modeIdx+1, accShapes, accStrides, remainingShape, nextStride,
                  lhsShapes, lhsStrides)
    else:
      let clampedShape = min(nextShape, remainingShape)
      divisibilityCheck(remainingShape, clampedShape)
      composeImpl(modeIdx+1,
                  concat(accShapes, clampedShape),
                  concat(accStrides, remainingStride * currStride),
                  remainingShape div clampedShape,
                  ceil_div(absRemainingStride, currShape) * sign(remainingStride),
                  lhsShapes, lhsStrides)

func composeDistribute(lhsShapes, lhsStrides: tuple; rhsShapes, rhsStrides: tuple): auto =
  ## Layer RHS modes one by one over the FULL coalesced LHS via mapModesWith.
  ## Nested RHS modes are handled by recursive composeDistribute calls;
  ## scalar modes go directly to composeImpl.
  mapModesWith(make_layout(rhsShapes, rhsStrides)):
    when it.shape is tuple:
      composeDistribute(lhsShapes, lhsStrides, it.shape, it.stride)
    else:
      composeImpl(0, (), (), it.shape, it.stride, lhsShapes, lhsStrides)


func compose*[A, B: Layout](a: A, b: B): auto =
  ## Layout composition.
  ##
  ## Returns a layout `R` such that `R(i) = A(B(i))` for all
  ## `i` in `0 ..< cosize(B)`.
  when a.shape isnot tuple:
    when b.stride is tuple:
      when rank(b.shape) != rank(flatten(b.shape)):
        composeDistribute((a.shape,), (a.stride,), b.shape, b.stride)
      else:
        let bSh = b.shape.flatten()
        let bSt = b.stride.flatten()
        make_layout(bSh, buildStride(bSt, a.stride))
    else:
      make_layout(b.shape.flatten(), b.stride.flatten() * a.stride)
  elif b.shape isnot tuple:
    # CuTe: coalesce LHS first (preserving trailing stride-0 modes), then compose with scalar RHS
    # Uses coalesce_preserve_trailing to match CuTe's coalesce_x in composition.
    let flatA = coalesce_preserve_trailing(a)
    when flatA.shape isnot tuple:
      # flatA is rank-1 scalar: RHS shape is result, strides = b.stride * flatA.stride
      make_layout(b.shape, b.stride.scaleBy(flatA.stride))
    else:
      let aFlatShape = flatA.shape.flatten()
      let aFlatStride = flatA.stride.flatten()
      composeImpl(0, (), (), b.shape, b.stride, aFlatShape, aFlatStride)
  else:
    # CuTe: coalesce LHS first (preserving trailing stride-0 modes), then compose with tuple RHS
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
#  Tuple tiler:    transform_layout(logical_divide, layout, tiler)  — per-mode divide
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

template logical_divide*[L: Layout](layout: L; tiler: int): auto =
  ## Dynamic int tiler → wrap in Layout → CuTe formula.
  block:
    evalOnceAs(lyt, layout)
    evalOnceAs(tl, tiler)
    logical_divide_impl(lyt, make_layout(tl))

template logical_divide*[L: Layout; V: static int](layout: L; tiler: Int[V]): auto =
  ## Static int tiler (Int[N]) → wrap in Layout → CuTe formula.
  logical_divide_impl(layout, make_layout(tiler))

template logical_divide*[L: Layout](layout: L; tiler: static int): auto =
  ## Compile-time int tiler (const) → preserve via Int[N] wrap → CuTe formula.
  logical_divide_impl(layout, make_layout(Int[tiler]()))

template logical_divide_builder*(layout: untyped; tiler: untyped; LayoutRank: static int; idx: static int; accSh, accSt: typed): auto =
  ## Recursive build helper for logical_divide(tuple tiler).
  ## Exported (`*`) to avoid generic sandwich / template self-reference issues.
  when idx >= max(rank(tiler), LayoutRank):
    make_layout(accSh, accSt)
  else:
    when idx < rank(tiler):
      let d = logical_divide(mode(layout, idx), tiler[idx])
      logical_divide_builder(layout, tiler, LayoutRank, idx + 1, concat(accSh, (d.shape,)), concat(accSt, (d.stride,)))
    else:
      let m = mode(layout, idx)
      logical_divide_builder(layout, tiler, LayoutRank, idx + 1, concat(accSh, (m.shape,)), concat(accSt, (m.stride,)))

template logical_divide*(layout: Layout; tiler: tuple): auto =
  ## Tuple tiler → per-mode divide (transform_layout).
  ## Each tiler element applies to the corresponding layout mode.
  ## Modes beyond len(tiler) pass through unchanged.
  const R = static(rank(layout))
  static: doAssert rank(tiler) <= R,
    "logical_divide: tiler has more modes (" & $rank(tiler) &
    ") than layout (" & $R & ")"
  logical_divide_builder(layout, tiler, R, 0, (), ())

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

template zipped_divide_builder*(layout, tiler: typed; LayoutRank: static int; idx: static;
                                 tileSh, tileSt, restSh, restSt: typed): auto =
  ## Recursive build helper for zipped_divide(tuple tiler).
  ## One-pass: builds (tile, rest) groups directly without intermediate
  ## logical_divide + tile_unzip.
  ## Avoids Nim tuple hash collision
  ## (see https://github.com/nim-lang/Nim/issues/25883#issuecomment-4658908569).
  when idx >= LayoutRank:
    make_layout(
      (tileSh, restSh),
      (tileSt, restSt)
    )
  else:
    when idx < rank(tiler):
      evalOnceAs d, logical_divide(mode(layout, idx), tiler[idx])
      zipped_divide_builder(layout, tiler, LayoutRank, idx + 1,
        concat(tileSh, (mode(d, 0).shape,)),
        concat(tileSt, (mode(d, 0).stride,)),
        concat(restSh, (mode(d, 1).shape,)),
        concat(restSt, (mode(d, 1).stride,)))
    else:
      evalOnceAs m, mode(layout, idx)
      zipped_divide_builder(layout, tiler, LayoutRank, idx + 1,
        tileSh, tileSt,
        concat(restSh, (m.shape,)),
        concat(restSt, (m.stride,)))

template zipped_divide*(layout: Layout; tiler: auto): auto =
  ## Divide layout by tiler and zip tile/rest modes into rank-2 result.
  ##
  ## CuTe: zipped_divide =
  ##   - Layout tiler: logical_divide(layout, tiler)
  ##   - tuple/int tiler: tile_unzip(logical_divide(layout, tiler), tiler)
  block:
    evalOnceAs(lyt, layout)
    evalOnceAs(tlr, tiler)
    when tiler is Layout:
      logical_divide(lyt, tlr)
    elif tiler is int or tiler is Int:
      # Scalar tiler
      logical_divide(lyt, tlr)
    else:
      # Tuple tiler — one-pass builder avoids intermediate concat types
      # that trigger Nim C++ backend struct hash collision
      # (see https://github.com/nim-lang/Nim/issues/25883#issuecomment-4658908569)
      block:
        evalOnceAs lyt, layout
        evalOnceAs tlr, tiler
        const R = static(rank(lyt))
        const Tr = static(rank(tlr))
        static: doAssert Tr <= R,
          "zipped_divide: tiler has more modes (" & $Tr & ") than layout (" & $R & ")"
        zipped_divide_builder(lyt, tlr, R, 0, (), (), (), ())

template tiled_divide*(layout: Layout; tiler: auto): auto =
  ## Like zipped_divide but unpack the second mode into individual modes.
  ## Keeps mode-0 grouped (the tile).
  block:
    evalOnceAs(lyt, layout)
    evalOnceAs(tlr, tiler)
    evalOnceAs(zd, zipped_divide(lyt, tlr))
    make_layout(
      concat(
        (flatten(mode(zd, 0).shape),),
        flatten(mode(zd, 1).shape)
      ),
      concat(
        (flatten(mode(zd, 0).stride),),
        flatten(mode(zd, 1).stride)
      )
    )

template flat_divide*(layout: Layout; tiler: auto): auto =
  ## Like zipped_divide but unpack BOTH modes into a flat layout.
  ## Difference from tiled_divide: tile modes are also unpacked.
  block:
    evalOnceAs(lyt, layout)
    evalOnceAs(tlr, tiler)
    evalOnceAs(zd, zipped_divide(lyt, tlr))
    make_layout(
      concat(
        flatten(mode(zd, 0).shape),
        flatten(mode(zd, 1).shape),
      ),
      concat(
        flatten(mode(zd, 0).stride),
        flatten(mode(zd, 1).stride),
      ),
    )

# ═══════════════════════════════════════════════════════════════
#  right_inverse — quasi-inverse sorted by stride
# ═══════════════════════════════════════════════════════════════

proc rightInverseChain*(
    strides, shapes, prefixProd: seq[int]; shNode: NimNode): LayoutCT {.compileTime.} =
  ## Return right-inverse modes as LayoutCT (empty if no chain found).
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

  # Multi-mode: extract values, fill LayoutCT via helper
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
  ## Sorts modes by stride, finds max contiguous chain.
  let c = coalesce(layout)
  rightInverseImpl(flatten(c.shape), flatten(c.stride))

# ═══════════════════════════════════════════════════════════════
#  left_inverse — left inverse (injective layouts only)
# ═══════════════════════════════════════════════════════════════

proc leftInverseModes*(
    strides, shapes, prefixProd: seq[int]; shNode: NimNode): LayoutCT {.compileTime.} =
  ## Return left-inverse modes as a LayoutCT.
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
      # First mode: computed shape, zero stride
      result.append(IntCT(strides[idx] div sizeSoFar), IntCT(0))
    else:
      # Intermediate mode: computed shape, previous prefix as stride
      result.append(IntCT(strides[idx] div sizeSoFar), IntCT(prevPrefix))
    sizeSoFar = strides[idx]
    prevIdx = idx
    prevPrefix = prefixProd[idx]
  # Last mode from original layout
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
  let acc = leftInverseModes(strides, shapes, prefixProd, sh)
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
  ## Categorical product of two layouts, preserving each argument's mode grouping.
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
  ## Like zipped_product but unpack the second mode.
  ## Keeps mode-0 grouped (the block).
  block:
    evalOnceAs(bk, blk)
    evalOnceAs(tlr, tiler)
    evalOnceAs(zp, zipped_product(bk, tlr))
    make_layout(
      concat(
        (flatten(mode(zp, 0).shape),),
        flatten(mode(zp, 1).shape),
      ),
      concat(
        (flatten(mode(zp, 0).stride),),
        flatten(mode(zp, 1).stride),
      ),
    )

template flat_product*(blk: Layout; tiler: auto): auto =
  ## Like zipped_product but unpack BOTH modes into a flat layout.
  ## Difference from tiled_product: block modes are also unpacked.
  block:
    evalOnceAs(bk, blk)
    evalOnceAs(tlr, tiler)
    evalOnceAs(zp, zipped_product(bk, tlr))
    make_layout(
      concat(
        flatten(mode(zp, 0).shape),
        flatten(mode(zp, 1).shape),
      ),
      concat(
        flatten(mode(zp, 0).stride),
        flatten(mode(zp, 1).stride),
      ),
    )

# ═══════════════════════════════════════════════════════════════
#  blocked_product — blocks laid out contiguously
# ═══════════════════════════════════════════════════════════════
#
#  blocked_product(block, tiler):
#    1. Append both to rank R = max(rank(block), rank(tiler))
#    2. result = logical_product(padded_block, padded_layout)
#    3. return zipModes(result[0], result[1])

func blocked_product*[A, B: Layout](blk: A; tiler: B): auto =
  ## Repeat block over tiler grid, each block contiguous.
  ## Results in ((BLK_A, TILER_A), (BLK_B, TILER_B), ...).
  const mxR = max(rank(type(blk)), rank(type(tiler)))
  let lp = logical_product(padRight(blk, mxR), padRight(tiler, mxR))
  let m0 = mode(lp, 0)
  let m1 = mode(lp, 1)
  zipModes(m0, m1)

# ═══════════════════════════════════════════════════════════════
#
#  raked_product(block, tiler):
#    1. Same logical_product as blocked_product
#    2. return zipModes(result[1], result[0])  (swapped order)

func raked_product*[A, B: Layout](blk: A; tiler: B): auto =
  ## Repeat block over tiler grid, blocks interleaved.
  ## Results in ((TILER_A, BLK_A), (TILER_B, BLK_B), ...).
  const mxR = max(rank(type(blk)), rank(type(tiler)))
  let lp = logical_product(padRight(blk, mxR), padRight(tiler, mxR))
  let m0 = mode(lp, 0)
  let m1 = mode(lp, 1)
  zipModes(m1, m0)

# ═══════════════════════════════════════════════════════════════
#  tile_to_shape — repeat block layout to fill target shape
# ═══════════════════════════════════════════════════════════════

template tile_to_shape*(blk: Layout; target_shape: typed; ord_shape: static StrideOrder = LayoutLeft): auto =
  ## Recipe:
  ##   1. Pad block to rank R
  ##   2. Compute block_shape = product_each(block.shape)   — per-mode products
  ##   3. Compute target_shape_flat = product_each(target_shape)    — per-mode products
  ##   4. product_shape = ceil_div(target_shape, block_shape) — repeats per mode
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
    let product_shape = zipModesWith(trg_flat, blk_shape): ceil_div(it_a, it_b)
    let tiler = make_layout(product_shape, ord_shape)
    blocked_product(padded_blk, tiler)

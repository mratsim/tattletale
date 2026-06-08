# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## CuTe-compatible layout algebra: coalesce, filter_zeros, filter, sort.

import std/macros
import std/sequtils
import std/algorithm
import std/typetraits
import ./int_tuples
import ./layouts

# ═══════════════════════════════════════════════════════════════
#  coalesce — merge contiguous modes where stride matches
# ═══════════════════════════════════════════════════════════════

macro coalesceBackward(csShape, csStride: typed): untyped =
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

func coalesce*(layout: Layout): auto =
  ## Merge contiguous modes. Flatten preserves Int[N] types with getTypeInst.
  coalesceBackward(
    flatten(layout.shape),
    flatten(layout.stride)
  )

func coalesce*(layout: static Layout): static Layout =
  ## Static overload: preserves constness for compile-time Layout values.
  coalesce(layout)

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

proc complementMulti(sh, st, boundExpr: NimNode): NimNode {.compileTime.} =
  ## Multi-mode complement: sort by stride, fold to fill gaps.
  ## All strides must be static Int[N] (compile-time check).

  # ── Read types from the typed AST nodes ──
  # `sh` and `st` are flattened shape/stride AST coming from complementImpl,
  # which received them as typed macro params.  Their type ASTs (getTypeInst)
  # are tuple types like `(Int[2], Int[4])` — we inspect tuple elements below.
  let stTyp = st.getTypeInst()  # e.g. `(Int[1], Int[4], Int[8])`
  let shTyp = sh.getTypeInst()  # e.g. `(Int[2], Int[1], Int[2])`

  # ── Collect (stride, shape) pairs ──
  # Strides must be Int[N] (CuTe requires static strides for multi-mode).
  # Shapes may be Int[N] (static) or `int` (dynamic).
  type Md = tuple[stVal: int,         # stride value (static, Int[N])
                   shNode: NimNode,   # shape AST node (static Int[N] or runtime int)
                   shStatic: bool,    # true if shape is Int[N]
                   shVal: int]        # shape value (0 if dynamic)
  var modes: seq[Md]

  for i in 0 ..< stTyp.len:
    let stNode = stTyp[i]; let shNode = shTyp[i]
    doAssert stNode.kind == nnkBracketExpr and $stNode[0] == "Int",
      "complement: multi-mode with dynamic strides not supported"
    let shSt = shNode.kind == nnkBracketExpr and $shNode[0] == "Int"
    modes.add (stVal: int(stNode[1].intVal),
               shNode: shNode,
               shStatic: shSt,
               shVal: (if shSt: int(shNode[1].intVal) else: 0))

  # ── Sort by stride ascending ──
  # This ensures we process strides in order so gap-detection is correct
  # (e.g. stride-2 before stride-4).  When strides are equal: static-shape
  # modes sort before dynamic-shape modes (deterministic ordering).
  modes.sort do (a, b: Md) -> int:
    result = cmp(a.stVal, b.stVal)
    if result == 0:
      if a.shStatic and b.shStatic: result = cmp(a.shVal, b.shVal)
      elif a.shStatic: result = -1
      elif b.shStatic: result = 1

  # ── Scan loop: emit gap modes + final remainder ──
  # `cur` tracks cumulative coverage = st × sh of processed modes (Nim int).
  # Gap check `m.stVal > cur` is always CT since strides are Int[N].
  # Remainder is always `ceil_div(bound, cur)`
  # `LayoutCT.emit()` resolves
  # automatically constant folds
  var acc = LayoutCT()
  var cur = 1

  for idx, m in modes:
    let gap = if m.stVal > cur: m.stVal div cur else: 1
    if gap > 1:
      acc.append(IntCT(gap), IntCT(cur))

    if not m.shStatic:
      # Dynamic shape — can't advance cur.  Emit remainder at current coverage.
      break

    cur = m.stVal * m.shVal

  # All modes processed — emit final remainder
  let rem = newCall(bindSym"ceil_div", boundExpr, IntCT(cur))
  acc.append(rem, IntCT(cur))
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
#  compose — CuTe-compatible layout composition
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

func buildStride(t: tuple; s: int or Int, idx: static int = 0): auto {.inline.} =
  # Broadcast helper: multiply each element of tuple t by scalar s
  # Builds concat(t[0]*s, concat(t[1]*s, ... ())) at compile time
  # via recursive template. No heap, no seq, no macro type introspection.
  when idx == t.tupleLen() - 1:
    concat(t[idx] * s, ())
  else:
    concat(t[idx] * s, buildStride(t, s, idx + 1))

func sign[V](x: Int[V]): Int[if V>0: 1 elif V<0: -1 else: 0] = discard

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
  when modeIdx >= lhsShapes.tupleLen() - 1:
    ## Last mode (R-1): append remaining RHS as final mode,
    ## but skip when RHS was fully consumed (remaining is an Int[1] artifact).
    const skipLast =
      when remainingShape is Int and typeof(remainingShape) is Int[1] and accShapes.tupleLen != 0: true
      else: false
    when skipLast:
      make_layout(accShapes, accStrides)
    else:
      make_layout(concat(accShapes, remainingShape),
                  concat(accStrides, remainingStride * lhsStrides[modeIdx]))
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

func composeDistribute(
    lhsShapes, lhsStrides: tuple;
    rhsShapes, rhsStrides: tuple;
    idx: static int = 0): auto {.inline.} =
  ## Layer RHS modes one by one over the FULL coalesced LHS.
  ##
  ## Each scalar RHS mode is composed against ALL lhs modes via
  ## composeImpl (a fold over LHS modes). Results are nested:
  ## make_layout(compose(LHS, RHS_mode_0), compose(LHS, RHS_mode_1), ...)
  let r = when typeof(rhsShapes[idx]) is tuple:
    composeDistribute(lhsShapes, lhsStrides, rhsShapes[idx], rhsStrides[idx])
  else:
    composeImpl(0, (), (), rhsShapes[idx], rhsStrides[idx], lhsShapes, lhsStrides)
  when idx >= rhsShapes.tupleLen() - 1:
    r
  else:
    let rest = composeDistribute(lhsShapes, lhsStrides, rhsShapes, rhsStrides, idx + 1)
    make_layout((r.shape, rest.shape), (r.stride, rest.stride))

func compose*[A, B: Layout](a: A, b: B): auto =
  ## Layout composition.
  ##
  ## Returns a layout `R` such that `R(i) = A(B(i))` for all
  ## `i` in `0 ..< cosize(B)`.
  when a.shape isnot tuple:
    when b.stride is tuple:
      when tupleLen(b.shape) != tupleLen(flatten(b.shape)):
        composeDistribute((a.shape,), (a.stride,), b.shape, b.stride)
      else:
        let bSh = b.shape.flatten()
        let bSt = b.stride.flatten()
        make_layout(bSh, buildStride(bSt, a.stride))
    else:
      make_layout(b.shape.flatten(), b.stride.flatten() * a.stride)
  elif b.shape isnot tuple:
    composeImpl(0, (), (), b.shape.flatten(), b.stride.flatten(), a.shape, a.stride)
  else:
    let flatA = coalesce(a)
    when flatA.shape isnot tuple:
      make_layout(b.shape.flatten(), buildStride(b.stride.flatten(), flatA.stride))
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

func logical_divide*[L: Layout](layout: L; tiler: int): auto =
  ## Dynamic int tiler → wrap in Layout → CuTe formula.
  logical_divide_impl(layout, make_layout(tiler))

func logical_divide*[L: Layout; V: static int](layout: L; tiler: Int[V]): auto =
  ## Static int tiler (Int[N]) → wrap in Layout → CuTe formula.
  logical_divide_impl(layout, make_layout(tiler))

func logical_divide*[L: Layout](layout: L; tiler: static int): auto =
  ## Compile-time int tiler (const) → preserve via Int[N] wrap → CuTe formula.
  logical_divide_impl(layout, make_layout(Int[tiler]()))

func logical_divide*[L: Layout](layout: L; tiler: tuple): auto =
  ## Tuple tiler → per-mode divide (transform_layout).
  ## Each tiler element applies to the corresponding layout mode.
  ## Modes beyond len(tiler) pass through unchanged.
  const LayoutRank =
    when L.Sh is tuple:
      L.Sh.tupleLen()
    else:
      1
  static: doAssert tupleLen(tiler) <= LayoutRank,
    "logical_divide: tiler has more modes (" & $tupleLen(tiler) &
    ") than layout (" & $LayoutRank & ")"

  template build(idx: static int; accSh, accSt: typed): auto =
    when idx >= max(tupleLen(tiler), LayoutRank):
      make_layout(accSh, accSt)
    else:
      when idx < tupleLen(tiler):
        let d = logical_divide(mode(layout, idx), tiler[idx])
        build(idx + 1, concat(accSh, (d.shape,)), concat(accSt, (d.stride,)))
      else:
        let m = mode(layout, idx)
        build(idx + 1, concat(accSh, (m.shape,)), concat(accSt, (m.stride,)))
  build(0, (), ())

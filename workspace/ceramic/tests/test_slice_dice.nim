## CuTe-compatible slice and dice tests.
##
## Reference:
##   [CUTE]  = CuTe C++: underscore.hpp, layout.hpp
##   [MOYE]  = MoYe.jl: int_tuple.jl, stride.jl
##   [PY-TL] = tensor-layouts: external.py, layouts.py
##
## Convention:
##   `_` — keep/slice dimension (CuTe Underscore)
##   int   — fix/dice dimension
##
## Note: Nim's `_` is a declaration placeholder, so we use `_`.
## Inside `layout[_, ...]` brackets `_` would work (special parsing),
## but for standalone calls the identifier is `_`.
##
## Tuple-level: slice(coord, target), dice(coord, target)
## Layout-level: slice(coord, layout), dice(coord, layout),
##                slice_and_offset(coord, layout)
import std/macros
import std/typetraits
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts

# ..< overload for Int[N] ranges
func `..<`*[V: static int](a: int; b: Int[V]): Slice[int] = a ..< V


# ═══════════════════════════════════════════════════════════════
#  Test entry point
# ═══════════════════════════════════════════════════════════════

proc runSliceDiceTests*: void

when isMainModule:
  runSliceDiceTests()

# ═══════════════════════════════════════════════════════════════
#  Forward declarations
# ═══════════════════════════════════════════════════════════════

proc runJokerTests*
proc runSliceScalarTests*
proc runSliceFlatTupleTests*
proc runSliceNestedTupleTests*
proc runDiceScalarTests*
proc runDiceFlatTupleTests*
proc runDiceNestedTupleTests*
proc runSliceLayoutBasicTests*
proc runDiceLayoutBasicTests*
proc runSliceAndOffsetBasicTests*
proc runIdx2crdTests*
proc runIndirectionTests*
proc runGenericProcTests*

# ═══════════════════════════════════════════════════════════════
#  run all
# ═══════════════════════════════════════════════════════════════

proc runSliceDiceTests* =
  echo "\n── Joker type [CUTE] ──"
  runJokerTests()
  echo "\n── slice on scalars [CUTE] ──"
  runSliceScalarTests()
  echo "\n── slice on flat tuples [CUTE][PY-TL] ──"
  runSliceFlatTupleTests()
  echo "\n── slice on nested tuples [CUTE][MOYE] ──"
  runSliceNestedTupleTests()
  echo "\n── dice on scalars [CUTE] ──"
  runDiceScalarTests()
  echo "\n── dice on flat tuples [CUTE][PY-TL] ──"
  runDiceFlatTupleTests()
  echo "\n── dice on nested tuples [CUTE] ──"
  runDiceNestedTupleTests()
  echo "\n── slice/dice/slice_and_offset on Layout [CUTE] ──"
  runSliceLayoutBasicTests()
  runDiceLayoutBasicTests()
  runSliceAndOffsetBasicTests()
  echo "\n── idx2crd (index → coordinate) [MOYE] ──"
  runIdx2crdTests()
  echo "\n── Indirection: const, let, `_` syntax ──"
  runIndirectionTests()
  echo "\n── Generic proc ──"
  runGenericProcTests()
  echo "\nALL SLICE/DICE TESTS PASSED"

# ═══════════════════════════════════════════════════════════════
#  A. Joker type
# ═══════════════════════════════════════════════════════════════

proc runJokerTests* =
  block:
    doAssert Joker isnot int
    doAssert Joker isnot Int
  block:
    doAssert _ is Joker
  echo "  2 checks OK"

# ═══════════════════════════════════════════════════════════════
#  B. slice on scalars [CUTE underscore.hpp]

proc runSliceScalarTests* =
  block:
    let r = slice(_, 4)
    doAssert r === 4
  block:
    let r = slice(0, 4)
    doAssert (r is tuple)
    doAssert tupleLen(r) == 0
  block:
    let r = slice(1, 4)
    doAssert (r is tuple)
    doAssert tupleLen(r) == 0
  block:
    let r = slice(_, Int[8]())
    doAssert r === Int[8]()
  echo "  4 cases OK"

# ═══════════════════════════════════════════════════════════════
#  C. slice on flat tuples

proc runSliceFlatTupleTests* =
  block:
    let r = slice((_, 0), (3, 4))
    doAssert (r is tuple)
    doAssert tupleLen(r) == 1
    doAssert r[0] === 3
  block:
    let r = slice((0, _), (3, 4))
    doAssert (r is tuple)
    doAssert tupleLen(r) == 1
    doAssert r[0] === 4
  block:
    let r = slice((_, _), (3, 4))
    doAssert (r is tuple)
    doAssert tupleLen(r) == 2
    doAssert r[0] === 3
    doAssert r[1] === 4
  block:
    let r = slice((0, 0), (3, 4))
    doAssert (r is tuple)
    doAssert tupleLen(r) == 0
  block:
    let r = slice((_, 0), (3, Int[4]()))
    doAssert tupleLen(r) == 1
    doAssert r[0] === 3
  echo "  5 cases OK"

# ═══════════════════════════════════════════════════════════════
#  D. slice on nested tuples

proc runSliceNestedTupleTests* =
  block:
    let r = slice((0, _), (3, 4))
    doAssert tupleLen(r) == 1
    doAssert r[0] === 4
  block:
    let r = slice(((_, 0), _), ((3, 4), 5))
    doAssert tupleLen(r) == 2
    doAssert r[0] === 3
    doAssert r[1] === 5
  block:
    let r = slice((_, (_, 0)), (3, (4, 5)))
    doAssert tupleLen(r) == 2
    doAssert r[0] === 3
    doAssert r[1] === 4
  block:
    let r = slice((_, (_, _)), ((2, 4), (4, 2)))
    doAssert tupleLen(r) == 3
  echo "  4 cases OK"

# ═══════════════════════════════════════════════════════════════
#  E. dice on scalars

proc runDiceScalarTests* =
  block:
    let r = dice(_, 4)
    doAssert (r is tuple)
    doAssert tupleLen(r) == 0
  block:
    let r = dice(0, 4)
    doAssert r === 4
  block:
    let r = dice(5, 4)
    doAssert r === 4
  echo "  3 cases OK"

# ═══════════════════════════════════════════════════════════════
#  F. dice on flat tuples

proc runDiceFlatTupleTests* =
  block:
    let r = dice((0, _), (3, 4))
    doAssert r === 3
  block:
    let r = dice((_, 0), (3, 4))
    doAssert r === 4
  block:
    let r = dice((0, 0), (3, 4))
    doAssert (r is tuple)
    doAssert tupleLen(r) == 2
    doAssert r[0] === 3
    doAssert r[1] === 4
  block:
    let r = dice((_, _), (3, 4))
    doAssert (r is tuple)
    doAssert tupleLen(r) == 0
  echo "  4 cases OK"

# ═══════════════════════════════════════════════════════════════
#  G. dice on nested tuples

proc runDiceNestedTupleTests* =
  block:
    let r = dice(((0, _), 0), ((3, 4), 5))
    doAssert r === (3, 5)
  block:
    let r = dice(((_, 0), 0), ((3, 4), 5))
    doAssert (r is tuple)
    doAssert tupleLen(r) == 2
    doAssert r[0] === 4
    doAssert r[1] === 5
  echo "  2 cases OK"

# ═══════════════════════════════════════════════════════════════
#  H. slice/dice/slice_and_offset on Layout

proc runSliceLayoutBasicTests* =
  block:
    let L = make_layout((4, 8), (1, 4))
    let sub = slice((_, 0), L)
    doAssert sub === (4, 1), "got " & $sub
  block:
    let L = make_layout((4, 8), (1, 4))
    let sub = slice((0, _), L)
    doAssert sub === (8, 4), "got " & $sub
  block:
    let L = make_layout((4, 8), (1, 4))
    let sub = slice((_, _), L)
    doAssert sub === L
  block:
    let L = make_layout((4, 8), (1, 4))
    let sub = slice((0, 0), L)
    doAssert size(sub) === 1
  block:
    let L = make_layout(4, 1)
    let sub = slice(_, L)
    doAssert sub === L
  block:
    let L = make_layout((2, 3, 4), (1, 2, 6))
    let sub = slice((_, 1, _), L)
    doAssert sub.shape[0] === 2
    doAssert sub.shape[1] === 4
    doAssert sub.stride[0] === 1
    doAssert sub.stride[1] === 6
  echo "  6 cases OK"

proc runDiceLayoutBasicTests* =
  block:
    let L = make_layout((3, 4), (1, 4))
    let sub = dice((0, _), L)
    doAssert sub === (3, 1), "got " & $sub
  block:
    let L = make_layout((3, 4), (1, 4))
    let sub = dice((_, 0), L)
    doAssert sub === (4, 4), "got " & $sub
  block:
    let L = make_layout((3, 4), (1, 4))
    let sub = dice((0, 0), L)
    doAssert sub === L
  block:
    let L = make_layout((3, 4), (1, 4))
    let sub = dice((_, _), L)
    doAssert size(sub) === 1
  block:
    let L = make_layout((3, 4), (1, 4))
    let sub = dice(0, L)
    doAssert sub === L
  echo "  5 cases OK"

proc runSliceAndOffsetBasicTests* =
  block:
    let L = make_layout((4, 8), (1, 4))
    let (sub, off) = slice_and_offset((_, 3), L)
    doAssert sub === (4, 1), "sub: " & $sub
    doAssert off === 12, "off: " & $off
  block:
    let L = make_layout((4, 8), (1, 4))
    let (sub, off) = slice_and_offset((2, _), L)
    doAssert sub === (8, 4), "sub: " & $sub
    doAssert off === 2, "off: " & $off
  block:
    let L = make_layout((4, 8), (1, 4))
    let (sub, off) = slice_and_offset((2, _), L)
    for i in 0 ..< size(sub):
      doAssert sub[i] + off === L[(2, i)],
        "i=" & $i & ": " & $(sub[i] + off) & " != " & $(L[(2, i)])
  block:
    let L = make_layout((4, 8), (1, 4))
    let (sub, off) = slice_and_offset((_, Int[3]()), L)
    doAssert sub === (4, 1), "sub: " & $sub
    doAssert off === 12, "off: " & $off
  block:
    let L = make_layout((2, 3, 4), (1, 2, 6))
    let (sub, off) = slice_and_offset((_, 1, _), L)
    doAssert off === 2, "3d off: " & $off
    for i in 0 ..< 2:
      for k in 0 ..< 4:
        doAssert sub[(i, k)] + off === L[(i, 1, k)],
          "(" & $i & "," & $k & "): " & $(sub[(i,k)] + off) & " != " & $(L[(i,1,k)])
  echo "  5 cases OK"

# ═══════════════════════════════════════════════════════════════
#  I. idx2crd

proc runIdx2crdTests* =
  block:
    let crd = idx2crd(5, (3, 4))
    doAssert crd[0] === 2
    doAssert crd[1] === 1
  block:
    let crd = idx2crd(0, (3, 4))
    doAssert crd[0] === 0
    doAssert crd[1] === 0
  block:
    let crd = idx2crd(11, (3, 4))
    doAssert crd[0] === 2
    doAssert crd[1] === 3
  block:
    let crd = idx2crd(9, (3, 4), (1, 3))
    doAssert crd[0] === 0
    doAssert crd[1] === 3
  echo "  4 cases OK"

# ═══════════════════════════════════════════════════════════════
#  J. Indirection: const, let

proc runIndirectionTests* =
  block:
    ## const indirection: const values -> Int[N]
    const C3 = 3
    const C4 = 4
    let r = slice((_, 0), (C3, C4))
    doAssert tupleLen(r) == 1
    doAssert r[0] === 3
  block:
    ## let indirection: runtime int values
    let d3 = 3
    let d4 = 4
    let r = slice((_, 0), (d3, d4))
    doAssert tupleLen(r) == 1
    doAssert r[0] === 3
  block:
    ## _ syntax on Layout
    let L = make_layout((4, 8), (1, 4))
    let sub = slice((_, 0), L)
    doAssert sub === (4, 1)
  block:
    ## _ syntax on Layout with slice_and_offset
    let L = make_layout((4, 8), (1, 4))
    let (sub, off) = slice_and_offset((2, _), L)
    doAssert off === 2
  block:
    ## const Int[N] as target
    const C2 = 2; const C4 = 4
    let l = make_layout((C2, C4))
    let sub = slice((_, 0), l)
    doAssert sub === (2, 1)
  block:
    ## IntOrIntTuple constraint: slice with Joker
    let r = slice((_, 0), (3, 4))
    doAssert tupleLen(r) == 1
    doAssert r[0] === 3
  block:
    ## const in coord
    const cIdx = 0
    let L = make_layout((4, 8), (1, 4))
    let sub = slice((_, cIdx), L)
    doAssert sub === (4, 1)
  block:
    ## let in coord
    let dIdx = 0
    let L = make_layout((4, 8), (1, 4))
    let sub = slice((_, dIdx), L)
    doAssert sub === (4, 1)
  block:
    ## const shapes in layout
    const C2 = 2; const C4 = 4
    let L = make_layout((C2, C4))
    let sub = slice((_, 0), L)
    doAssert sub === (2, 1)
  block:
    ## let shapes in layout
    let d2 = 2; let d4 = 4
    let L = make_layout((d2, d4))
    let sub = slice((_, 0), L)
    doAssert sub === (2, 1)
  block:
    ## const in layout + const in coord combined
    const C2 = 2; const C4 = 4; const cIdx = 0
    let L = make_layout((C2, C4))
    let sub = slice((_, cIdx), L)
    doAssert sub === (2, 1)
  block:
    ## let in layout + let in coord combined
    let d2 = 2; let d4 = 4; let dIdx = 0
    let L = make_layout((d2, d4))
    let sub = slice((_, dIdx), L)
    doAssert sub === (2, 1)
  echo "  12 cases OK"

# ═══════════════════════════════════════════════════════════════
#  K. Generic proc

proc runGenericProcTests* =
  block:
    ## Generic proc: slice inside a generic function
    proc foo[T](shape: T; stride: T): auto =
      let L = make_layout(shape, stride)
      slice((_, 0), L)

    let r = foo((4, 8), (1, 4))
    doAssert r === (4, 1), "generic proc: " & $r

  block:
    ## Generic proc with _ inside
    proc bar[T](a: T; b: T): auto =
      slice((_, 0), (a, b))

    let r = bar(3, 4)
    doAssert tupleLen(r) == 1
    doAssert r[0] === 3

  block:
    ## Generic proc: dice with inline coord
    proc qux[T](target: T): auto =
      dice((0, _), target)

    let r = qux((3, 4))
    doAssert r === 3

  block:
    ## Generic proc: const shapes + const coord
    proc gem[T](a: T; b: T): auto =
      const C = 0
      slice((_, C), (a, b))
    let r = gem(3, 4)
    doAssert tupleLen(r) == 1
    doAssert r[0] === 3

  block:
    ## Generic proc: let shapes + let coord
    proc hem[T](a: T; b: T): auto =
      let C = 0
      slice((_, C), (a, b))
    let r = hem(3, 4)
    doAssert tupleLen(r) == 1
    doAssert r[0] === 3

  echo "  5 cases OK"

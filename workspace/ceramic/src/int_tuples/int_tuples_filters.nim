## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import ./int_tuples_datatypes
import ./int_tuples_transforms

proc substIt(ast, aElem, bElem: NimNode): NimNode =
  ## Replace `it_a` with `aElem`, `it_b` with `bElem` in `ast`.
  proc inspect(n: NimNode): NimNode =
    if n.kind in {nnkIdent, nnkSym}:
      if n.eqIdent("it_a"):
        return aElem
      if n.eqIdent("it_b"):
        return bElem
    if n.len == 0:
      return n
    result = n.kind.newTree()
    for child in n:
      result.add inspect(child)
  result = inspect(ast)

func tupleType(n: NimNode): NimNode {.compileTime.} =
  ## Resolve to the underlying TupleConstr node, handling values, consts,
  ## and type aliases uniformly.
  let t = n.getType()
  let inner =
    if t.kind == nnkBracketExpr and t[0].eqIdent("typeDesc"):
      t[1]
    else:
      t
  inner.getTypeImpl()

func isTuple(n: NimNode): bool {.compileTime.} =
  n.kind in {nnkTupleConstr, nnkPar} or n.tupleType().kind in {nnkTupleConstr, nnkTupleTy}

func tupleTypeLen(n: NimNode): int {.compileTime.} =
  n.tupleType().len

func tupleElement(n: NimNode; i: int): NimNode {.compileTime.} =
  ## For literal tuples return the raw AST child (value or type symbol),
  ## for indirected tuples return the type symbol from the resolved type.
  if n.kind in {nnkTupleConstr, nnkPar}:
    n[i]
  else:
    n.tupleType()[i]

macro filterZipWith*(a: typed; b: typed; body: untyped): untyped =
  ## Zip two tuples element-wise, apply `body` to each pair, and
  ## concatenate the kept elements into a flat result tuple.
  ##
  ## `a`, `b` — two tuples of same length, zipped element-wise
  ## `body` — expression returning `()` to drop or `(expr,)` to keep
  ##
  ## `it_a` and `it_b` are injected for the corresponding elements.
  ## The keep/drop decision must be compile-time.
  ##
  ## Pattern: positions in `a` control keep/drop based on type:
  ##   type X = object
  ##   filterZipWith((X, X), (10, 20)):
  ##     (when it_a is X: (it_b,) else: ())
  ##
  ## Multiple marker types:
  ##   type Y = object
  ##   filterZipWith((X, Y), (10, 20)):
  ##     (when it_a is X: (it_b,) elif it_a is Y: (it_a,) else: ())
  if a.isTuple():
    let n = a.tupleTypeLen()
    let bLen = b.tupleTypeLen()
    doAssert n == bLen, "filterZipWith: rank mismatch (" & $n & " vs " & $bLen & ")"
    var parts: seq[NimNode] = @[]
    for i in 0 ..< n:
      let subA = a.tupleElement(i)
      let subB = if b.kind in {nnkTupleConstr, nnkPar}: b[i] else: nnkBracketExpr.newTree(b, newLit(i))
      let subIsTuple = subA.isTuple()
      if subIsTuple:
        parts.add newCall(bindSym"filterZipWith", subA, subB, body)
      else:
        parts.add substIt(body, subA, subB)
    if parts.len == 0:
      result = nnkTupleConstr.newTree()
    else:
      result = nnkTupleConstr.newTree()
      for i in countdown(parts.len - 1, 0):
        result = newCall(bindSym"concat", parts[i], result)
  else:
    result = substIt(body, a, b)

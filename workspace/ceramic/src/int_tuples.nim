# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.


import std/macros
import std/typetraits
import ./macros/static_for

# ═══════════════════════════════════════════════════════════════
#  Int[N] — compile-time integer type
# ═══════════════════════════════════════════════════════════════

type Int*[V: static int] = object
  ## Compile-time integer literal, analogous to CuTe's `Int<N>`.
  ##   Int<4>  ⇔  Int[4]
  ##
  ## Only `static int` values can inhabit Int[N] — no runtime dispatching.

type IntOrIntTuple* = int | Int | tuple
  ## Shape/stride element type alias for convenience.

converter toInt*[V: static int](x: Int[V]): int = V

func toIntVal*(x: int): int = x
func toIntVal*[V: static int](x: Int[V]): int = V

func `$`*[V: static int](x: Int[V]): string = $V

# ═══════════════════════════════════════════════════════════════
#  isConst — compile-time detection (runtime via proc dispatch)
# ═══════════════════════════════════════════════════════════════

proc isConst*(a: static int): static bool = true
template isConst*(a: int): bool = false
proc isConst*[V: static int](a: Int[V]): static bool = true

# ═══════════════════════════════════════════════════════════════
#  Int[N] == int — global overloads for tuple comparison
# ═══════════════════════════════════════════════════════════════

func `<=`*[V: static int](a: Int[V]; b: int): bool {.inline.} = V <= b
func `<=`*[V: static int](a: int; b: Int[V]): bool {.inline.} = a <= V
func `>=`*[V: static int](a: Int[V]; b: int): bool {.inline.} = V >= b
func `>=`*[V: static int](a: int; b: Int[V]): bool {.inline.} = a >= V
func `<=`*[V, U: static int](a: Int[V]; b: Int[U]): static bool = V <= U
func `>=`*[V, U: static int](a: Int[V]; b: Int[U]): static bool = V >= U

# ═══════════════════════════════════════════════════════════════
#  `===` — deep element-wise tuple comparison (handles Int[N] vs int)
# ═══════════════════════════════════════════════════════════════

func `===`*(a, b: int): bool {.inline.} = a == b
func `===`*(a, b: static int): static bool = a == b
func `===`*[V, U: static int](a: Int[V]; b: Int[U]): static bool = V == U

func `===`*[V: static int](a: Int[V]; b: int): bool {.inline.} = V == b
func `===`*[V: static int](a: int; b: Int[V]): bool {.inline.} = a == V
func `===`*[V: static int](a: Int[V]; b: static int): static bool = V == b
func `===`*[V: static int](a: static int; b: Int[V]): static bool = a == V

func `===`*[T: tuple, U: tuple](a: T; b: U): bool {.inline.} =
  ## Deep element-wise tuple comparison.
  ## Handles Int[N] vs int mismatches via per-element === overloads.
  when tupleLen(T) != tupleLen(U):
    false
  else:
    staticFor i, 0, tupleLen(T):
      if not (a[i] === b[i]):
        return false
    true

func `===`*[T: tuple](a: T; b: int): bool {.inline.} =
  ## Compare a tuple against an int — only valid for 1-element tuples.
  when tupleLen(T) == 1:
    a[0] === b
  else:
    false

func `===`*[U: tuple](a: int; b: U): bool {.inline.} =
  ## Compare an int against a tuple — only valid for 1-element tuples.
  when tupleLen(U) == 1:
    a === b[0]
  else:
    false

# ═══════════════════════════════════════════════════════════════
#  `!==` — negation of deep element-wise tuple comparison
# ═══════════════════════════════════════════════════════════════

func `!==`*(a, b: auto): bool {.inline.} = not (a === b)

# ═══════════════════════════════════════════════════════════════
#  Int[N] arithmetic
# ═══════════════════════════════════════════════════════════════

func ceil_div*(a, b: int): int =
  (a + b - 1) div b

func abs*[V: static int](x: Int[V]): Int[abs(V)] = Int[abs(V)]()

func sign*[V: static int](x: Int[V]): Int[if V > 0: 1 elif V < 0: -1 else: 0] = discard

template genBinOp(op: untyped): untyped =
  func op*[V, U: static int](a: Int[V]; b: Int[U]): Int[op(V, U)] = Int[op(V, U)]()
  func op*[V: static int](a: Int[V]; b: static int): Int[op(V, b)] = Int[op(V, b)]()
  func op*[V: static int](a: static int; b: Int[V]): Int[op(a, V)] = Int[op(a, V)]()
  func op*[V: static int](a: Int[V]; b: int): int = op(V, b)
  func op*[V: static int](a: int; b: Int[V]): int = op(a, V)

genBinOp(`+`)
genBinOp(`-`)
genBinOp(`*`)
genBinOp(`div`)
genBinOp(`mod`)

genBinOp(`max`)
genBinOp(`min`)
genBinOp(`ceil_div`)

func `+=`*[V: static int](a: var int; b: Int[V]) = a += V
func `+=`*[V: static int](a: var Int[V]; b: int) = a = Int[V](V + b)

# ═══════════════════════════════════════════════════════════════
#  Int[N] compile-time helpers (for macros)
# ═══════════════════════════════════════════════════════════════

func IntCT*(val: int): NimNode {.compileTime.} =
  ## Shorthand: Int[val]() AST node.
  newNimNode(nnkObjConstr).add(
    newNimNode(nnkBracketExpr).add(ident"Int", newLit(val)))

func isStaticInt*(t: NimNode): bool {.compileTime.} =
  (t.kind == nnkBracketExpr and $t[0] == "Int") or t.kind == nnkIntLit

func isStaticOne*(t: NimNode): bool {.compileTime.} =
  (t.kind == nnkBracketExpr and $t[0] == "Int" and t[1].intVal == 1) or
  (t.kind == nnkIntLit and t.intVal == 1)

func getStaticInt*(t: NimNode): int {.compileTime.} =
  if t.kind == nnkBracketExpr and $t[0] == "Int": int(t[1].intVal)
  elif t.kind == nnkIntLit: int(t.intVal)
  else: error("getStaticInt on non-static: " & t.repr)

func toSeqStaticInts*(t: NimNode): seq[int] {.compileTime.} =
  ## Extract Int[N] values from a tuple type AST node.
  ## Returns 0 for non-static (dynamic int) elements.
  for i in 0 ..< t.len:
    let node = t[i]
    if node.kind == nnkBracketExpr and $node[0] == "Int":
      result.add int(node[1].intVal)
    else:
      result.add 0

func prefixProduct*(vals: seq[int]): seq[int] {.compileTime.} =
  ## Prefix product of a flat seq (0 entries treated as 1 for scan,
  ## but produce 0 in output to mark unknown positions).
  result = @[1]
  for i in 0 ..< vals.len:
    if vals[i] != 0:
      result.add result[^1] * vals[i]
    else:
      result.add 0

# ═══════════════════════════════════════════════════════════════
#  evalOnceAs — evaluate at most once, preserve Int[N] for CT exprs
# ═══════════════════════════════════════════════════════════════

macro evalOnceAs*(expAlias: untyped{nkIdent}, exp: typed): untyped =
  ## Injects `expAlias` in caller scope, evaluating `exp` at most once.
  ## `expAlias` becomes a 0-arg template that yields the captured value.
  ##
  ## - Int[N] / int-literal / all-args-CT function call → `const` + `Int[val]()`
  ## - Runtime expression → `let` binding (evaluated once)
  ## - Plain symbol → reused directly
  expectKind(expAlias, nnkIdent)
  var val = exp
  result = newStmtList()

  if exp.kind == nnkSym:
    val = exp
  else:
    let typ = exp.getTypeInst()
    var isCT = typ.kind == nnkBracketExpr and $typ[0] == "Int"
    if not isCT and (exp.kind == nnkCall or exp.kind == nnkHiddenCallConv):
      var allArgsCT = true
      for i in 1 ..< exp.len:
        let arg = exp[i]
        if arg.kind != nnkIntLit:
          if arg.kind == nnkSym:
            try:
              let impl = arg.getImpl()
              if impl.kind != nnkConstSection:
                allArgsCT = false
            except:
              allArgsCT = false
          else:
            allArgsCT = false
      if allArgsCT and exp.len > 1:
        isCT = true
    if isCT or exp.kind == nnkIntLit:
      let tmp = ident("ctEval_" & $exp.repr & $exp.kind.repr)
      result.add nnkConstSection.newTree(
        nnkConstDef.newTree(tmp, newEmptyNode(), exp)
      )
      val = nnkObjConstr.newTree(
        nnkBracketExpr.newTree(bindSym"Int", tmp)
      )
    else:
      let tmp = ident("rtEval_" & $exp.repr & $exp.kind.repr)
      result.add nnkLetSection.newTree(
        nnkIdentDefs.newTree(tmp, newEmptyNode(), exp)
      )
      val = tmp

  result.add(
    newProc(name = genSym(nskTemplate, $expAlias), params = [getType(untyped)],
      body = val, procType = nnkTemplateDef))

# ═══════════════════════════════════════════════════════════════
#  scaleBy — element-wise tuple scaling preserving nesting
# ═══════════════════════════════════════════════════════════════

macro scaleBy*(t: typed; multiplier: typed): untyped =
  ## Multiply each leaf of a (possibly nested) tuple by scalar m.
  ## Preserves nesting structure.
  # `typed` required: type-class constraints (tuple, int or Int)
  # fail in generic functions where types aren't concrete yet.
  proc scaleImpl(nestedExpr, node, mul: NimNode): NimNode =
    if node.kind == nnkTupleConstr:
      result = newNimNode(nnkTupleConstr)
      for i in 0 ..< node.len:
        result.add scaleImpl(
          nnkBracketExpr.newTree(nestedExpr, newLit(i)), node[i], mul)
    else:
      result = newCall(bindSym"*", nestedExpr, mul)
  result = scaleImpl(t, t.getTypeImpl(), multiplier)

# ═══════════════════════════════════════════════════════════════
#  fold — left-fold reduction with Int[N] support
# ═══════════════════════════════════════════════════════════════
#
#  Adapted from the pattern in int_tuples.nim.bak:
#    - Scalar `int` → inject `acc`, `it`, evaluate body
#    - Scalar `Int[N]` → inject `acc`, `it` (Int[V] → int via * overloads)
#    - Tuple → recurse over fields via `for f in fields(t)`
#
#  Injects `acc` (accumulator, type int) and `it` (current element).
#  `body` returns new accumulator.  Always returns `int`.
#
#  Examples:
#    fold(5, 1, acc * it)            → 5
#    fold(Int[5](), 1, acc * it)     → 5  (Int[V] extracted via * overload)
#    fold((2,3,4), 1, acc * it)      → 24
#    fold((2,(3,4)), 1, acc * it)    → 24
# ═══════════════════════════════════════════════════════════════

template fold_recurse*(idx: static int; t: tuple; state: typed; body: untyped): auto =
  let field = fold(t[idx], state, body)
  when idx == tupleLen(t) - 1:
    field
  else:
    fold_recurse(idx + 1, t, field, body)

template fold*(t: IntOrIntTuple; startingAcc: typed; body: untyped): auto =
  ## Fold over all leaves of t with an accumulator.
  ## Sub-tuples are handled recursively via fold_recurse.
  ## Returns Int[N] for all-Int[N] leaf paths, int otherwise.
  when t is int or t is Int:
    block:
      let acc {.inject.} = startingAcc
      let it {.inject.} = t
      body
  else:  # tuple
    when tupleLen(t) == 0:
      startingAcc
    else:
      fold_recurse(0, t, startingAcc, body)

template foldZipWith_recurse*(idx: static int; a, b: tuple; state: typed; body: untyped): auto =
  static: doAssert tupleLen(a) == tupleLen(b), "foldZipWith: tuples must have same rank"
  const N = tupleLen(a)
  let field = foldZipWith(a[idx], b[idx], state, body)
  when idx == N - 1:
    field
  else:
    foldZipWith_recurse(idx + 1, a, b, field, body)

template foldZipWith*(a, b: typed; startingAcc: typed; body: untyped): auto =
  ## Fold over paired leaves of a and b.
  ## Injects `acc`, `it_a`, `it_b`.
  when a is tuple and b is tuple:
    foldZipWith_recurse(0, a, b, startingAcc, body)
  else:
    block:
      let acc {.inject.} = startingAcc
      let it_a {.inject.} = a
      let it_b {.inject.} = b
      body

# ═══════════════════════════════════════════════════════════════
#  prefix_scanIt / suffix_scanIt - scans while preserving constness
# ═══════════════════════════════════════════════════════════════
#
#  Recursive template block + concat for type-correct tuple building.
#  Injects `acc` (accumulator before element) and `it` (element).
#  Returns tuple where each element = `acc` BEFORE that element.
# ═══════════════════════════════════════════════════════════════

template prefix_scanIt_recurse*(idx: static int; t: tuple; state: typed; body: untyped): untyped =
  ## Recursive prefix scan. Each level injects acc/it into a block scope.
  ##
  ## Due to generic sandwich / template symbol resolution issues
  ## this is exported but it really is an internal module
  block:
    let it {.inject.} = t[idx]
    let acc {.inject.} = state
    let newState = body
    when idx == tupleLen(t) - 1:
      (acc,)
    else:
      concat((acc,), prefix_scanIt_recurse(idx + 1, t, newState, body))

template suffix_scanIt_recurse*(idx: static int; t: tuple; state: typed; body: untyped): untyped =
  ## Recursive suffix scan. Each level injects acc/it into a block scope.
  ##
  ## Due to generic sandwich / template symbol resolution issues
  ## this is exported but it really is an internal module
  block:
    let it {.inject.} = t[idx]
    let acc {.inject.} = state
    let newState = body
    when idx == 0:
      (acc,)
    else:
      concat(suffix_scanIt_recurse(idx - 1, t, newState, body), (acc,))

template prefix_scanIt*(t: untyped; startingAcc: auto; body: untyped): untyped =
  ## Left-to-right prefix scan. Injects `acc`, `it`; body → new accumulator.
  when t is int or t is Int:
    startingAcc
  else:
    prefix_scanIt_recurse(0, t, startingAcc, body)

template suffix_scanIt*(t: untyped; startingAcc: auto; body: untyped): untyped =
  ## Right-to-left suffix scan. Injects `acc`, `it`; body → new accumulator.
  when t is int or t is Int:
    startingAcc
  else:
    suffix_scanIt_recurse(tupleLen(t) - 1, t, startingAcc, body)

# ═══════════════════════════════════════════════════════════════
#  makeIntTuple — wrap static ints / Int literals in Int[N]
# ═══════════════════════════════════════════════════════════════

#  Leaf procs — dispatch on exact type

func makeIntTupleLeaf(leaf: int): int = leaf
  ## Dynamic int scalar — passthrough.

func makeIntTupleLeaf(leaf: static int): auto = Int[leaf]()
  ## Static int literal — wrap in Int[N].

func makeIntTupleLeaf[V: static int](x: Int[V]): Int[V] = x
  ## Already Int[N] — passthrough.

#  Compile-time type helpers for the recursive macro

func isIntType(x: NimNode): bool {.compileTime.} =
  ## True if `x` is typed as plain `int`.
  sameType(x, bindSym"int")

func isTupleType(x: NimNode): bool {.compileTime.} =
  ## True if `x` is typed as a tuple type.
  x.getTypeImpl().kind == nnkTupleConstr

func isStaticIntType(x: NimNode): bool {.compileTime.} =
  ## True if `x` is typed as `Int[N]`.
  let t = x.getTypeInst()
  t.kind == nnkBracketExpr and $t[0] == "Int"

#  Recursive macro — wraps const ints in Int[N] throughout tuples

macro makeIntTupleRec*(a: IntOrIntTuple): untyped =
  ## Recursively wrap static ints / Int literals in Int[N].
  ## - `int` literal/const → `Int[val]()`  (via makeIntTupleLeaf's static overload)
  ## - `int` runtime → passthrough (via makeIntTupleLeaf's int overload)
  ## - `Int[N]` → passthrough
  ## - tuple → recursively process each field
  if a.isIntType():
    # Delegate to makeIntTupleLeaf which has both `int` and `static int` overloads.
    # The compiler picks the right one — no manual CT-detection needed.
    result = newCall(bindSym"makeIntTupleLeaf", a)
  elif a.isTupleType():
    if a.kind == nnkTupleConstr:
      # Literal tuple: iterate children directly (preserves static types)
      result = newNimNode(nnkTupleConstr)
      for child in a:
        result.add newCall(bindSym"makeIntTupleRec", child)
    else:
      # Variable/function tuple: use bracket access
      let ttype = a.getTypeImpl()
      let tup = newNimNode(nnkTupleConstr)
      for i in 0 ..< ttype.len:
        tup.add newCall(bindSym"makeIntTupleRec", newTree(nnkBracketExpr, a, newLit(i)))
      result = tup
  elif a.isStaticIntType():
    result = a
  else:
    let msg = "[makeIntTupleRec] invalid type: " & a.repr & " got " & a.getTypeInst().repr
    error msg

template makeIntTuple(a: IntOrIntTuple): untyped =
  ## Public face: wraps static ints in Int[N] via the recursive macro.
  makeIntTupleRec(a)


template prefix_product*(shape: IntOrIntTuple): untyped =
  ## Cumulative left-to-right product scan.
  ##
  ## Each output element is the product of all input elements
  ## **before** that position. This is a **scan** (not a reduce),
  ## so the first output is always 1 (the multiplicative identity
  ## over zero elements).
  ##
  ## Examples:
  ##   prefix_product(5)        → 1         (scalar)
  ##   prefix_product((a,))     → (1,)
  ##   prefix_product((a, b))   → (1, a)
  ##   prefix_product((2,3,4))  → (1, 2, 6)
  ##   prefix_product((2,3,4))  === (1, 2, 6)
  ##
  ## Identical to CuTe's prefix_product / make_seq.
  ##
  ## See also: suffix_product, prefix_scanIt
  prefix_scanIt(shape, Int[1](), acc * it)

template suffix_product*(shape: IntOrIntTuple): untyped =
  ## Cumulative right-to-left product scan.
  ##
  ## Right-to-left variant of prefix_product:
  ##   suffix_product((a,b,c))  → (b*c, c, 1)
  ##   suffix_product((2,3,4))  → (12, 4, 1)
  ##
  ## The last output is always 1 (identity over zero elements).
  ##
  ## See also: prefix_product, suffix_scanIt
  suffix_scanIt(shape, Int[1](), acc * it)
# ═══════════════════════════════════════════════════════════════
#  Reductions
# ═══════════════════════════════════════════════════════════════

func product*(t: IntOrIntTuple): auto =
  ## Product of all leaves (like size for shapes). Returns Int[N] for all-Int, int otherwise.
  fold(t, Int[1](), acc * it)

func max*(t: IntOrIntTuple): auto =
  ## Maximum leaf value. Returns Int[N] for all-Int, int otherwise.
  fold(t, Int[low(int)](), max(acc, it))

func min*(t: IntOrIntTuple): auto =
  ## Minimum leaf value. Returns Int[N] for all-Int, int otherwise.
  fold(t, Int[high(int)](), min(acc, it))

# ═══════════════════════════════════════════════════════════════
#  flatten — recursively collect leaf elements of a tuple
# ═══════════════════════════════════════════════════════════════

macro flattenImpl(t: IntOrIntTuple): untyped =
  let tNode = t
  let ttype = tNode.getTypeImpl()

  proc isLeaf(t: NimNode): bool =
    (t.kind == nnkSym and $t == "int") or
    (t.kind == nnkBracketExpr and $t[0] == "Int")

  proc collect(acc: var NimNode; e: NimNode; t: NimNode) =
    if t.kind == nnkTupleConstr:
      for idx in 0 ..< t.len:
        let fd = t[idx]
        let fa = newTree(nnkBracketExpr, e, newLit(idx))
        if isLeaf(fd):
          acc.add fa
        else:
          collect(acc, fa, fd)
    else:
      acc.add e

  # nnkPar: single-item result stays scalar (avoids explicit `if result.len == 1`).
  # Multi-item: construct a tuple like nnkTupleConstr.
  result = newNimNode(nnkPar)
  collect(result, tNode, ttype)

proc flatten*(t: IntOrIntTuple): auto =
  ## Recursively collect leaf fields of a (possibly nested) tuple.
  ## Scalars and Int[N] are leaves; tuples are expanded.
  ##
  ## CuTe: `flatten(t)` → flat tuple of leaf elements.
  ##
  ## Examples:
  ##   flatten(5)          → 5
  ##   flatten((1,2,3))    → (1,2,3)
  ##   flatten((1,(2,3)))  → (1,2,3)
  flattenImpl(t)

proc flatten*(t: static IntOrIntTuple): static auto =
  ## Recursively collect leaf fields of a (possibly nested) tuple.
  ## Scalars and Int[N] are leaves; tuples are expanded.
  ##
  ## CuTe: `flatten(t)` → flat tuple of leaf elements.
  ##
  ## Examples:
  ##   flatten(5)          → 5
  ##   flatten((1,2,3))    → (1,2,3)
  ##   flatten((1,(2,3)))  → (1,2,3)
  ##
  ## Compile-time overload for full-compile-time input
  flattenImpl(t)

# ═══════════════════════════════════════════════════════════════
#  concat — concatenate two tuples or a scalar and a tuple
# ═══════════════════════════════════════════════════════════════
#
#  CuTe: `append(t, x)` / `prepend(t, x)` — concat is the unified version.
#  Replaces both append and prepend via overloads.
#
#  Overloads:
#    concat(a: int;        b: tuple)     — int + tuple
#    concat(a: static int; b: tuple)     — static int + tuple  (→ Int[N] concatenation)
#    concat(a: Int[N];     b: tuple)     — Int[N] + tuple
#    concat(a: tuple;      b: int)       — tuple + int
#    concat(a: tuple;      b: static int) — tuple + static int
#    concat(a: tuple;      b: Int[N])    — tuple + Int[N]
#    concat(a, b: tuple)                  — tuple + tuple
#    concat(a, b: int)                    — int + int  (→ tuple)
#    concat(a: int; b: static int)        — int + static int
#    concat(a: static int; b: int)        — static int + int
#    concat(a, b: static int)             — static int + static int
#    concat(a: Int[N]; b: Int[N])         — Int[N] + Int[N]
# ═══════════════════════════════════════════════════════════════

proc concat*(a: int; b: tuple): auto =
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    let bType = bNode.getTypeImpl()
    result = newNimNode(nnkTupleConstr)
    result.add aNode
    for idx in 0 ..< bType.len:
      result.add newTree(nnkBracketExpr, bNode, newLit(idx))
  concatImpl()

proc concat*(a: static int; b: tuple): auto =
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    let bType = bNode.getTypeImpl()
    result = newNimNode(nnkTupleConstr)
    result.add newNimNode(nnkObjConstr).add(
      newNimNode(nnkBracketExpr).add(ident"Int", aNode))
    for idx in 0 ..< bType.len:
      result.add newTree(nnkBracketExpr, bNode, newLit(idx))
  concatImpl()

proc concat*[V: static int](a: Int[V]; b: tuple): auto =
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    let bType = bNode.getTypeImpl()
    result = newNimNode(nnkTupleConstr)
    result.add aNode
    for idx in 0 ..< bType.len:
      result.add newTree(nnkBracketExpr, bNode, newLit(idx))
  concatImpl()

proc concat*(a: tuple; b: int): auto =
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    let aType = aNode.getTypeImpl()
    result = newNimNode(nnkTupleConstr)
    for idx in 0 ..< aType.len:
      result.add newTree(nnkBracketExpr, aNode, newLit(idx))
    result.add bNode
  concatImpl()

proc concat*(a: tuple; b: static int): auto =
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    let aType = aNode.getTypeImpl()
    result = newNimNode(nnkTupleConstr)
    for idx in 0 ..< aType.len:
      result.add newTree(nnkBracketExpr, aNode, newLit(idx))
    result.add newNimNode(nnkObjConstr).add(
      newNimNode(nnkBracketExpr).add(ident"Int", bNode))
  concatImpl()

proc concat*[V: static int](a: tuple; b: Int[V]): auto =
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    let aType = aNode.getTypeImpl()
    result = newNimNode(nnkTupleConstr)
    for idx in 0 ..< aType.len:
      result.add newTree(nnkBracketExpr, aNode, newLit(idx))
    result.add bNode
  concatImpl()

proc concat*(a, b: tuple): auto =
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    let aType = aNode.getTypeImpl(); let bType = bNode.getTypeImpl()
    result = newNimNode(nnkTupleConstr)
    for idx in 0 ..< aType.len:
      result.add newTree(nnkBracketExpr, aNode, newLit(idx))
    for idx in 0 ..< bType.len:
      result.add newTree(nnkBracketExpr, bNode, newLit(idx))
  concatImpl()

proc concat*(a, b: int): auto =
  ## int + int → (int, int) tuple
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    result = newNimNode(nnkTupleConstr)
    result.add aNode
    result.add bNode
  concatImpl()

proc concat*(a: int; b: static int): auto =
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    result = newNimNode(nnkTupleConstr)
    result.add aNode
    result.add newNimNode(nnkObjConstr).add(
      newNimNode(nnkBracketExpr).add(ident"Int", bNode))
  concatImpl()

proc concat*(a: static int; b: int): auto =
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    result = newNimNode(nnkTupleConstr)
    result.add newNimNode(nnkObjConstr).add(
      newNimNode(nnkBracketExpr).add(ident"Int", aNode))
    result.add bNode
  concatImpl()

proc concat*(a, b: static int): auto =
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    result = newNimNode(nnkTupleConstr)
    result.add newNimNode(nnkObjConstr).add(
      newNimNode(nnkBracketExpr).add(ident"Int", aNode))
    result.add newNimNode(nnkObjConstr).add(
      newNimNode(nnkBracketExpr).add(ident"Int", bNode))
  concatImpl()

proc concat*[V1, V2: static int](a: Int[V1]; b: Int[V2]): static auto =
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    result = newNimNode(nnkTupleConstr)
    result.add aNode
    result.add bNode
  concatImpl()

proc concat*[V: static int](a: Int[V] or int; b: int or Int[V]): auto =
  ## Int[N] + int tuple
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    result = newNimNode(nnkTupleConstr)
    result.add aNode
    result.add bNode
  concatImpl()

proc concat*[V: static int](a: Int[V]; b: static int): static auto =
  ## Int[N] + static int tuple (static int → Int[b])
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    result = newNimNode(nnkTupleConstr)
    result.add aNode
    result.add newNimNode(nnkObjConstr).add(
      newNimNode(nnkBracketExpr).add(ident"Int", bNode))
  concatImpl()

proc concat*[V: static int](a: static int; b: Int[V]): static auto =
  ## static int + Int[N] tuple (static int → Int[a])
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    result = newNimNode(nnkTupleConstr)
    result.add newNimNode(nnkObjConstr).add(
      newNimNode(nnkBracketExpr).add(ident"Int", aNode))
    result.add bNode
  concatImpl()

# ═══════════════════════════════════════════════════════════════
#  zip2_by — guided zip for rank-2 tuples
# ═══════════════════════════════════════════════════════════════

func zip2_by*(t: tuple; guide: int): auto =
  ## CuTe: zip2_by(t, guide) — tuple_algorithms.hpp
  ## Terminal guide: t must be a pair, returned as-is.
  t
func zip2_by*[V: static int](t: tuple; guide: Int[V]): auto =
  ## Terminal Int[N] guide.
  t
func zip2_by*(t: tuple; guide: auto): auto =
  ## Terminal guide fallback (Layout, etc.).
  t
func zip2_by*[T: tuple, G: tuple](t: T; guide: G): auto =
  ## Guided zip: split flat tuple `t` into (first_parts, second_parts).
  ##
  ## For each i where guide[i] is a tuple: recurse zip2_by(t[i], guide[i]).
  ## For each i where guide[i] is scalar: t[i] must be a pair; its two
  ##   elements go to first_parts and second_parts respectively.
  ## Extra elements of t (beyond guide length) append to second_parts.
  ##
  ## Result: (first_parts_tuple, second_parts_tuple) — rank-2.
  ##
  ## CuTe: zip2_by(t, guide) — tuple_algorithms.hpp line ~739
  ## Used by tile_unzip → zipped_divide/zipped_product.
  macro impl: untyped =
    let tNode = bindSym"t"
    let guideNode = bindSym"guide"
    let guideLen = guideNode.getTypeInst().len
    let tLen = tNode.getTypeInst().len
    var firstParts = newNimNode(nnkTupleConstr)
    var secondParts = newNimNode(nnkTupleConstr)
    # Guided elements: each produces (first_elem, second_elem)
    for i in 0 ..< guideLen:
      let ti = nnkBracketExpr.newTree(tNode, newLit(i))
      let gi = nnkBracketExpr.newTree(guideNode, newLit(i))
      let splitPair = newCall(bindSym"zip2_by", ti, gi)
      firstParts.add nnkBracketExpr.newTree(splitPair, newLit(0))
      secondParts.add nnkBracketExpr.newTree(splitPair, newLit(1))
    # Unguided tail: append to second parts
    for i in guideLen ..< tLen:
      secondParts.add nnkBracketExpr.newTree(tNode, newLit(i))
    result = nnkTupleConstr.newTree(firstParts, secondParts)
  impl()

# ═══════════════════════════════════════════════════════════════
#  map — apply fn to each tuple element
#  zipWith — zip two tuples element-wise, append leftovers
# ═══════════════════════════════════════════════════════════════

macro mapModesWith*[T: IntOrIntTuple](t: T; body: untyped): untyped =
  ## Apply `body` to each top-level element of tuple `t` (does NOT recurse into nested tuples).
  ## `it` binds to the current element.
  ##
  ## Example:
  ##   map((2, 4, 6)): it * 2  →  (4, 8, 12)
  let tt = getTypeInst(t)
  let n = if tt.kind == nnkTupleConstr: tt.len else: 1

  proc subst(x: NimNode; i: int; ttup: NimNode): NimNode =
    if x.kind in {nnkIdent, nnkSym} and x.eqIdent("it"):
      result = nnkBracketExpr.newTree(ttup, newLit(i))
    else:
      result = x.copyNimTree()
      for j in 0 ..< x.len:
        result[j] = subst(x[j], i, ttup)

  var items: seq[NimNode]
  for i in 0 ..< n:
    items.add subst(body, i, t)
  # nnkPar: single-item result stays scalar (avoids explicit `if result.len == 1`).
  # Multi-item: construct a tuple like nnkTupleConstr.
  result = nnkPar.newTree(items)

macro zipModesWith*[A, B: IntOrIntTuple](a: A; b: B; body: untyped): untyped =
  ## Zip top-level elements of tuples `a` and `b` pairwise via `body` (does NOT recurse into nested tuples).
  ## `it_a` / `it_b` bind to corresponding elements.
  ## Leftover elements from the longer tuple are appended unchanged.
  ##
  ## Example:
  ##   zipWith((2, 4), (10, 20)): it_a + it_b  →  (12, 24)
  ##   zipWith((2, 4, 6), (10, 20)): it_a + it_b  →  (12, 24, 6)
  let ta = getTypeInst(a)
  let tb = getTypeInst(b)
  let RA = if ta.kind == nnkTupleConstr: ta.len else: 1
  let RB = if tb.kind == nnkTupleConstr: tb.len else: 1
  let rMin = min(RA, RB)
  let rMax = max(RA, RB)

  proc subst(x: NimNode; i: int; la, lb: NimNode): NimNode =
    if x.kind in {nnkIdent, nnkSym} and x.eqIdent("it_a"):
      result = nnkBracketExpr.newTree(la, newLit(i))
    elif x.kind in {nnkIdent, nnkSym} and x.eqIdent("it_b"):
      result = nnkBracketExpr.newTree(lb, newLit(i))
    else:
      result = x.copyNimTree()
      for j in 0 ..< x.len:
        result[j] = subst(x[j], i, la, lb)

  result = newStmtList()
  var items: seq[NimNode]
  for i in 0 ..< rMax:
    let name = ident("__zw" & $i)
    if i < rMin:
      items.add name
      result.add newLetStmt(name, subst(body, i, a, b))
    elif i < RA:
      items.add nnkBracketExpr.newTree(a, newLit(i))
    else:
      items.add nnkBracketExpr.newTree(b, newLit(i))
  # nnkPar: single-item result stays scalar (avoids explicit `if result.len == 1`).
  # Multi-item: construct a tuple like nnkTupleConstr.
  result.add nnkPar.newTree(items)

# ═══════════════════════════════════════════════════════════════
#  product_each — product of each top-level tuple element
# ═══════════════════════════════════════════════════════════════

func product_each*(t: IntOrIntTuple): auto =
  ## For each top-level element of tuple `t`, compute the product of its leaves.
  ##
  ## Examples:
  ##   product_each(((2,2), (2,8)))  →  (4, 16)
  ##   product_each(5)                →  5
  ##
  mapModesWith(t): product(it)

# ═══════════════════════════════════════════════════════════════
#  Joker — CuTe Underscore: keep/slice marker for coordinates
# ═══════════════════════════════════════════════════════════════

type Joker* = object

type CoordType* = int | Int | Joker | tuple
  ## Marker type for "keep this dimension" in slice/dice.
  ## Analogous to CuTe's `Underscore` / `_`.
  ## Use `_` as marker in macro/template context.

template `_`*: Joker = Joker()

func `$`*(x: Joker): string = "_"
func crd2idx*(coord: Joker; shape: int): int = 0
  ## Joker contributes 0 to indexing.
func crd2idx*[V: static int](coord: Joker; shape: Int[V]): int = 0
func crd2idx*(coord: Joker; shape, stride: int): int = 0
func crd2idx*[V: static int](coord: Joker; shape, stride: Int[V]): int = 0
func `*`*(c: Joker; s: int): int = 0
func `*`*(s: int; c: Joker): int = 0
func `*`*[V: static int](c: Joker; s: Int[V]): int = 0
func `*`*[V: static int](s: Int[V]; c: Joker): int = 0
func `+`*(c: Joker; s: int): int = s
func `+`*(s: int; c: Joker): int = s
func `+`*[V: static int](c: Joker; s: Int[V]): int = V
func `+`*[V: static int](s: Int[V]; c: Joker): int = V


func isJokerNode*(n: NimNode): bool {.compileTime.} =
  ## True if NimNode represents a Joker value.
  let t = n.getTypeInst()
  t.kind == nnkSym and $t == "Joker"

# ═══════════════════════════════════════════════════════════════
#  slice(coord, target) — keep elements paired with Joker
#  dice(coord, target)  — keep elements paired with int
# ═══════════════════════════════════════════════════════════════
##
## Both are compile-time tuple filtering operations.
## CuTe C++: underscore.hpp
##
## slice(_, b)           → b        (bare scalar, joker on scalar)
## slice(0, b)           → ()       (empty tuple)
## slice((_, 0), (a,b))  → (a,)     (keep a, drop b)
## dice(_, b)            → ()       (empty tuple)
## dice(0, b)            → b        (bare scalar, int on scalar)
## dice((0, _), (a,b))   → a        (keep a, drop b)

macro slice*(coord: CoordType; target: IntOrIntTuple): untyped =
  ## CuTe-compatible slice: keep elements of target paired with joker/`_`.
  ## Returns a tuple of kept elements (or bare element for scalar joker case).
  ##
  ## Type-constrained: target must be int, Int[N], or tuples thereof.
  runnableExamples:
    let r = slice((_, 0), (3, 4))
    doAssert r[0] == 3

  # Replace `_` identifiers with Joker() so `_` syntax works
  proc clense(n: NimNode): NimNode =
    if n.kind == nnkIdent and n.eqIdent("_"):
      result = newCall(bindSym"Joker")
    else:
      result = n.copyNimTree()
      for i in 0 ..< n.len:
        result[i] = clense(n[i])
  let c = clense(coord)
  let t = target

  # Collect all (coord_leaf, target_leaf_index_path) pairs
  # where target_leaf_index_path is the sequence of indices needed to access it
  proc collectLeaves(cNode: NimNode; path: seq[int]): seq[(NimNode, seq[int])] =
    if cNode.kind == nnkTupleConstr:
      for i in 0 ..< cNode.len:
        for pair in collectLeaves(cNode[i], path & i):
          result.add pair
    else:
      result.add (cNode, path)

  let leaves = collectLeaves(c, @[])

  # Build a flat tuple: for each joker leaf, add the target element at the path
  var parts: seq[NimNode] = @[]
  for (coordLeaf, path) in leaves:
    if isJokerNode(coordLeaf):
      # Keep: construct target path access like t[i][j]...
      var access = t
      for idx in path:
        access = newCall(bindSym"[]", access, newLit(idx))
      parts.add access
    # else: int -> drop (don't add)

  if parts.len == 0:
    result = nnkPar.newTree()  # empty tuple ()
  elif parts.len == 1 and c.kind != nnkTupleConstr:
    # Bare joker on scalar: return bare
    result = parts[0]
  else:
    # Tuple coord -> always tuple result
    result = nnkTupleConstr.newTree(parts)

macro dice*(coord: CoordType; target: IntOrIntTuple): untyped =
  ## CuTe-compatible dice: keep elements of target paired with ints.
  ## Returns target element (for scalar int coord) or tuple of kept elements.
  ##
  ## Type-constrained: target must be int, Int[N], or tuples thereof.
  runnableExamples:
    let r = dice((_, 0), (3, 4))
    doAssert r == 4

  # Replace `_` identifiers with Joker()
  proc clense(n: NimNode): NimNode =
    if n.kind == nnkIdent and n.eqIdent("_"):
      result = newCall(bindSym"Joker")
    else:
      result = n.copyNimTree()
      for i in 0 ..< n.len:
        result[i] = clense(n[i])
  let c = clense(coord)
  let t = target
  # Validate: coord must contain Joker, int, Int[N], or tuples thereof
  # target must contain int, Int[N], or tuples thereof
  # (Implicitly checked by the fact we only generate valid accesses)

  # Collect all (coord_leaf, target_leaf_index_path) pairs
  proc collectLeaves(cNode: NimNode; path: seq[int]): seq[(NimNode, seq[int])] =
    if cNode.kind == nnkTupleConstr:
      for i in 0 ..< cNode.len:
        for pair in collectLeaves(cNode[i], path & i):
          result.add pair
    else:
      result.add (cNode, path)

  let leaves = collectLeaves(c, @[])

  # Build result: for each int leaf, add target element; for joker, drop
  var parts: seq[NimNode] = @[]
  for (coordLeaf, path) in leaves:
    if not isJokerNode(coordLeaf):
      # Keep: construct target path access
      var access = t
      for idx in path:
        access = newCall(bindSym"[]", access, newLit(idx))
      parts.add access

  if parts.len == 0:
    result = nnkPar.newTree()  # empty
  elif parts.len == 1 and c.kind != nnkTupleConstr:
    # Bare int/joker on scalar: return bare
    result = parts[0]
  else:
    # Tuple coord -> always tuple result
    result = nnkTupleConstr.newTree(parts)

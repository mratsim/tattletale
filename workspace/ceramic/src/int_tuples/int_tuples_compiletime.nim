# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import workspace/math
import ./int_tuples_datatypes

const DynamicSentinel* = low(int)
  ## Sentinel value used throughout the library to mark "unknown at compile time" (dynamic/runtime int).
  ## 0-stride instead can be confused with a broadcasted dimension.

# ═══════════════════════════════════════════════════════════════
#  isConst, compile-time detection with a runtime dispatch
# ═══════════════════════════════════════════════════════════════

template isConst*(a: static int): bool = true
template isConst*(a: int): bool = false
template isConst*[V: static int](a: Int[V]): bool = true

template isConst*(a: static tuple): bool = true
template isConst*(a: tuple): bool = false

# ═══════════════════════════════════════════════════════════════
#  Int[N] compile-time helpers for macros
# ═══════════════════════════════════════════════════════════════

func IntCT*(val: int): NimNode {.compileTime.} =
  ## Shorthand: Int[val]() AST node.
  newNimNode(nnkObjConstr).add(newNimNode(nnkBracketExpr).add(ident"Int", newLit(val)))

func intTy*(val: int): NimNode {.compileTime.} =
  ## Int[val] bracket for the compile-time value `val`
  IntCT(val)[0]

func divmod*(a, b: int): (int, int) {.compileTime.} =
  ## Quotient and remainder of `a` divided by `b`.
  (a div b, a mod b)

func isStaticInt*(t: NimNode): bool {.compileTime.} =
  (t.kind == nnkBracketExpr and $t[0] == "Int") or t.kind == nnkIntLit

func isStaticOne*(t: NimNode): bool {.compileTime.} =
  (t.kind == nnkBracketExpr and $t[0] == "Int" and t[1].intVal == 1) or
  (t.kind == nnkIntLit and t.intVal == 1)

func getStaticInt*(t: NimNode): int {.compileTime.} =
  ## Static Int value of a node.
  ## - the result is DynamicSentinel when the node lacks a static Int:
  ##   not a literal, not an `Int[V]` type or construction
  ##
  ## TODO: duplicating Nim constant-folding via Int overload is unwanted.
  case t.kind
  of nnkIntLit, nnkUIntLit:
    int(t.intVal)
  of nnkStmtListExpr:
    # evalOnceAs caching, `const q = X; X`, the trailing value is static
    t[^1].getStaticInt()
  of nnkCall, nnkBracketExpr:
    # typeDesc[Int[V]] wrappers: the typed-delegate path produces these
    if t[0].eqIdent("typeDesc"):
      t[1].getStaticInt()
    elif t[0].eqIdent("Int"):
      t[1].getStaticInt()
    elif t[0].eqIdent("ceil_div") and t.len == 3:
      # A `ceil_div` call over two static operands folds to its value.
      # Nim applies the same folding to static ints during type resolution.
      let a = t[1].getStaticInt()
      let b = t[2].getStaticInt()
      if a == DynamicSentinel or b == DynamicSentinel or b == 0:
        DynamicSentinel
      else:
        math.ceil_div(a, b)
    elif t[0].eqIdent("round_multiple_up") and t.len == 3 and
        t[1].kind in {nnkIntLit, nnkUIntLit} and t[2].kind in {nnkIntLit, nnkUIntLit}:
      let a = t[1].intVal.int
      let b = t[2].intVal.int
      if b == 0:
        DynamicSentinel
      else:
        math.round_multiple_up(a, b)
    elif t[0].eqIdent("round_multiple_down") and t.len == 3 and
        t[1].kind in {nnkIntLit, nnkUIntLit} and t[2].kind in {nnkIntLit, nnkUIntLit}:
      let a = t[1].intVal.int
      let b = t[2].intVal.int
      if b == 0:
        DynamicSentinel
      else:
        math.round_multiple_down(a, b)
    else:
      DynamicSentinel
  of nnkInfix:
    # Algebra emission produces natural arithmetic such as `Int[a] div Int[b]`,
    # folded here one operator per step, before Nim semchecks the expression.
    let a = t[1].getStaticInt()
    let b = t[2].getStaticInt()
    if a == DynamicSentinel or b == DynamicSentinel:
      return DynamicSentinel
    case $t[0]
    of "+":
      a + b
    of "-":
      a - b
    of "*":
      a * b
    of "div":
      if b != 0:
        a div b
      else:
        DynamicSentinel
    of "mod":
      if b != 0:
        a mod b
      else:
        DynamicSentinel
    else:
      DynamicSentinel
  of nnkObjConstr:
    if t[0].kind == nnkBracketExpr and t[0][0].eqIdent("Int"):
      t[0].getStaticInt()
    else:
      DynamicSentinel
  of nnkSym:
    let ti = t.getTypeInst()
    if ti.kind != nnkSym:
      ti.getStaticInt()
    else:
      DynamicSentinel
  else:
    DynamicSentinel

# ═══════════════════════════════════════════════════════════════
#  AST syntax sugar
# ═══════════════════════════════════════════════════════════════

func tupleTypeWithoutNames(t: NimNode): NimNode {.compileTime.} =
  # Keep only the types from a named tuple like  `tuple[a: int, b: int]`
  if t.kind != nnkTupleTy:
    return t
  var fields: seq[NimNode] = @[]
  for f in t:
    if f.kind == nnkIdentDefs:
      # a grouped `tuple[a, b: int]` IdentDefs carries several
      # identifiers and one shared type, the type appends once per identifier
      for _ in 0 ..< f.len - 2:
        fields.add f[^2].tupleTypeWithoutNames()
    else:
      fields.add f
  result = nnkTupleTy.newTree(fields)

func getTypeOf*(n: NimNode): NimNode {.compileTime.} =
  case n.kind
  of nnkStmtListExpr:
    result = getTypeOf(n[^1])
  of nnkCall, nnkObjConstr:
    if n[0].kind == nnkBracketExpr:
      result = n[0]
    elif n[0].eqIdent("E"):
      # a CoordStride atom constructor, the type comes from the arguments
      var coords = nnkPar.newTree()
      for i in 1 ..< n.len:
        coords.add n[i]
      result = nnkBracketExpr.newTree(ident"CoordStride", coords)
    else:
      result = n.getTypeInst()
  of nnkSym:
    result = n.getTypeInst()
  of nnkIdent, nnkIntLit .. nnkUInt64Lit, nnkFloatLit:
    result = nil
  of nnkPar, nnkTupleConstr:
    if n.kind == nnkPar and n.len == 1:
      result = getTypeOf(n[0])
    else:
      result = nnkTupleTy.newTree()
      for child in n:
        result.add getTypeOf(child)
  of nnkBracketExpr:
    let base = getTypeOf(n[0])
    if base != nil and base.kind in {nnkTupleTy, nnkTupleConstr} and
        n[1].kind in {nnkIntLit .. nnkUInt64Lit}:
      result = base[int(n[1].intVal)]
    else:
      result = n.getTypeInst()
  else:
    result = n.getTypeInst()

func getTupleType*(n: NimNode): NimNode {.compileTime.} =
  if n.kind == nnkTupleTy:
    return n.tupleTypeWithoutNames()
  let t = n.getType()
  let inner =
    if t.kind == nnkBracketExpr and t[0].eqIdent("typeDesc"):
      t[1]
    else:
      t
  result = inner.getTypeImpl().tupleTypeWithoutNames()

func isTupleTy*(t: NimNode): bool {.compileTime.} =
  if t == nil:
    return false
  case t.kind
  of nnkTupleTy, nnkTupleConstr:
    true
  of nnkPar:
    t.len != 1 # A single elem in parenthesis is not a tuple
  of nnkCall, nnkObjConstr:
    if t[0].kind == nnkBracketExpr:
      # an Int[v]() / Layout[A, B](...) constructor is a scalar value
      false
    else:
      t.getTupleType().kind in {nnkTupleConstr, nnkTupleTy}
  of nnkIntLit, nnkInt64Lit, nnkFloatLit, nnkCharLit, nnkStrLit,
     nnkNilLit, nnkEmpty:
    # fresh macro nodes, no type to resolve and never tuples
    false
  of nnkIdentDefs:
    # a typed declaration node is a scalar, never a tuple
    false
  of nnkBracketExpr:
    # a bracket is a tuple type only in its `tuple[...]` form, index expressions are scalars
    t.len > 0 and t[0].kind in {nnkSym, nnkIdent} and t[0].eqIdent("tuple")
  else:
    t.getTupleType().kind in {nnkTupleConstr, nnkTupleTy}

func isIntTyHead*(n: NimNode): bool {.compileTime.} =
  n.kind == nnkSym and (n.sameType(bindSym"Int") or $n == "Int")

func isScalarTy*(t: NimNode): bool {.compileTime.} =
  t.kind == nnkIntLit or
    (t.kind == nnkBracketExpr and t[0].isIntTyHead()) or
    (t.kind == nnkCall and t[0].kind == nnkBracketExpr and t[0][0].isIntTyHead()) or
    (t.kind == nnkSym and t.sameType(bindSym"int"))

func `*`*(a, b: NimNode): NimNode {.compileTime.} =
  nnkInfix.newTree(ident"*", a, b)

func `div`*(a, b: NimNode): NimNode {.compileTime.} =
  nnkInfix.newTree(ident"div", a, b)

func `+`*(a, b: NimNode): NimNode {.compileTime.} =
  nnkInfix.newTree(ident"+", a, b)

func `-`*(a, b: NimNode): NimNode {.compileTime.} =
  nnkInfix.newTree(ident"-", a, b)

func `mod`*(a, b: NimNode): NimNode {.compileTime.} =
  nnkInfix.newTree(ident"mod", a, b)

func abs*(a: NimNode): NimNode {.compileTime.} =
  bindSym"abs".newCall(a)

func min*(a, b: NimNode): NimNode {.compileTime.} =
  bindSym"min".newCall(a, b)

func max*(a, b: NimNode): NimNode {.compileTime.} =
  bindSym"max".newCall(a, b)

func ceil_div*(a, b: NimNode): NimNode {.compileTime.} =
  bindSym"ceil_div".newCall(a, b)

func round_multiple_up*(a, b: NimNode): NimNode {.compileTime.} =
  bindSym"round_multiple_up".newCall(a, b)

func sign*(a: NimNode): NimNode {.compileTime.} =
  bindSym"sign".newCall(a)

proc newLetAsgn*(stmts: var NimNode, name: string, value: NimNode): NimNode {.compileTime.} =
  result = genSym(nskLet, name)
  stmts.add result.newLetStmt value

# ═══════════════════════════════════════════════════════════════
#  evalOnceAs
# ═══════════════════════════════════════════════════════════════

macro evalOnceAs*(alias: untyped{nkIdent}, expression: typed{lvalue|lit|`let`|`const`|`var`}): untyped =
  ## Create an `alias` for `expression`
  ## - Returns: a template expanding to `expression`, evaluated once
  ##  ⚠  Every use inside a template body goes inside a `block:` scope.
  ##  ⚠  Standalone statements in a func body need no such scope.

  result = newProc(
    name = genSym(nskTemplate, $alias),
    params = [getType(untyped)],
    body = expression,
    procType = nnkTemplateDef
  )

macro evalOnceAs*[V: static int](alias: untyped{nkIdent}, expression: Int[V]): untyped =
  ## Create an `alias` for `expression`
  ## - Returns: a template expanding to `expression`, evaluated once
  ##  ⚠  Every use inside a template body goes inside a `block:` scope.
  ##  ⚠  Standalone statements in a func body need no such scope.
  ##     See the section comment at the top of this file.

  result = newStmtList()
  let evalOnceCT_staticInt = genSym(nskConst, "evalOnceCT_staticInt")

  # The expression may be a `let` binding which would lead to "cannot evaluate at compile-time"
  # So we rebuild a constant from the type.
  # As a side-benefit, the C++ compiler should dead-code eliminate the unused `let` expression.
  result.add newConstStmt(evalOnceCT_staticInt, IntCT(V))
  result.add newProc(
    name = genSym(nskTemplate, $alias),
    params = [getType(untyped)],
    body = evalOnceCT_staticInt,
    procType = nnkTemplateDef
  )

macro evalOnceAs*(alias: untyped{nkIdent}, expression: typed): untyped =
  ## Create an `alias` for `expression`
  ## - Returns: a template expanding to `expression`, evaluated once
  ##  ⚠  Every use inside a template body goes inside a `block:` scope.
  ##  ⚠  Standalone statements in a func body need no such scope.
  ##     See the section comment at the top of this file.

  # Uses a generated `when expression is static:` to choose
  # between `const` (compile-time) and `let` (runtime) storage.
  # The template name is genSym'd to avoid collisions when multiple
  # evalOnceAs calls exist in the same scope.
  #
  # Implementation notes, alternatives rejected:
  # - when compiles(static(expr)): crashes the nimvm in a `static:` context
  # - when compiles(const = expression): crashes in another context
  # - when is static: works and is what we use here, but it needs a macro
  #   for gensym of template symbols
  # - isCompileTime() macro: fragile, misses many AST node kinds, getTypeInst() hard-errors on untyped nodes
  let aName = genSym(nskTemplate, $alias)
  result = newStmtList()
  result.add quote do:
    when `expression` is static:
      const ct_tmp {.genSym.} = `expression`
      template `aName`(): untyped = ct_tmp
    else:
      let rt_tmp {.genSym.} = `expression`
      template `aName`(): untyped = rt_tmp

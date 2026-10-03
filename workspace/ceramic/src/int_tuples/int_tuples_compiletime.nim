# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
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
  newNimNode(nnkObjConstr).add(
    newNimNode(nnkBracketExpr).add(ident"Int", newLit(val)))

func isStaticInt*(t: NimNode): bool {.compileTime.} =
  (t.kind == nnkBracketExpr and $t[0] == "Int") or t.kind == nnkIntLit

func isStaticOne*(t: NimNode): bool {.compileTime.} =
  (t.kind == nnkBracketExpr and $t[0] == "Int" and t[1].intVal == 1) or
  (t.kind == nnkIntLit and t.intVal == 1)

func getStaticInt*(t: NimNode): int {.compileTime.} =
  ## Static Int value of a node, DynamicSentinel when the node carries
  ## no static Int (not a literal, not an Int[V] type or construction).
  case t.kind
  of nnkIntLit, nnkUIntLit:
    int(t.intVal)
  of nnkCall, nnkBracketExpr:
    if t.len >= 1 and $t[0] == "Int" and t[1].kind == nnkIntLit:
      int(t[1].intVal)
    else:
      DynamicSentinel
  else:
    DynamicSentinel

# ═══════════════════════════════════════════════════════════════
#  AST syntax sugar
# ═══════════════════════════════════════════════════════════════

func getTupleType*(n: NimNode): NimNode {.compileTime.} =
  ## The node's tuple type, values, consts, type aliases, and bare type nodes resolved uniformly.
  let t = n.getType()
  let inner =
    if t.kind == nnkBracketExpr and t[0].eqIdent("typeDesc"):
      t[1]
    else:
      t
  inner.getTypeImpl()

func isTupleTy*(t: NimNode): bool {.compileTime.} =
  ## True for tuple values and tuple types
  let ty = t.getTupleType()
  ty.kind in {nnkTupleConstr, nnkTupleTy}

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

func ceil_div*(a, b: NimNode): NimNode {.compileTime.} =
  bindSym"ceil_div".newCall(a, b)

func sign*(a: NimNode): NimNode {.compileTime.} =
  bindSym"sign".newCall(a)

proc newLetAsgn*(stmts: var NimNode; name: string; value: NimNode): NimNode {.compileTime.} =
  result = genSym(nskLet, name)
  stmts.add result.newLetStmt value

# ═══════════════════════════════════════════════════════════════
#  Constant foldable check
# ═══════════════════════════════════════════════════════════════

func isCompileTime*(node: NimNode): bool {.compileTime.} =
  ## True if `node` is a compile-time known integer expression.
  #
  # Branch analysis:
  #
  # `nnkIntLit`
  #   Matches literal integers: `1`, `16`, `1024`.
  #
  # `Int[N]`
  #
  # all-args-CT call
  #   Matches function/macro calls where EVERY argument passes `isCompileTime` recursively.
  #   This handles expressions like `1 + 2` (infix is a call).
  #
  # `nnkSym` → `nnkConstSection`
  #   Matches identifiers (symbols) that resolve to a `const` definition.
  #
  # `false` (default)
  #   Everything else
  #
  # Note: unfortunately this is very hard to get right and it is still incomplete
  if node.kind in nnkLiterals:
    return true
  if node.kind == nnkSym:
    let impl = node.getImpl()
    return impl.kind == nnkConstSection
  if node.kind in {nnkEmpty, nnkNone}:
    # Empty / None: trivially CT (no-op)
    return true
  if node.kind in {nnkIdent, nnkAccQuoted}:
    # Ident / AccQuoted: unresolved names (e.g. macro-injected it_sh, it_st)
    return false
  if node.kind == nnkPostfix and node.len >= 1:
    # Postfix / Prefix: declaration modifiers (e.g. `{.inject.} it`)
    return isCompileTime(node[^1])
  if node.kind == nnkPrefix and node.len > 0:
    for i in 0 ..< node.len:
      if not isCompileTime(node[i]):
        return false
    return true
  if node.kind == nnkBindStmt:
    return true
  if node.kind == nnkExprColonExpr:
    # ExprColonExpr (a: 7 inside named tuples): only the value (child 1) matters
    return isCompileTime(node[1])
  if node.kind in {nnkCall, nnkHiddenCallConv}:
    if node.len == 1:
      # No-arg call (e.g. Int[3]()): compile-time by nature
      return true
    if node.len > 1:
      for i in 1 ..< node.len:
        if not isCompileTime(node[i]):
          return false
    return true
  if node.kind == nnkDotExpr and node.len >= 1:
    # DotExpr: only check the base (child 0), field name (child 1) is an identifier
    return isCompileTime(node[0])
  if node.kind == nnkBlockExpr and node.len > 1:
    # BlockExpr: child 0 is a label/nil, check body from index 1
    for i in 1 ..< node.len:
      if not isCompileTime(node[i]):
        return false
    return true
  if node.kind in {nnkStmtList, nnkStmtListExpr} and node.len > 0:
    for i in 0 ..< node.len:
      if not isCompileTime(node[i]):
        return false
    return true
  if node.kind == nnkIdentDefs and node.len > 0:
    # IdentDefs: a single binding (ident, type, value) inside LetSection/VarSection
    for i in 0 ..< node.len:
      if not isCompileTime(node[i]):
        return false
    return true
  if node.kind in {nnkLetSection, nnkVarSection, nnkConstSection} and node.len > 0: # LetSection / VarSection / ConstSection: recurse into binding children
    for i in 0 ..< node.len:
      if not isCompileTime(node[i]):
        return false
    return true
  if node.kind in {nnkAsgn, nnkFastAsgn} and node.len > 1:
    # Asgn: assignment (a = b), check the value
    return isCompileTime(node[1])
  if node.kind in {nnkBracketExpr, nnkPar, nnkTupleConstr} and node.len > 0:
    # Tuple / bracket constructors
    for i in 0 ..< node.len:
      if not isCompileTime(node[i]):
        return false
    return true
  false

# ═══════════════════════════════════════════════════════════════
#  evalOnceAs
# ═══════════════════════════════════════════════════════════════

macro evalOnceAs*(alias: untyped{nkIdent}, expression: typed{lvalue|lit|`let`|`const`|`var`}): untyped =
  ## Create an `alias` for `expression`
  ## Ensuring it is evaluated only once if it is a `rvalue`
  ## or passed through if it is an lvalue.
  ##
  ## Constant expressions are constant-folded
  ##
  ##  ⚠  Wrap every use inside a template body in a `block:`.
  ##     Standalone statements in a func body need no `block:`.

  # template `alias`(): untyped =
  #   expression
  result = newProc(
    name = genSym(nskTemplate, $alias),
    params = [getType(untyped)],
    body = expression,
    procType = nnkTemplateDef
  )

macro evalOnceAs*[V: static int](alias: untyped{nkIdent}, expression: Int[V]): untyped =
  ## Create an `alias` for `expression`
  ## Ensuring it is evaluated only once if it is a `rvalue`
  ## or passed through if it is an lvalue.
  ##
  ## Constant expressions are constant-folded
  ##
  ##  ⚠  Wrap every use inside a template body in a `block:`.
  ##     Standalone statements in a func body need no `block:`.
  ##     See the section comment at the top of this file.

  # const evalOnceCT_staticInt = expression
  # template `alias`(): untyped =
  #   evalOnceCT_staticInt
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
  ## Ensuring it is evaluated only once if it is a `rvalue`
  ## or passed through if it is an lvalue.
  ##
  ## Constant expressions are constant-folded.
  ##
  ##  ⚠  Wrap every use inside a template body in a `block:`.
  ##     Standalone statements in a func body need no `block:`.
  ##     See the section comment at the top of this file.

  # Uses a generated `when expression is static:` to choose
  # between `const` (compile-time) and `let` (runtime) storage.
  # The template name is genSym'd to avoid collisions when multiple
  # evalOnceAs calls exist in the same scope.
  #
  # Implementation note — alternative approaches we tried:
  #   when compiles(static(expr)) — crashes the nimvm in a `static:` context
  #   when compiles(const = expression) — crashes in another context
  #   when is static — seems to work and is what we use here,
  #     but we need a macro for gensym of template symbols
  #   isCompileTime() macro — fragile, misses many AST node kinds,
  #     and getTypeInst() produces hard errors on untyped nodes
  let aName = genSym(nskTemplate, $alias)
  result = newStmtList()
  result.add quote do:
    when `expression` is static:
      const ct_tmp {.genSym.} = `expression`
      template `aName`(): untyped = ct_tmp
    else:
      let rt_tmp {.genSym.} = `expression`
      template `aName`(): untyped = rt_tmp

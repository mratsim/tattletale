# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import ./int_tuples_datatypes

const DynamicSentinel* = low(int)
  ## Sentinel value used throughout the library to mark a shape
  ## or stride as "unknown at compile time" (dynamic/runtime int).
  ## Returned by `toSeqStaticInts` for non-`Int[N]` elements.
  ## Code processing compile-time-known shape/stride arrays should
  ## compare against this sentinel rather than 0, to avoid confusion
  ## with a stride of literal 0 (Int[0], broadcasting).

# ═══════════════════════════════════════════════════════════════
#  isConst — compile-time detection (runtime via proc dispatch)
# ═══════════════════════════════════════════════════════════════

template isConst*(a: static int): bool = true
template isConst*(a: int): bool = false
template isConst*[V: static int](a: Int[V]): bool = true

template isConst*(a: static tuple): bool = true
template isConst*(a: tuple): bool = false

# ═══════════════════════════════════════════════════════════════
#  Int[N] compile-time helpers (for macros)
# ═══════════════════════════════════════════════════════════════

# TODO rationalize this section as there are duplicate use cases

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

#  Compile-time type helpers for the recursive macro
# --------------------------------------------------

func isIntType(x: NimNode): bool {.compileTime.} =
  ## True if `x` is typed as plain `int`.
  sameType(x, bindSym"int")

func isTupleType*(x: NimNode): bool {.compileTime.} =
  ## True if `x` is typed as a tuple type.
  x.getTypeImpl().kind == nnkTupleConstr

func isStaticIntType(x: NimNode): bool {.compileTime.} =
  ## True if `x` is typed as `Int[N]`.
  let t = x.getTypeInst()
  t.kind == nnkBracketExpr and $t[0] == "Int"

#  Constant foldable check
# --------------------------------------------------

func isCompileTime*(node: NimNode): bool {.compileTime.} =
  ## True if `node` is a compile-time known integer expression.
  ##
  ## Branch analysis:
  ##
  ## `nnkIntLit`
  ##   Matches literal integers: `1`, `16`, `1024`.
  ##   These are always compile-time values.
  ##   Example: `tiler[0]` when tiler is `(1, nr)` → the `1` is an nnkIntLit.
  ##
  ## `Int[N]` (via getTypeInst)
  ##   Matches expressions whose type is `Int[V]` for some static V.
  ##   Example: `Int[16]()` — type `Int[16]` — known at compile time.
  ##   The output of `prefix_product((Int[1], Int[16]))` is `(Int[1], Int[1])` —
  ##   each element has type `Int[1]`, so this branch catches them.
  ##
  ## all-args-CT call
  ##   Matches function/macro calls where EVERY argument passes
  ##   `isCompileTime` recursively. Index 0 (the callee) is skipped.
  ##   Examples: `max(1, Int[1]())`, `ceil_div(1024, 16)`.
  ##   This handles expressions like `1 + 2` (infix is a call).
  ##
  ## `nnkSym` → `nnkConstSection`
  ##   Matches identifiers (symbols) that resolve to a `const` definition.
  ##   Example: `const nr = 16; ... nr ...` — the reference `nr` is a sym
  ##   whose `getImpl()` returns a `nnkConstSection`.
  ##   Non-const syms (runtime `let` bindings, function parameters) fall through.
  ##
  ## `false` (default)
  ##   Everything else — runtime variables, function calls with runtime args.
  ##   Examples: `let kc = computeKc(); ... kc ...`, `someRuntimeFn(x)`.
  # Note: unfortunately this is very hard to get right.
  if node.kind in nnkLiterals:
    return true
  # Symbol: resolve to const section — do this before structural recursion
  # since getTypeInst on a const symbol returns the TYPE which can look like a
  # data constructor (e.g. nnkBracketExpr tuple type), causing spurious recursion.
  if node.kind == nnkSym:
    let impl = node.getImpl()
    return impl.kind == nnkConstSection
  # Empty / None: trivially CT (no-op)
  if node.kind in {nnkEmpty, nnkNone}:
    return true
  # Ident / AccQuoted: unresolved names (e.g. macro-injected it_sh, it_st) — never CT
  if node.kind in {nnkIdent, nnkAccQuoted}:
    return false
  # Postfix / Prefix: declaration modifiers (e.g. `{.inject.} it`) —
  # child 0 is a pragma annotation, child 1 (last) is the actual identifier
  if node.kind == nnkPostfix and node.len >= 1:
    return isCompileTime(node[^1])
  if node.kind == nnkPrefix and node.len > 0:
    for i in 0 ..< node.len:
      if not isCompileTime(node[i]):
        return false
    return true
  # BindStmt: compile-time directive (e.g. `bind makeIntTupleLeaf`) — always CT
  if node.kind == nnkBindStmt:
    return true
  # ExprColonExpr (a: 7 inside named tuples): only the value (child 1) matters
  if node.kind == nnkExprColonExpr:
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
  # DotExpr: only check the base (child 0), field name (child 1) is an identifier
  if node.kind == nnkDotExpr and node.len >= 1:
    return isCompileTime(node[0])
  # BlockExpr: child 0 is a label/nil, check body from index 1
  if node.kind == nnkBlockExpr and node.len > 1:
    for i in 1 ..< node.len:
      if not isCompileTime(node[i]):
        return false
    return true
  # StmtList / StmtListExpr: all children are statements to check
  if node.kind in {nnkStmtList, nnkStmtListExpr} and node.len > 0:
    for i in 0 ..< node.len:
      if not isCompileTime(node[i]):
        return false
    return true
  # IdentDefs: a single binding (ident, type, value) inside LetSection/VarSection
  if node.kind == nnkIdentDefs and node.len > 0:
    for i in 0 ..< node.len:
      if not isCompileTime(node[i]):
        return false
    return true
  # LetSection / VarSection / ConstSection: recurse into binding children
  if node.kind in {nnkLetSection, nnkVarSection, nnkConstSection} and node.len > 0:
    for i in 0 ..< node.len:
      if not isCompileTime(node[i]):
        return false
    return true
  # Asgn: assignment (a = b) — check the value (child 1)
  if node.kind in {nnkAsgn, nnkFastAsgn} and node.len > 1:
    return isCompileTime(node[1])
  # Tuple / bracket constructors: all children are values
  if node.kind in {nnkBracketExpr, nnkPar, nnkTupleConstr} and node.len > 0:
    for i in 0 ..< node.len:
      if not isCompileTime(node[i]):
        return false
    return true
  false

# ═══════════════════════════════════════════════════════════════
#  Compile-time seq[int]
# ═══════════════════════════════════════════════════════════════

func toSeqStaticInts*(t: NimNode): seq[int] {.compileTime.} =
  ## Recursively extract Int[N] values from a (possibly nested) tuple type AST node.
  ## Returns 0 for non-static (dynamic int) elements. Handles:
  ##   ((Int[1], Int[16]), (Int[512], Int[64]))  → @[1, 16, 512, 64]
  ##   ((int, int), (int, int))                  → @[0, 0, 0, 0]
  ##   Int[64]                                   → @[64]
  if t.kind == nnkBracketExpr and $t[0] == "Int":
    # Single Int[N] (scalar type, not tuple)
    result.add int(t[1].intVal)
  elif t.kind == nnkTupleConstr or t.kind == nnkTupleTy:
    # Recurse into tuple elements
    for i in 0 ..< t.len:
      result.add toSeqStaticInts(t[i])
  else:
    # Dynamic int (or other) — mark as unknown (low(int))
    result.add low(int)

func prefixProduct*(vals: seq[int]): seq[int] {.compileTime.} =
  ## Prefix product of a flat seq (DynamicSentinel treated as 1 for scan,
  ## but produce DynamicSentinel in output to mark unknown positions).
  result = @[1]
  for i in 0 ..< vals.len:
    if vals[i] != DynamicSentinel:
      result.add result[^1] * vals[i]
    else:
      result.add DynamicSentinel

# ═══════════════════════════════════════════════════════════════
#  evalOnceAs — evaluate at most once, preserve Int[N] for CT exprs
# ═══════════════════════════════════════════════════════════════

macro evalOnceAs*(alias: untyped{nkIdent}, expression: typed{lvalue|lit|`let`|`const`|`var`}): untyped =
  ## Create an `alias` for `expression`
  ## Ensuring it is evaluated only once if it is a `rvalue`
  ## or passed through if it is an lvalue.
  ##
  ## Constant expressions are constant-folded

  # Generate the following with `genSym` alias to avoid collisions
  #
  # template `alias`(): untyped =
  #   expression
  result = newProc(
    name = genSym(nskTemplate, $alias),
    params = [getType(untyped)],
    body = expression,
    procType = nnkTemplateDef
  )

macro evalOnceAs*(alias: untyped{nkIdent}, expression: Int): untyped =
  ## Create an `alias` for `expression`
  ## Ensuring it is evaluated only once if it is a `rvalue`
  ## or passed through if it is an lvalue.
  ##
  ## Constant expressions are constant-folded

  # const evalOnceCT_staticInt = expression
  # template `alias`(): untyped =
  #   evalOnceCT_staticInt
  result = newStmtList()
  let evalOnceCT_staticInt = genSym(nskConst, "evalOnceCT_staticInt")
  result.add newConstStmt(evalOnceCT_staticInt, expression)
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
  ## Uses a generated `when expression is static:` to choose
  ## between `const` (compile-time) and `let` (runtime) storage.
  ## The template name is genSym'd to avoid collisions when multiple
  ## evalOnceAs calls exist in the same scope.
  ##
  ## Implementation note — alternative approaches we tried:
  ##   when compiles(static(expr)) — crashes the nimvm in a `static:` context
  ##   when compiles(const = expression) — crashes in another context
  ##   when is static — seems to work and is what we use here,
  ##     but we need a macro for gensym of template symbols
  ##   isCompileTime() macro — fragile, misses many AST node kinds,
  ##     and getTypeInst() produces hard errors on untyped nodes
  let aName = genSym(nskTemplate, $alias)
  result = newStmtList()
  result.add quote do:
    when `expression` is static:
      const ct_tmp {.genSym.} = `expression`
      template `aName`(): untyped = ct_tmp
    else:
      let rt_tmp {.genSym.} = `expression`
      template `aName`(): untyped = rt_tmp

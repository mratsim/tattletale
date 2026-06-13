# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

const DynamicSentinel* = low(int)
  ## Sentinel value used throughout the library to mark a shape
  ## or stride as "unknown at compile time" (dynamic/runtime int).
  ## Returned by `toSeqStaticInts` for non-`Int[N]` elements.
  ## Code processing compile-time-known shape/stride arrays should
  ## compare against this sentinel rather than 0, to avoid confusion
  ## with a stride of literal 0 (Int[0], broadcasting).

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
  if node.kind == nnkIntLit:
    return true
  let typ = node.getTypeInst()
  if typ.kind == nnkBracketExpr and $typ[0] == "Int":
    return true
  if node.kind in {nnkCall, nnkHiddenCallConv} and node.len > 1:
    for i in 1 ..< node.len:
      if not isCompileTime(node[i]):
        return false
    return true
  if node.kind == nnkSym:
    let impl = node.getImpl()
    return impl.kind == nnkConstSection
  false

macro evalOnceAs*(expAlias: untyped{nkIdent}, exp: typed): untyped =
  ## Injects `expAlias` in caller scope, evaluating `exp` at most once.
  ## `expAlias` becomes a 0-arg template that yields the captured value.
  ##
  ## Branch analysis:
  ##
  ## `exp.kind == nnkSym` — Symbol reference (let/const/param).
  ##   No temporary needed — reuses the sym directly.
  ##   `evalOnceAs(a, x)` → `a()` yields `x`.
  ##   `isCompileTime` is NOT called: all syms (const or runtime) are
  ##   already single-evaluation by language guarantee.
  ##
  ## `elif isCompileTime(exp)` — Non-sym expression known at compile time.
  ##   Creates `const ctEval_... = exp` then wraps in `Int[ctEval_]()`.
  ##   The value is never materialized at runtime.
  ##
  ## `else` — Runtime expression (function calls, complex exprs).
  ##   Creates `let rtEval_... = exp`. The template forwards to the let.
  ##
  expectKind(expAlias, nnkIdent)
  var val = exp
  result = newStmtList()

  if exp.kind == nnkSym:
    val = exp
  elif isCompileTime(exp):
    let tmp = ident("ctEval_" & $exp.lineInfoObj)
    result.add nnkConstSection.newTree(
      nnkConstDef.newTree(tmp, newEmptyNode(), exp)
    )
    val = nnkCall.newTree(
      nnkBracketExpr.newTree(bindSym"Int", tmp)
    )
  else:
    let tmp = genSym(nskLet, "rtEval")
    result.add nnkLetSection.newTree(
      nnkIdentDefs.newTree(tmp, newEmptyNode(), exp)
    )
    val = tmp

  result.add(
    newProc(name = genSym(nskTemplate, $expAlias), params = [getType(untyped)],
      body = val, procType = nnkTemplateDef))

#  Convention: use evalOnceAs for ALL `let` bindings inside macros.
#  This ensures CT-known values become `const` (inlined) and
#  lvalues are reused directly.  Runtime expressions become `let`.

proc evalOnceField*(name: NimNode; field: string): NimNode {.compileTime.} =
  ## Build `name().field` — the call syntax for evalOnceAs'd templates.
  nnkDotExpr.newTree(nnkCall.newTree(name), ident(field))

proc evalOnceCall*(name: NimNode): NimNode {.compileTime.} =
  ## Build `name()` — call an evalOnceAs'd template.
  nnkCall.newTree(name)

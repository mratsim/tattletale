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

  # TODO: doesn't support dotExpr for const field access or method call syntax
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

template evalOnceAs*(alias: untyped{nkIdent}, expression: typed{lvalue|lit|`let`|`const`|`var`}): untyped =
  ## Create an `alias` for `expression`
  ## Ensuring it is evaluated only once if it is a `rvalue`
  ## or passed through if it is an lvalue.
  ##
  ## Constant expressions are constant-folded
  template `alias`(): untyped =
    expression

template evalOnceAs*(alias: untyped{nkIdent}, expression: typed): untyped =
  ## Create an `alias` for `expression`
  ## Ensuring it is evaluated only once if it is a `rvalue`
  ## or passed through if it is an lvalue.
  ##
  ## Constant expressions are constant-folded
  when compiles(static(expression)):
    const evalOnceCT_tmp = expression
    template `alias`(): untyped =
      evalOnceCT_tmp
  else:
    let evalOnceRT_tmp = expression
    template `alias`(): untyped =
      evalOnceRT_tmp

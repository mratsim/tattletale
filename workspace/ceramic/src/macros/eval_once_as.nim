# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Evaluate an expression once, bind it to an alias, reuse by name.

import std/macros
import workspace/ceramic/src/integer_semi_modules

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

  # A `let` binding expression would fail with "cannot evaluate at compile-time"
  # So we rebuild a constant from the type.
  # As a side-benefit, the C++ compiler should dead-code eliminate the unused `let` expression.
  result.add newConstStmt(evalOnceCT_staticInt, IntCT(V))
  result.add newProc(
    name = genSym(nskTemplate, $alias),
    params = [getType(untyped)],
    body = evalOnceCT_staticInt,
    procType = nnkTemplateDef
  )

macro evalOnceAs*(alias: untyped{nkIdent|nkSym}, expression: typed): untyped =
  ## Create an `alias` for `expression`
  ## - Returns: a template expanding to `expression`, evaluated once
  ##  ⚠  Every use inside a template body goes inside a `block:` scope.
  ##  ⚠  Standalone statements in a func body need no such scope.
  ##     See the section comment at the top of this file.

  # A generated `when expression is static:` chooses
  # between `const` (compile-time) and `let` (runtime) storage.
  # A genSym'd template name avoids collisions when multiple
  # evalOnceAs calls exist in the same scope.
  #
  # Implementation notes, alternatives rejected:
  # - when compiles(static(expr)): crashes the nimvm in a `static:` context
  # - when compiles(const = expression): crashes in another context
  # - when is static: works and is what we use here, but it needs a macro
  #   for gensym of template symbols
  # - an isCompileTime() macro: fragile, misses many AST node kinds, getTypeInst() hard-errors on untyped nodes
  let aName = if alias.kind == nnkSym: alias else: genSym(nskTemplate, $alias)
  result = newStmtList()
  result.add quote do:
    when `expression` is static:
      const ct_tmp {.genSym.} = `expression`
      template `aName`(): untyped = ct_tmp
    else:
      let rt_tmp {.genSym.} = `expression`
      template `aName`(): untyped = rt_tmp

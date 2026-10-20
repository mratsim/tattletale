## Scope table test: shadowing across push/pop
##
## Test that two same-name variables in different scopes map to distinct
## Symbols and that popping the inner scope restores the outer binding
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/symbol_resolution/t_ir_scope.nim

import workspace/crucible/src/codegen/ir/gpu_ast_datatypes
import workspace/crucible/src/codegen/passes/pass_datatypes

block:
  var ctx = GpuContext()
  let outerSym = newSymbol("x", iSym = "x_hash2_outer", typ = GpuType(kind: gtInt32), symKind = gsLocal)
  let innerSym = newSymbol("x", iSym = "x_hash2_inner", typ = GpuType(kind: gtInt32), symKind = gsLocal)

  # Outer scope
  ctx.scopeSymsStack.add(ctx.currentScopeSyms)
  ctx.currentScopeSyms = @[]
  scopeAdd(ctx.currentScopeSyms, "x", outerSym)

  # Inner scope (push)
  ctx.scopeSymsStack.add(ctx.currentScopeSyms)
  ctx.currentScopeSyms = @[]
  scopeAdd(ctx.currentScopeSyms, "x", innerSym)

  doAssert outerSym != innerSym,
    "Inner 'x' Symbol should be different object from outer 'x'"
  doAssert outerSym.iSym != innerSym.iSym,
    "Inner 'x' iSym should differ from outer 'x' iSym"
  doAssert scopeGet(ctx.currentScopeSyms, "x") == innerSym, "Inner scope should map to inner symbol"

  # Pop inner scope
  ctx.currentScopeSyms = ctx.scopeSymsStack.pop()
  doAssert scopeGet(ctx.currentScopeSyms, "x") == outerSym, "Outer scope should map to outer symbol after pop"

  # Pop outer scope
  ctx.currentScopeSyms = ctx.scopeSymsStack.pop()

echo "t_ir_scope: shadowing across push/pop verified"

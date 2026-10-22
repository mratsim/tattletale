## IR test utilities: one registered pass over hand-built input IR,
## executed at compile time.
##
## Pass bodies carry compile-time-only diagnostics (`error`, via std/macros).
## Runtime call sites sem-check those bodies
## in a context that rejects the magic, so tests call these procs
## from `static:` blocks:
##
## - the pipeline runs in the VM
## - a failing `doAssert` surfaces as a compile-time error naming the check
## - a clean compile is the pass signal

import std / [tables, sequtils]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/pass_registry

proc runPass*(register: proc(reg: var PassRegistry) {.nimcall.}, fn: GpuAst): GpuAst =
  ## Registers one pass in a fresh context and runs it over `fn`.
  ##
  ## Returns:
  ##   The transformed `fn`. A nameless fixture fn gets a fresh ident.
  var ctx = GpuContext()
  if fn.pName.isNil:
    fn.pName = GpuAst(kind: gpuIdent, symbol: newSymbol("f", iSym = "f"))
  ctx.allFnTab[fn.pName] = fn
  var reg = PassRegistry.new()
  register(reg)
  ctx.runPasses(reg)
  result = fn

proc runPassOnNode*(register: proc(reg: var PassRegistry) {.nimcall.}, n: GpuAst): seq[GpuAst] =
  ## Runs one registered pass over `n`, wrapped as a single-statement fn body.
  ##
  ## Returns:
  ##   The body statements after the pass. `[lowered]` when the pass keeps
  ##   the statement single, several entries on decomposition.
  let fn = GpuAst(kind: gpuProc,
                  pName: GpuAst(kind: gpuIdent, symbol: newSymbol("f", iSym = "f")),
                  pRetType: GpuType(kind: gtVoid),
                  pBody: GpuAst(kind: gpuBlock, statements: @[n]))
  result = register.runPass(fn).pBody.statements

proc runPassPrepared*(register: proc(reg: var PassRegistry) {.nimcall.}, ctx: var GpuContext) =
  ## Registers one pass and runs it over the caller-prepared `ctx`, in place.
  ##
  ## - the pass may read context state: globals, fnTab, globalBlocks fixtures
  ## - the test asserts on context side effects through `ctx`
  var reg = PassRegistry.new()
  register(reg)
  ctx.runPasses(reg)

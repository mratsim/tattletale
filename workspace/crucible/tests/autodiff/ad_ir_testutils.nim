# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Autodiff IR test utilities: build a kernel through the real frontend,
## run the common passes plus the autodiff transposer,
## and return the resulting procs to the test.
##
## Call the macros from a `static:` block.
## Pass bodies carry compile-time-only diagnostics:
## - a refusal surfaces as a compile error at the call site
## - a clean compile is the pass signal

import std/[macros, strutils, tables]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/passes_autodiff
import workspace/crucible/src/builtins/builtins_pragmas
import workspace/crucible/src/builtins/builtins_functions
import workspace/crucible/src/builtins/builtins_catalog

# Kernel snippets built on these utilities declare global or device procs
# and call device builtins.
# Both must resolve in the scope of the importing test.
export builtins_pragmas
export builtins_functions
export builtins_catalog

macro adKernels*(body: typed): seq[GpuAst] =
  ## Runs the frontend, the common passes, and the autodiff transposer over
  ## `body`. Returns every proc in the context in registration order:
  ##
  ## - the forward kernel first
  ## - then the generated backward kernel
  ## - then the generated scalar-reduce kernels
  var ctx = GpuContext()
  var typeReg = TypeRegistry(types: ctx.types)
  var gpuAst = nim_to_crucible.toCrucibleAst(ctx, typeReg, body)
  ctx.types = typeReg.types
  var reg = PassRegistry.new()
  reg.registerCommonPasses()
  reg.registerAutodiffPasses()
  ctx.runPasses(reg)
  var procs: seq[GpuAst]
  for fn in ctx.allFnTab.values():
    procs.add fn
  result = newLit(procs)

macro frontKernels*(body: typed): seq[GpuAst] =
  ## Runs only the frontend and the common passes over `body`.
  ## No transposition. Returns the procs of the context.
  ##
  ## Use it for hazard-negative comparisons against `adKernels`.
  var ctx = GpuContext()
  var typeReg = TypeRegistry(types: ctx.types)
  var gpuAst = nim_to_crucible.toCrucibleAst(ctx, typeReg, body)
  ctx.types = typeReg.types
  var reg = PassRegistry.new()
  reg.registerCommonPasses()
  ctx.runPasses(reg)
  var procs: seq[GpuAst]
  for fn in ctx.allFnTab.values():
    procs.add fn
  result = newLit(procs)

proc procNamed*(procs: seq[GpuAst], prefix: string, suffix = ""): GpuAst =
  ## Returns the proc whose name starts with `prefix` and, when given,
  ## ends with `suffix`, nil when absent.
  ##
  ## Name matching is prefix/suffix based: the mangleNames pass appends
  ## a hash to every kernel name.
  for fn in procs:
    if fn.kind == gpuProc and fn.pName.ident().startsWith(prefix) and
        (suffix.len == 0 or fn.pName.ident().endsWith(suffix)):
      return fn
  result = nil

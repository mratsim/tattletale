# Constantine
# Copyright (c) 2018-2019    Status Research & Development GmbH
# Copyright (c) 2020-Present Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std / tables
import ../ir/gpu_types
import ../builtins/nim_builtins
import ../passes/passes_optimizations

proc getFnName*(ctx: GpuContext; backend: BackendKind; call: GpuAst): string =
  ## Backend function name of a call to a numeric builtin function
  ## (min/max/abs), resolved via the per-backend per-type map
  ## (NimGpuNumericBuiltinsFnNames, nim_builtins.nim). Calls to anything
  ## else keep their name unchanged.
  let name = call.cName.ident()
  if name in NimGpuNumericBuiltinsFunctions:
    let t = ctx.operandType(call.cArgs[0]) # builtin min/max/abs always have 1 or 2 cArgs
    if not t.isNil:
      return getMaxMinAbsBuiltinFnName(backend, name, t.kind)
  name

proc address*(a: string): string = "&" & a
proc size*(a: string): string = "sizeof(" & a & ")"

proc isGlobal*(fn: GpuAst): bool =
  doAssert fn.kind == gpuProc, "Not a function, but: " & $fn.kind
  result = attGlobal in fn.pAttributes

proc farmTopLevel*(ctx: var GpuContext, ast: GpuAst, kernel: string, varBlock: var GpuAst) =
  case ast.kind
  of gpuProc:
    ctx.allFnTab[ast.pName] = ast
    if kernel.len > 0 and ast.pName.ident() == kernel and ast.isGlobal():
      ctx.fnTab[ast.pName] = ast.clone()
    elif kernel.len == 0 and ast.isGlobal():
      ctx.fnTab[ast.pName] = ast.clone()
  of gpuBlock:
    for ch in ast:
      ctx.farmTopLevel(ch, kernel, varBlock)
  of gpuVar, gpuConstexpr:
    varBlock.statements.add ast
  of gpuTypeDef, gpuAlias:
    raiseAssert "Unexpected type def / alias def found. These should be in `ctx.types` now: " & $ast
  else:
    discard

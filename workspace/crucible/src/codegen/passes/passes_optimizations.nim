## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
import std / [sequtils, tables]
import ../ir/gpu_types
import ./pass_datatypes


proc isLvalue*(n: GpuAst): bool =
  ## Returns true if the AST node is an lvalue (can have its address taken).
  case n.kind
  of gpuIdent: true
  of gpuIndex: true
  of gpuDeref: true
  else: false

proc walkNonLvalueArgs(ctx: var GpuContext; n: var GpuAst) =
  case n.kind
  of gpuCall:
    let fnParams = ctx.getFnParams(n.cName)
    for i, arg in n.cArgs:
      if i < fnParams.len and fnParams[i].passByRef and not arg.isLvalue():
        n.cArgs[i] = GpuAst(kind: gpuMaterialize,
          mExpr: arg,
          mType: fnParams[i].typ)
    for ch in n.mitems:
      walkNonLvalueArgs(ctx, ch)
  else:
    for ch in n.mitems:
      walkNonLvalueArgs(ctx, ch)

proc materializePassByRefArgs*(ctx: var GpuContext) =
  ## Transforms non-lvalue arguments to passByRef parameters into
  ## gpuMaterialize nodes that backends can handle appropriately.
  for fnKey in ctx.allFnTab.keys:
    var fn = ctx.allFnTab[fnKey]
    walkNonLvalueArgs(ctx, fn.pBody)
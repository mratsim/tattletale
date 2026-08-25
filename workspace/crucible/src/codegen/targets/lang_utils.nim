# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std / tables
import ../ir/gpu_types
import ../builtins/builtins_functions
import ../builtins/builtins_gpu_types
import ../passes/passes_optimizations

proc getFnName*(ctx: GpuContext; backend: BackendKind; call: GpuAst): string =
  ## Backend function name of a call to a numeric builtin function
  ## (min/max/abs), resolved via the per-backend per-type map
  ## (NimGpuNumericBuiltinsFnNames, builtins_functions.nim). Calls to anything
  ## else keep their name unchanged.
  let name = call.cName.ident()
  if name in NimGpuNumericBuiltinsFunctions:
    let t = ctx.operandType(call.cArgs[0]) # builtin min/max/abs always have 1 or 2 cArgs
    if not t.isNil:
      return getMaxMinAbsBuiltinFnName(backend, name, t.kind)
  NimGpuFp16ConversionBuiltins.getOrDefault((backend, name), name)

proc address*(a: string): string = "&" & a
proc size*(a: string): string = "sizeof(" & a & ")"

proc genAsmStmt*(ast: GpuAst): string =
  ## Substitute the TAG_IDENT_IN_ASM<n> placeholders with the operand
  ## symbols' display names.
  ## Backtick identifiers in `asm` resolve to Nim symbols.
  ## The IR stores them as gpuIdent ops and the printers substitute the
  ## (possibly mangled) name at codegen time.
  var s = ast.stmt
  var i = 0
  while i < s.len:
    if s[i] == TAG_IDENT_IN_ASM[0]:
      var j = i + 1
      var idx = 0
      while j < s.len and s[j] in {'0' .. '9'}:
        idx = idx * 10 + (ord(s[j]) - ord('0'))
        inc j
      result.add ast.ops[idx].symbol.name
      i = j
    else:
      result.add s[i]
      inc i

proc genEmitStmt*(ctx: var GpuContext; ast: GpuAst;
                  renderExpr: proc(ctx: var GpuContext; n: GpuAst): string): string =
  ## Renders a `gpuEmit` statement: literal parts pass through verbatim,
  ## expression parts render through the per-target expression codegen.
  for part in ast.parts:
    case part.kind
    of peLiteral:
      result.add part.literal
    of peExpr:
      result.add renderExpr(ctx, part.expr)

proc isSelfTerminating*(el: GpuAst): bool =
  ## True when the statement renders its own terminator and the gpuBlock loop
  ## must not append `;` (nested blocks and `gpuEmit` raw text).
  result = el.kind in {gpuBlock, gpuEmit}

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

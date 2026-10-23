# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std / [tables, sequtils]
import ../ir/ast_datatypes
import ../ir/ast_type_resolvers
import ../../builtins/builtins_functions
import ../../builtins/builtins_gpu_types

# ═══════════════════════════════════════════════════════════════════════
# Builtin function names
# ═══════════════════════════════════════════════════════════════════════

proc getFnName*(ctx: GpuContext; backend: BackendKind; call: GpuAst): string =
  ## Returns the emitted name for a call, resolving the ambiguous builtins per backend:
  ## - `min`/`max`/`abs` per operand type through NimGpuNumericBuiltinsFnNames
  ## - the fp16 conversions through NimGpuFp16ConversionBuiltins
  ## - everything else verbatim
  ##
  ## A materialized numeric builtin reaches the printer as a mangled display name (`max_RJ1UXDu`, `max___ZE4xeJFcdj6wQZpDx9bBlvQ_SStiky4`).
  ## A backend that declares an intrinsic name for the builtin/operand-type combo prints that intrinsic and drops the materialized body.
  ## A backend without a declared mapping keeps the display name and emits the body alongside the call (GLSL/CUDA/OpenCL/Metal).
  let name = call.cName.ident()
  let builtin = numericBuiltinDisplayName(name)
  if builtin.len > 0:
    let t = ctx.getOperandType(call.cArgs[0]) # min/max/abs always have 1 or 2 cArgs
    if not t.isNil and hasMaxMinAbsBuiltinFnName(backend, builtin, t.kind):
      return getMaxMinAbsBuiltinFnName(backend, builtin, t.kind)
    return name
  NimGpuFp16ConversionBuiltins.getOrDefault((backend, name), name)

# ═══════════════════════════════════════════════════════════════════════
# Raw statement rendering
# ═══════════════════════════════════════════════════════════════════════

func genAsmStmt*(ast: GpuAst): string =
  var s = ast.stmt
  var i = 0
  while i < s.len:
    if s[i] == TAG_IDENT_INLINE_ASM[0]:
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
  for part in ast.parts:
    case part.kind
    of peLiteral:
      result.add part.literal
    of peExpr:
      result.add renderExpr(ctx, part.expr)

# ═══════════════════════════════════════════════════════════════════════
# IR predicates and farming
# ═══════════════════════════════════════════════════════════════════════

func isSelfTerminating*(el: GpuAst): bool =
  result = el.kind in {gpuBlock, gpuEmit}

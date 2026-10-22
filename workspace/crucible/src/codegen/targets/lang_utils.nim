# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std / [tables, sequtils]
import ../ir/gpu_ast_datatypes
import ../ir/gpu_ast_type_resolvers
import ../../builtins/builtins_functions
import ../../builtins/builtins_gpu_types

# ═══════════════════════════════════════════════════════════════════════
# Builtin function names
# ═══════════════════════════════════════════════════════════════════════

proc getFnName*(ctx: GpuContext; backend: BackendKind; call: GpuAst): string =
  let name = call.cName.ident()
  if name in NimGpuNumericBuiltinsFunctions:
    let t = ctx.getOperandType(call.cArgs[0]) # min/max/abs always have 1 or 2 cArgs
    if not t.isNil:
      return getMaxMinAbsBuiltinFnName(backend, name, t.kind)
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
# Metal printer helpers
# ═══════════════════════════════════════════════════════════════════════

func variantSuffix*(spaces: seq[AddressSpace]): string =
  result = "_"
  for space in spaces:
    case space
    of asDevice: result.add "device"
    of asConstant: result.add "const"
    of asSMEM: result.add "smem"
    of asRMEM: result.add "rmem"

# ═══════════════════════════════════════════════════════════════════════
# IR predicates and farming
# ═══════════════════════════════════════════════════════════════════════

func isSelfTerminating*(el: GpuAst): bool =
  result = el.kind in {gpuBlock, gpuEmit}

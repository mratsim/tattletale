## filterPragmas pass
##
## Test that filterPragmas maps raw proc pragmas to typed `pAttributes`
##
## - `device` and `global` land as `attDevice` and `attGlobal`
## - Nim-specific pragmas drop, `inline` becomes `attForceInline`
## - empty `pRawPragmas` yields empty `pAttributes`
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/functions/t_ir_filterPragmas.nim

import std / [tables, sequtils]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/pass_registry

proc runOnFn(regProc: proc(reg: var PassRegistry) {.nimcall.}, fn: GpuAst) =
  ## Runs one registered pass over a fixture fn, the public-API path.
  var ctx = GpuContext()
  if fn.pName.isNil:
    fn.pName = GpuAst(kind: gpuIdent, symbol: newSymbol("f", iSym = "f"))
  ctx.allFnTab[fn.pName] = fn
  var reg = PassRegistry.new()
  regProc(reg)
  ctx.runPasses(reg)

# ═══════════════════════════════════════════════════════════════════════
# 1. Device pragma is preserved
# ═══════════════════════════════════════════════════════════════════════
block:
  var fn = GpuAst(kind: gpuProc, pRawPragmas: @["device"])
  runOnFn(registerFilterPragmas, fn)
  doAssert attDevice in fn.pAttributes, "device pragma should be in pAttributes"
  echo "  OK — device pragma preserved"

# ═══════════════════════════════════════════════════════════════════════
# 2. Global pragma is preserved
# ═══════════════════════════════════════════════════════════════════════
block:
  var fn = GpuAst(kind: gpuProc, pRawPragmas: @["global"])
  runOnFn(registerFilterPragmas, fn)
  doAssert attGlobal in fn.pAttributes, "global pragma should be in pAttributes"
  echo "  OK — global pragma preserved"

# ═══════════════════════════════════════════════════════════════════════
# 3. Nim-specific pragmas (nimcall) are dropped, inline preserved
# ═══════════════════════════════════════════════════════════════════════
block:
  var fn = GpuAst(kind: gpuProc, pRawPragmas: @["nimcall", "noSideEffect", "inline"])
  runOnFn(registerFilterPragmas, fn)
  doAssert attForceInline in fn.pAttributes, "inline pragma should be preserved"
  doAssert attDevice notin fn.pAttributes, "nimcall should NOT produce device attribute"
  echo "  OK — Nim-specific pragmas dropped, inline preserved"

# ═══════════════════════════════════════════════════════════════════════
# 4. Mixed pragmas
# ═══════════════════════════════════════════════════════════════════════
block:
  var fn = GpuAst(kind: gpuProc, pRawPragmas: @["device", "forceinline", "nimcall", "closure"])
  runOnFn(registerFilterPragmas, fn)
  doAssert attDevice in fn.pAttributes, "device should be in pAttributes"
  doAssert attForceInline in fn.pAttributes, "forceinline should be in pAttributes"
  doAssert fn.pAttributes.len == 2,
    "Should have exactly 2 attributes (device + forceinline), got " & $fn.pAttributes.len
  echo "  OK — mixed pragmas filtered correctly"

# ═══════════════════════════════════════════════════════════════════════
# 5. Empty pRawPragmas produces empty pAttributes
# ═══════════════════════════════════════════════════════════════════════
block:
  var fn = GpuAst(kind: gpuProc, pRawPragmas: @[])
  runOnFn(registerFilterPragmas, fn)
  doAssert fn.pAttributes.len == 0, "Empty raw pragmas should produce empty pAttributes"
  echo "  OK — empty pRawPragmas produces empty pAttributes"

# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "  All filterPragmas tests passed."

## filterPragmas pass
##
## Test that filterPragmas maps raw proc pragmas to typed `pAttributes`
##
## - `device` and `global` land as `attDevice` and `attGlobal`
## - Nim-specific pragmas drop, `inline` becomes `attForceInline`
## - empty `pRawPragmas` yields empty `pAttributes`
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/functions/t_ir_filterPragmas.nim

import std / [tables, sequtils]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_normalizations {.all.}
import workspace/crucible/tests/ir/ir_testutils
import workspace/crucible/src/codegen/passes/pass_registry

# ═══════════════════════════════════════════════════════════════════════
# 1. Device pragma is preserved
# ═══════════════════════════════════════════════════════════════════════
static:
  var fn = GpuAst(kind: gpuProc, pRawPragmas: @["device"])
  discard registerFilterPragmas.runPass(fn)
  doAssert attDevice in fn.pAttributes, "device pragma should be in pAttributes"
  echo "  OK — device pragma preserved"

# ═══════════════════════════════════════════════════════════════════════
# 2. Global pragma is preserved
# ═══════════════════════════════════════════════════════════════════════
static:
  var fn = GpuAst(kind: gpuProc, pRawPragmas: @["global"])
  discard registerFilterPragmas.runPass(fn)
  doAssert attGlobal in fn.pAttributes, "global pragma should be in pAttributes"
  echo "  OK — global pragma preserved"

# ═══════════════════════════════════════════════════════════════════════
# 3. Nim-specific pragmas (nimcall) are dropped, inline preserved
# ═══════════════════════════════════════════════════════════════════════
static:
  var fn = GpuAst(kind: gpuProc, pRawPragmas: @["nimcall", "noSideEffect", "inline"])
  discard registerFilterPragmas.runPass(fn)
  doAssert attForceInline in fn.pAttributes, "inline pragma should be preserved"
  doAssert attDevice notin fn.pAttributes, "nimcall should NOT produce device attribute"
  echo "  OK — Nim-specific pragmas dropped, inline preserved"

# ═══════════════════════════════════════════════════════════════════════
# 4. Mixed pragmas
# ═══════════════════════════════════════════════════════════════════════
static:
  var fn = GpuAst(kind: gpuProc, pRawPragmas: @["device", "forceinline", "nimcall", "closure"])
  discard registerFilterPragmas.runPass(fn)
  doAssert attDevice in fn.pAttributes, "device should be in pAttributes"
  doAssert attForceInline in fn.pAttributes, "forceinline should be in pAttributes"
  doAssert fn.pAttributes.len == 2,
    "Should have exactly 2 attributes (device + forceinline), got " & $fn.pAttributes.len
  echo "  OK — mixed pragmas filtered correctly"

# ═══════════════════════════════════════════════════════════════════════
# 5. Empty pRawPragmas produces empty pAttributes
# ═══════════════════════════════════════════════════════════════════════
static:
  var fn = GpuAst(kind: gpuProc, pRawPragmas: @[])
  discard registerFilterPragmas.runPass(fn)
  doAssert fn.pAttributes.len == 0, "Empty raw pragmas should produce empty pAttributes"
  echo "  OK — empty pRawPragmas produces empty pAttributes"

# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "  All filterPragmas tests passed."

## Phase 4: filterPragmas pass test
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off --debugger:native \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_filterPragmas.nim

import std / [tables, sequtils]
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/passes/passes_normalizations

# ═══════════════════════════════════════════════════════════════════════
# 1. Device pragma is preserved
# ═══════════════════════════════════════════════════════════════════════
block:
  var fn = GpuAst(kind: gpuProc, pRawPragmas: @["device"])
  filterPragmasImpl(fn)
  doAssert attDevice in fn.pAttributes, "device pragma should be in pAttributes"
  echo "  OK — device pragma preserved"

# ═══════════════════════════════════════════════════════════════════════
# 2. Global pragma is preserved
# ═══════════════════════════════════════════════════════════════════════
block:
  var fn = GpuAst(kind: gpuProc, pRawPragmas: @["global"])
  filterPragmasImpl(fn)
  doAssert attGlobal in fn.pAttributes, "global pragma should be in pAttributes"
  echo "  OK — global pragma preserved"

# ═══════════════════════════════════════════════════════════════════════
# 3. Nim-specific pragmas (nimcall) are dropped, inline preserved
# ═══════════════════════════════════════════════════════════════════════
block:
  var fn = GpuAst(kind: gpuProc, pRawPragmas: @["nimcall", "noSideEffect", "inline"])
  filterPragmasImpl(fn)
  doAssert attForceInline in fn.pAttributes, "inline pragma should be preserved"
  doAssert attDevice notin fn.pAttributes, "nimcall should NOT produce device attribute"
  echo "  OK — Nim-specific pragmas dropped, inline preserved"

# ═══════════════════════════════════════════════════════════════════════
# 4. Mixed pragmas
# ═══════════════════════════════════════════════════════════════════════
block:
  var fn = GpuAst(kind: gpuProc, pRawPragmas: @["device", "forceinline", "nimcall", "closure"])
  filterPragmasImpl(fn)
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
  filterPragmasImpl(fn)
  doAssert fn.pAttributes.len == 0, "Empty raw pragmas should produce empty pAttributes"
  echo "  OK — empty pRawPragmas produces empty pAttributes"

# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "  All filterPragmas tests passed."

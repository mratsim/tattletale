## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Vulkan IR legalization pass 4: fail-loudly subgroup-size guard.
##
## Kernels whose transitive call graph reaches a subgroup-shuffle reduction
## builtin get `if (gl_SubgroupSize < 32u) { return; }` as their first statement,
## and their lane id is rewritten to `gl_SubgroupInvocationID`.
## Runs after pass 3.

import std/[sets]
import ../ir/gpu_types
import ./passes_legalization_vulkan_ptr_in_struct

# ═════════════════════════════════════════════════════════════════════════
#  Pass 4: vulkanSubgroupGuard32 (GPU-B-001)
# ═════════════════════════════════════════════════════════════════════════

proc usesReductionBuiltin(n: GpuAst): bool =
  ## True when the body contains a subgroup-shuffle (reduction) builtin call.
  if n.isNil: return false
  if n.kind == gpuCall and n.cName.symbol != nil and
     n.cName.symbol.reductionBuiltin != gbkNone:
    return true
  for ch in n:
    if usesReductionBuiltin(ch):
      return true

proc subgroupGuard32*(ctx: var GpuContext) =
  ## GPU-B-001: the fp16-subgroup shuffle path (tileKMax reduction trees,
  ## universalMma8x8x8) assumes 32-lane subgroups. The shuffle trees use
  ## deltas up to 16 and a 32-lane bit decomposition, both undefined on
  ## 8/16-lane-subgroup devices (Intel Gen9+, some Mali/Adreno). Fail
  ## loudly instead of silently computing wrong results:
  ##  - every kernel whose transitive call graph reaches a reduction builtin
  ##    gets `if (gl_SubgroupSize < 32u) { return; }` as its first statement
  ##    (the kernel returns without writing its outputs, so a host-side
  ##    value check fails loudly).
  ##  - in those fns, the lane id comes from `gl_SubgroupInvocationID` (the
  ##    true subgroup lane) instead of `gl_LocalInvocationIndex` (the
  ##    workgroup lane, equal only when workgroup == subgroup, which the
  ##    guard fixes at 32 alongside the kernels' baked 32-wide workgroups).
  ## The engine does not ingest VkPhysicalDeviceSubgroupProperties, so the subgroup size is not confirmed at runtime.
  let reachable = reachableFns(ctx)
  # transitive closure over the call graph: a fn is shuffle-reachable when
  # its body contains a reduction builtin or calls a shuffle-reachable fn
  var shuffleReachable = initHashSet[string]()
  var changed = true
  while changed:
    changed = false
    for fn in reachable:
      if fn.pName.symbol.iSym in shuffleReachable: continue
      var hits = not fn.pBody.isNil and usesReductionBuiltin(fn.pBody)
      if not hits and not fn.pBody.isNil:
        var calls: seq[GpuAst]
        collectCalls(fn.pBody, calls)
        for c in calls:
          if c.cName.symbol != nil and c.cName.symbol.iSym in shuffleReachable:
            hits = true
            break
      if hits:
        shuffleReachable.incl fn.pName.symbol.iSym
        changed = true
  # lane id: thread_index_in_threadgroup → gl_SubgroupInvocationID in
  # shuffle-reachable bodies. Replace the node rather than mutating it: the
  # catalog ident node is sigTab-shared across the module, so an in-place
  # symbol swap would leak the subgroup lane into every non-shuffle fn that
  # still references the shared node. Non-shuffle fns
  # must keep gl_LocalInvocationIndex (gl_SubgroupInvocationID is only valid
  # where the subgroup extensions are enabled).
  proc rewriteLaneId(n: var GpuAst) =
    if n.kind == gpuIdent and n.symbol != nil and
       n.symbol.coordBuiltin == gbkThreadIndexInThreadgroup:
      n = GpuAst(kind: gpuIdent, symbol: newSymbol("gl_SubgroupInvocationID",
                 typ = n.symbol.typ, symKind = gsBuiltin))
    else:
      for ch in n.mitems:
        rewriteLaneId(ch)
  for fn in reachable:
    if fn.pName.symbol.iSym in shuffleReachable and not fn.pBody.isNil:
      rewriteLaneId(fn.pBody)
  # guard: first statement of every subgroup-using kernel
  for fn in reachable:
    if fn.isGlobalFn() and fn.pName.symbol.iSym in shuffleReachable and
       not fn.pBody.isNil:
      let guard = GpuAst(kind: gpuEmit, parts: @[GpuEmitPart(
        kind: peLiteral, literal: "if (gl_SubgroupSize < 32u) { return; }")])
      if fn.pBody.kind == gpuBlock:
        fn.pBody.statements.insert(guard, 0)
      else:
        fn.pBody = GpuAst(kind: gpuBlock, statements: @[guard, fn.pBody])

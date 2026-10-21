# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[macros, sequtils, strutils, tables]
import ../ir/gpu_ast_datatypes
import ./pass_datatypes
import ./pass_registry
import ./passes_legalizations
import ./passes_validations
import ./passes_preprocessing
import ./passes_lowering # insertPointerDeref

proc sameLayout(a, b: seq[(string, GpuType, AddressSpace)]): bool =
  ## Element-wise compare, direct calls only: the generic seq/tuple
  ## `==` cannot lift the side-effecting `GpuType ==`
  if a.len != b.len:
    return false
  for i in 0 ..< a.len:
    if a[i][0] != b[i][0] or a[i][1] != b[i][1] or a[i][2] != b[i][2]:
      return false
  result = true

proc validateSharedPushConstLayout(ctx: var GpuContext) =
  ## Vulkan kernels in one source share one push-constant block: every
  ## kernel's ordered non-smem params must agree, else a launch serializes
  ## its args at offsets another kernel reads as different members
  var reference: seq[(string, GpuType, AddressSpace)]
  for _, fn in ctx.allFnTab:
    if not fn.isGlobalFn():
      continue
    var layout: seq[(string, GpuType, AddressSpace)]
    for p in fn.pParams:
      if p.addressSpace != asSMEM:
        layout.add (p.ident.ident(), p.typ, p.addressSpace)
    if reference.len == 0:
      reference = layout
    elif not sameLayout(layout, reference):
      let names = layout.mapIt(it[0]).join(", ")
      error "Multi-kernel Vulkan sources share one push-constant block, all kernels " &
            "must declare the same ordered params, got: " & names

proc registerVulkanPasses*(reg: var PassRegistry) =
  ## Register Vulkan-specific passes: GLSL keyword rejection, pointer deref
  ## insertion for buffer-reference rendering.
  reg.register("rejectVulkanKeywords", pkValidation, phaseMain,
    "Rejects identifiers that are reserved GLSL keywords",
    proc(ctx: var GpuContext): void =
      ctx.checkReservedKeywords(["extern", "interface", "buffer"], "GLSL")
  )
  reg.register("validateSharedPushConstLayout", pkValidation, phaseMain,
    "Rejects multi-kernel sources whose push-constant layouts differ",
    proc(ctx: var GpuContext): void =
      ctx.validateSharedPushConstLayout()
  )
  # GLSL buffer references cannot be indexed directly, the deref must be
  # explicit in the IR for the renderer to compose `.x`.
  reg.register(
    name = "insertPointerDeref",
    kind = pkTransform,
    phase = phaseMain,
    description = "Wraps pointer-typed index and field parents in explicit gpuDeref",
    dependsOn = @[
      "rewriteIndexDeref"],
    run = insertPointerDeref
  )

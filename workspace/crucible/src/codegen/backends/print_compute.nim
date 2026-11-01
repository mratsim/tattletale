# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Compute-tier emission shared by the CUDA C++, OpenCL C, GLSL and MSL printers:
## the module's struct declarations and program-scope globals.
##
## Each printer owns a `PrintCtx` (`ctx: var auto`) and supplies the target
## vocabulary as `mixin` procs beside it, matching the `print_common` pattern.
##
## - `emitStructField(struct, field)`: one struct member's declaration text

import ../ir/cirv2_lang
import ./print_common

proc emitFeatureLines*(features: ModuleFeatures, lines: openArray[tuple[on: bool, text: string]]): string =
  ## Extension pragma block of one module over the target's feature table.
  ##
  ## Expected input:
  ## - `features`: the stored module feature scan (`CirModule.features`)
  ## - `lines`: the target's table, one (feature flag, pragma line) pair per
  ##   line the block can emit, in emission order
  ##
  ## Output: one pragma line per set flag, table order, then a blank line.
  ##
  ##   emitFeatureLines(features, openclFeatureTable)  # pragma lines or ""
  ##
  for entry in lines:
    if entry.on:
      result.add entry.text
  result.add "\n"

proc printComputeGlobals*(ctx: var auto, m: CirModule, spaces: set[AddressSpace]): string =
  ## Program-scope declarations of the relocated storage locals in `CirModule.globals`.
  ##
  ## Expected input:
  ## - `m.globals`: the program-scope declarations that the target's lowering pass relocated (const_mem, and smem on GLSL)
  ## - `spaces`: the address spaces this target declares at program scope
  ## - `emitComputeGlobal(s)` (mixin): one declaration's text, "" skips the entry
  ##
  ## Output: one declaration per `sVar` in `spaces`, encounter order, then a blank line after a non-empty section.
  ##
  ##   src &= printComputeGlobals(ctx, m, {caConstant})
  mixin emitComputeGlobal
  for s in m.globals:
    if s.kind == sVar and s.space in spaces:
      let decl = ctx.emitComputeGlobal(s)
      if decl.len > 0:
        result &= decl & '\n'
  if result.len > 0:
    result &= '\n'

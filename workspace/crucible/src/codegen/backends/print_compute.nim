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

proc declareExtensions*(features: ModuleFeatures, lines: openArray[tuple[on: bool, text: string]]): string =
  ## Extension pragma block of one module over the target's feature table.
  ##
  ## Expected input:
  ## - `features`: the stored module feature scan (`CirModule.features`)
  ## - `lines`: the target's table, one (feature flag, pragma line) pair per
  ##   line the block can emit, in emission order
  ##
  ## Output: one pragma line per set flag, table order, then a blank line.
  ##
  ##   declareExtensions(features, openclFeatureTable)  # pragma lines or ""
  ##
  for entry in lines:
    if entry.on:
      result.add entry.text
  result.add "\n"

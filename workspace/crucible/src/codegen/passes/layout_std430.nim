# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Struct layout analysis: one `pkAnalysis` pass per target filling `CirModule.layouts` with the packed layout of every nominal struct type.
##
##   last target transform → glslComputeStructLayouts | wgslComputeStructLayouts → validation
##
## Packing rules, what a host-side packer and a `sizeof`-style query must both compute:
## - a leaf kind packs in its byte width: a scalar `bits div 8`, a bool `boolWidth`, a pointer 8, a proc type 0
## - a vector packs its `vN` components back to back, its alignment equals the component width
## - an array packs `aN` element copies back to back
## - a struct member starts at the first offset that its alignment allows, and the struct size rounds up to the struct alignment
##
## Rule-set variant: `boolWidth`. A bool packs in `boolWidth` bytes:
## - GLSL sets it to 4: glslang compiles a bool member to a 32-bit integer, matching the word that the launch ABI packs
## - WGSL sets it to 4, one launch-ABI word per bool
##
## Input domain: `CirType`. The SPIR-V backend keeps `std430Layout` local over the post-patch `SpvTy` domain:
##
## - the SPIR-V bool retype pass rewrites bool kernel parameters and globals to i32 before printing
## - no CirType-domain bool width describes SPIR-V member widths
##
## SPIR-V packing deltas over the CirType rules:
## - the SPIR-V rule set aligns 3- and 4-component vectors at four times the element alignment
## - the SPIR-V rule set rounds array strides up to the element alignment
## - the binary encodes those SPIR-V values as array strides and as struct member offsets
## - the aligned buffer stores take their element alignment from the same values
##
## Contract:
## - each target registers its pass after its last transform and before the check-only validation stage
## - analysis results are valid only if no transform runs after them, which runPasses enforces
## - the printer reads the stored records, it never recomputes a struct layout

import std/tables
import ../ir/cirv2_lang
import ../backends/print_common
import ./pass_datatypes

# ═══════════════════════════════════════════════════════════════════════
# Packing rules
# ═══════════════════════════════════════════════════════════════════════

const std430BoolWidth* = 4
## Bool byte width under the `std430` packing that the GLSL blocks declare:
## glslang compiles a Vulkan std430 bool member to a 32-bit integer,
## a full 4-byte word the launch ABI packs as well.

const launchAbiBoolWidth* = 4
## Bool byte width under the launch ABI: one 4-byte word per bool.

proc packedAlign(t: CirType, boolWidth: static int): int =
  ## Launch-packing byte alignment of a type.
  case t.kind
  of ctStruct:
    var align = 1
    for f in t.fields:
      align = max(align, packedAlign(f.typ, boolWidth))
    align
  of ctArray: packedAlign(t.aElem, boolWidth)
  of ctVector: max(scalarByteSize(t.vElem, boolWidth), 1)
  of ctPtr, ctAptr: 8
  else: max(scalarByteSize(t, boolWidth), 1)

proc packedSize(t: CirType, boolWidth: static int): int =
  ## Launch-packing byte size of a type.
  case t.kind
  of ctVoid, ctBool, ctInt, ctUint, ctFloat:
    scalarByteSize(t, boolWidth)
  of ctVector:
    scalarByteSize(t.vElem, boolWidth) * t.vN
  of ctArray:
    packedSize(t.aElem, boolWidth) * t.aN
  of ctStruct:
    var off = 0
    var align = 1
    for f in t.fields:
      let fa = packedAlign(f.typ, boolWidth)
      off = roundUpMultiple(off, fa) + packedSize(f.typ, boolWidth)
      align = max(align, fa)
    roundUpMultiple(off, align)
  of ctPtr, ctAptr: 8
  of ctSpan, ctProc: 0

proc computeStructLayout(t: CirType, boolWidth: static int): StructLayout =
  ## Packed layout of one struct type under the launch packing rules.
  ##
  ## Returns:
  ## - the padded struct size
  ## - the strictest member alignment
  doAssert t.kind == ctStruct
  var off = 0
  var align = 1
  for f in t.fields:
    let fa = packedAlign(f.typ, boolWidth)
    off = roundUpMultiple(off, fa)
    off += packedSize(f.typ, boolWidth)
    align = max(align, fa)
  result.align = align
  result.size = roundUpMultiple(off, align)

proc typeByteSize*(layouts: Table[string, StructLayout], t: CirType, boolWidth: static int): int =
  ## Packed byte size of any type, matching the `eSizeof` value.
  ##
  ## - a nominal struct reads its stored record in `layouts`, which the layout
  ##   pass filled for every struct in `CirModule.types`
  ## - every other kind derives from the packing rules
  ##
  ##   typeByteSize(m.layouts, tVec(3, tFloat(32)), std430BoolWidth) == 12
  case t.kind
  of ctStruct:
    doAssert layouts.hasKey(t.name), "no stored layout for struct " & t.name
    layouts[t.name].size
  of ctVoid, ctBool, ctInt, ctUint, ctFloat:
    scalarByteSize(t, boolWidth)
  of ctVector:
    scalarByteSize(t.vElem, boolWidth) * t.vN
  of ctArray:
    typeByteSize(layouts, t.aElem, boolWidth) * t.aN
  of ctPtr, ctAptr: 8
  of ctSpan, ctProc: 0

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

proc glslRun(m: CirModule): string {.nimcall.} =
  for t in m.types:
    if t.kind == ctStruct:
      m.layouts[t.name] = computeStructLayout(t, std430BoolWidth)

proc wgslRun(m: CirModule): string {.nimcall.} =
  for t in m.types:
    if t.kind == ctStruct:
      m.layouts[t.name] = computeStructLayout(t, launchAbiBoolWidth)

proc registerGlslLayoutPass*(reg: var PassRegistry) =
  ## Contract:
  ## - fills `CirModule.layouts` with the `std430`-bool packing of every
  ##   nominal struct type
  ## - registers after the target's last transform, before the check-only
  ##   validation stage
  ## - runPasses refuses a transform registered after an analysis
  reg.register("glslComputeStructLayouts", pkAnalysis,
    "Computes the packed layout of every nominal struct type once:" &
    " member offsets, struct alignment, padded struct size",
    run = glslRun)

proc registerWgslLayoutPass*(reg: var PassRegistry) =
  ## Contract:
  ## - fills `CirModule.layouts` with the launch-ABI packing of every
  ##   nominal struct type, 4-byte bools included
  ## - registers after the target's last transform, before the check-only
  ##   validation stage
  ## - runPasses refuses a transform registered after an analysis
  reg.register("wgslComputeStructLayouts", pkAnalysis,
    "Computes the packed layout of every nominal struct type once:" &
    " member offsets, struct alignment, padded struct size",
    run = wgslRun)

## Lowering stage over CIRv2: shared, target-independent lowering passes.
## Per-target transforms register from the passes_lowering_<target> files,
## the printers in backends/ are table-driven.

import ./pass_datatypes
import ./passes_low_views

func registerSharedLoweringPasses*(reg: var PassRegistry) =
  ## Shared target-independent passes:
  ##
  ## - `lowerSpans`: openArray spans split to pointer + length pairs
  ##
  ## Struct kernel parameters stay whole:
  ##
  ## - the push-block targets (GLSL, HLSL, SPIR-V) pack them into the block
  ## - the flat resource targets register the explosion per target
  reg.registerLowerSpans()

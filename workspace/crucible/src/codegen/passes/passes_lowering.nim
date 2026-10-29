## Lowering stage over CIRv2: shared, target-independent lowering passes.
## Printers are table-driven (backends/), there is no per-backend
## lowering pass file yet.

import ./pass_datatypes

func registerSharedLoweringPasses*(reg: var PassRegistry) =
  ## Shared target-independent passes. Struct kernel parameters stay whole:
  ## the push-block targets (GLSL, HLSL, SPIR-V) pack them into the block,
  ## the flat resource targets register the explosion per target.

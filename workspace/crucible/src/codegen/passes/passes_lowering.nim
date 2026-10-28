## Lowering stage over CIRv2: the shared, target-independent lowering passes.
## The printers are table-driven (backends/), there is no per-backend
## lowering pass file yet.

import ./pass_datatypes
import ./passes_low_kernel_params

func registerSharedLoweringPasses*(reg: var PassRegistry) =
  reg.registerExplodeKernelStructs()

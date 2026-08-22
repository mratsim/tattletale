## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Universal scalar-FMA atom instantiations (bk_FMA).
##
## The 1×1×1 degenerate MMA: one thread, one value per operand, computing
## D = A·B + C in plain arithmetic. It is the fallback for backends and
## datatypes with no MMA instruction (Vulkan/WebGPU cannot reach MMAs,
## wgpu-native executes on CPU), and the correctness reference for the
## tiled GEMM machinery, whose partitioning degenerates to the trivial
## (T=1, V=1) layouts with no special casing.
##
## Lineage: CuTe `UniversalFMA<D, A, B, C>` (CUTLASS `cute/arch/mma.hpp`)
## registers D[1]/A[1]/B[1]/C[1] with `fma(d, a, b, c)`; its `MMA_Traits`
## give Shape_MNK (1, 1, 1), ThrID `Layout<_1>` and A/B/C layouts
## `Layout<Shape<_1, _1>>` (one thread, one value). MoYe's `UniversalFMA`
## (`src/arch/mma/mma.jl`) has the same degenerate shape and its `fma!`
## is the plain arithmetic `d .= a .* b .+ c`.

import ../int_tuples
import ../layouts
import ../layout_constructors
import ../atoms

const UNIVERSAL_FMA_F32* = MmaAtom[
    typeof(make_layout((1, 1))), typeof(make_layout((1, 1))), typeof(make_layout((1, 1)))
  ](
    name: "UNIVERSAL_FMA_F32",
    mnk: (m: 1, n: 1, k: 1),
    aType: mdtF32, bType: mdtF32, cType: mdtF32,
    kind: bk_FMA,
    instr: "",
    aLayout: make_layout((1, 1)),
    bLayout: make_layout((1, 1)),
    cLayout: make_layout((1, 1)),
  )
  ## Scalar-FMA atom: 1 thread × 1 value × 1×1×1 tile, float32 operands
  ## and accumulator. The degenerate case that makes the tiled machinery
  ## express a plain FMA GEMM.

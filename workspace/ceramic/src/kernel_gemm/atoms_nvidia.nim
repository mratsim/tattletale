## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## NVIDIA tensor-core MMA atom instantiations (bkGPU_TensorCore).

import ../int_tuples
import ../layouts
import ../layout_constructors
import ../atoms

# ═════════════════════════════════════════════════════════════════════════
#  Reusable layout aliases
# ═════════════════════════════════════════════════════════════════════════

const
  SM80_16x8_Row* = make_layout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    ## (T32,V4) → (M16,N8) — C fragment of m16n8k8/m16n8k16 f16·bf16·tf32
  SM80_8x8_Row* = make_layout(((4, 8), 2), ((16, 1), 8))
    ## (T32,V2) → (M8,K8) — B fragment of m16n8k8
  SM80_16x8x8_A_TF32* = make_layout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    ## (T32,V4) → (M16,K8) — A fragment of m16n8k8 tf32
  SM80_16x8x8_B_TF32* = make_layout(((4, 8), 2), ((8, 1), 32))
    ## (T32,V2) → (N8,K8) — B fragment of m16n8k8 tf32

# ═════════════════════════════════════════════════════════════════════════
#  Atoms
# ═════════════════════════════════════════════════════════════════════════

const SM80_16x8x8_F32TF32TF32F32_TN* = MmaAtom[
    typeof(SM80_16x8x8_A_TF32), typeof(SM80_16x8x8_B_TF32), typeof(SM80_16x8_Row)
  ](
    name: "SM80_16x8x8_F32TF32TF32F32_TN",
    mnk: (m: 16, n: 8, k: 8),
    aType: mdtTF32, bType: mdtTF32, cType: mdtF32,
    kind: bkGPU_TensorCore,
    instr: "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
    aLayout: SM80_16x8x8_A_TF32,
    bLayout: SM80_16x8x8_B_TF32,
    cLayout: SM80_16x8_Row,
  )

## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Intel AMX tensor-core MMA atom instantiations (bkCPU_X86_AMX).
##
## AMX is the CPU tensor core: TILECFG + tmm tile registers (16×16 bf16 per
## tile), tdpbf16ps/tdpbssd instructions, operands loaded from PLAIN memory
## via _tile_loadd (no smem staging). The kind discriminates MEMORY CLASS.
## T=1 layouts: the whole tile is one "thread" (the core).
## Reference: ktransformers amx.hpp (GemmKernel224BF) + Intel AMX spec.

import ../int_tuples
import ../layouts
import ../layout_constructors
import ../atoms

# ═════════════════════════════════════════════════════════════════════════
#  AMX tile layouts — T=1, V = tile size
# ═════════════════════════════════════════════════════════════════════════

const
  AMX_1x256* = make_layout((1, 256), (0, 1))
    ## (T1,V256) → flat 256-element tile (16×16), single thread
  AMX_1x128* = make_layout((1, 128), (0, 1))
    ## (T1,V128) → 128-element panel (16×8)

# ═════════════════════════════════════════════════════════════════════════
#  Atoms
# ═════════════════════════════════════════════════════════════════════════

const AMX_16x16x16_TDPBF16PS* = MmaAtom[typeof(AMX_1x256), typeof(AMX_1x256), typeof(AMX_1x256)](
    name: "AMX_16x16x16_TDPBF16PS",
    mnk: (m: 16, n: 16, k: 16),          # tdpbf16ps: 16×16 tile, K=2 per instr
    aType: mdtBF16, bType: mdtBF16, cType: mdtF32,
    scaleMode: smSoftware, blockSize: 32, # per-32-block software scale
    sfaType: mdtF32, sfbType: mdtF32,
    kind: bkCPU_X86_AMX,
    instr: "tdpbf16ps",
    aLayout: AMX_1x256,
    bLayout: AMX_1x256,
    cLayout: AMX_1x256,
    scaleVec: sv1X,                       # unused — AMX scale is smSoftware
  )

const AMX_16x16x16_TDPBSSD* = MmaAtom[typeof(AMX_1x256), typeof(AMX_1x256), typeof(AMX_1x256)](
    name: "AMX_16x16x16_TDPBSSD",
    mnk: (m: 16, n: 16, k: 16),
    aType: mdtInt8, bType: mdtInt8, cType: mdtInt32,
    scaleMode: smSoftware, blockSize: 32,
    sfaType: mdtF32, sfbType: mdtF32,
    kind: bkCPU_X86_AMX,
    instr: "tdpbssd",
    aLayout: AMX_1x256,
    bLayout: AMX_1x256,
    cLayout: AMX_1x256,
    scaleVec: sv1X,
  )

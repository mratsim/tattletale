## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## ARM SIMD ukernel atom instantiations (bkCPU_SIMD).
## NEON SDOT / i8mm integer kernels (llama.cpp vec_dot paths).

import ./atoms

const ARM_NEON_SDOT_8x8x4* = MmaAtom[DefaultLayout2, DefaultLayout2, DefaultLayout2](
    name: "ARM_NEON_SDOT_8x8x4",
    mnk: (m: 8, n: 8, k: 4),
    aType: mdtInt8, bType: mdtInt8, cType: mdtInt32,
    scaleMode: smSoftware, blockSize: 32,
    sfaType: mdtF32, sfbType: mdtF32,
    kind: bkCPU_SIMD,
    isa: siSDOT, nbScalars: 8, nbVecsNr: 2,
    conversionPoint: cpPerBlock,
  )

const ARM_I8MM_16x16x8* = MmaAtom[DefaultLayout2, DefaultLayout2, DefaultLayout2](
    name: "ARM_I8MM_16x16x8",
    mnk: (m: 16, n: 16, k: 8),
    aType: mdtInt8, bType: mdtInt8, cType: mdtInt32,
    scaleMode: smSoftware, blockSize: 32,
    sfaType: mdtF32, sfbType: mdtF32,
    kind: bkCPU_SIMD,
    isa: siI8MM, nbScalars: 16, nbVecsNr: 4,
    conversionPoint: cpPerBlock,
  )

## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## x86 SIMD ukernel atom instantiations (bkCPU_SIMD).
##
## laser-style microkernels: isa + nbScalars + nbVecsNr + conversionPoint.
## AVX512 sgemm (laser): mr=14, nb_vecs_nr=2, nb_scalars=16 → 14×32 f32.

import ./atoms

const X86_AVX512_SGEMM_14x32* = MmaAtom[NoLayout, NoLayout, NoLayout](
    name: "X86_AVX512_SGEMM_14x32",
    mnk: (m: 14, n: 32, k: 1),
    aType: mdtF32, bType: mdtF32, cType: mdtF32,
    scaleMode: smNone, blockSize: 0,
    sfaType: mdtF32, sfbType: mdtF32,
    kind: bkCPU_SIMD,
    isa: siAVX512, nbScalars: 16, nbVecsNr: 2,
    conversionPoint: cpEndOfK,
  )

const X86_AVX512_VNNI_DPBSSD* = MmaAtom[NoLayout, NoLayout, NoLayout](
    name: "X86_AVX512_VNNI_DPBSSD",
    mnk: (m: 16, n: 16, k: 4),
    aType: mdtInt8, bType: mdtInt8, cType: mdtInt32,
    scaleMode: smSoftware, blockSize: 32,   # llama.cpp per-32-block quant scale
    sfaType: mdtF32, sfbType: mdtF32,
    kind: bkCPU_SIMD,
    isa: siVNNI, nbScalars: 16, nbVecsNr: 4,
    conversionPoint: cpPerBlock,
  )

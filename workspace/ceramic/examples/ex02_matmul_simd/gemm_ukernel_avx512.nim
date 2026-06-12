## SIMD micro-kernel — AVX-512 (float32)
##
## Tile: mr=14, nr=32 (four ZMM vectors per k-step)

{.push localpassC: "-mavx512f".}

import std/math
import ./simd

proc gemm_ukernel_avx512*[MR, NR: static int](
    packA, packB: ptr UncheckedArray[float32];
    AB: var array[MR, array[NR, float32]];
    kc: int) =
  const NbVecs = NR div 16  # 2 vectors per row (32 float32 = 2×m512)
  var ABv {.noInit.}: array[MR, array[NbVecs, m512]]
  for i in 0 ..< MR:
    for j in 0 ..< NbVecs:
      ABv[i][j] = mm512_setzero_ps()

  for k in 0 ..< kc:
    let Bv0 = mm512_loadu_ps(cast[ptr float32](packB[k * NR + 0].addr))
    let Bv1 = mm512_loadu_ps(cast[ptr float32](packB[k * NR + 16].addr))
    for i in 0 ..< MR:
      let ai = mm512_set1_ps(packA[k * MR + i])
      ABv[i][0] = mm512_fmadd_ps(ai, Bv0, ABv[i][0])
      ABv[i][1] = mm512_fmadd_ps(ai, Bv1, ABv[i][1])

  for i in 0 ..< MR:
    mm512_storeu_ps(cast[ptr float32](AB[i][0].addr), ABv[i][0])
    mm512_storeu_ps(cast[ptr float32](AB[i][16].addr), ABv[i][1])

{.pop.}

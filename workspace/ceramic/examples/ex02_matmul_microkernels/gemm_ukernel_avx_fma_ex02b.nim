## SIMD micro-kernel — AVX+FMA (float32) — ex02b layout-algebra variant
##
## Uses the `localpassC` pragma to enable AVX+FMA only in this file.
## Tile: mr=6, nr=16 (two AVX vectors per k-step)
## Simple k-loop, no unrolling, unaligned loads.

{.push localpassC: "-mavx -mfma".}

import workspace/cpuplatforms/x86/simd_x86

proc gemm_ukernel_avx_fma*[MR, NR: static int](
    packA, packB: ptr UncheckedArray[float32];
    AB: var array[MR, array[NR, float32]];
    kc: int) =
  ## AVX+FMA micro-kernel for float32.
  ## Simple k-loop, unaligned loads.
  const NbVecs = NR div 8   # 2 vectors per row (16 float32 = 2×m256)
  var ABv {.noInit.}: array[MR, array[NbVecs, m256]]
  for i in 0 ..< MR:
    for j in 0 ..< NbVecs:
      ABv[i][j] = mm256_setzero_ps()

  for k in 0 ..< kc:
    # Load B[k*NR .. k*NR+15] into 2 vectors
    let Bv0 = mm256_loadu_ps(cast[ptr float32](packB[k * NR + 0].addr))
    let Bv1 = mm256_loadu_ps(cast[ptr float32](packB[k * NR + 8].addr))
    for i in 0 ..< MR:
      let ai = mm256_set1_ps(packA[k * MR + i])
      ABv[i][0] = mm256_fmadd_ps(ai, Bv0, ABv[i][0])
      ABv[i][1] = mm256_fmadd_ps(ai, Bv1, ABv[i][1])

  for i in 0 ..< MR:
    mm256_storeu_ps(cast[ptr float32](AB[i][0].addr), ABv[i][0])
    mm256_storeu_ps(cast[ptr float32](AB[i][8].addr), ABv[i][1])

{.pop.}

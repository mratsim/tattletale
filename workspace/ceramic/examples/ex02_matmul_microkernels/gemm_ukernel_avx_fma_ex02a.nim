## SIMD micro-kernel — AVX+FMA (float32) — ex02a hand-tuned variant
##
## Uses the `localpassC` pragma to enable AVX+FMA only in this file.
## Tile: mr=6, nr=16 (two AVX vectors per k-step)
## Manually unrolled MR loop with interleaved broadcasts for register rotation.
## K-loop unrolled by 2 with prefetch of next B tile.

{.push localpassC: "-mavx -mfma".}

import workspace/cpuplatforms/x86/simd_x86

proc gemm_ukernel_avx_fma*[MR, NR: static int](
    packA, packB: ptr UncheckedArray[float32];
    AB: var array[MR, array[NR, float32]];
    kc: int) =
  ## AVX+FMA micro-kernel for float32.
  ## K-loop unrolled by 2 with prefetch.
  const NbVecs = NR div 8
  var ABv {.noInit.}: array[MR, array[NbVecs, m256]]
  for i in 0 ..< MR:
    for j in 0 ..< NbVecs:
      ABv[i][j] = mm256_setzero_ps()

  # Unrolled k-loop by 2
  var k = 0
  let kc_aligned = (kc div 2) * 2
  while k < kc_aligned:
    let Bv0 = mm256_load_ps(cast[ptr float32](packB[k * NR + 0].addr))
    let Bv1 = mm256_load_ps(cast[ptr float32](packB[k * NR + 8].addr))
    let a0 = mm256_set1_ps(packA[k * MR + 0])
    ABv[0][0] = mm256_fmadd_ps(a0, Bv0, ABv[0][0])
    ABv[0][1] = mm256_fmadd_ps(a0, Bv1, ABv[0][1])
    let a1 = mm256_set1_ps(packA[k * MR + 1])
    ABv[1][0] = mm256_fmadd_ps(a1, Bv0, ABv[1][0])
    ABv[1][1] = mm256_fmadd_ps(a1, Bv1, ABv[1][1])
    let a2 = mm256_set1_ps(packA[k * MR + 2])
    ABv[2][0] = mm256_fmadd_ps(a2, Bv0, ABv[2][0])
    ABv[2][1] = mm256_fmadd_ps(a2, Bv1, ABv[2][1])
    let a3 = mm256_set1_ps(packA[k * MR + 3])
    ABv[3][0] = mm256_fmadd_ps(a3, Bv0, ABv[3][0])
    ABv[3][1] = mm256_fmadd_ps(a3, Bv1, ABv[3][1])
    let a4 = mm256_set1_ps(packA[k * MR + 4])
    ABv[4][0] = mm256_fmadd_ps(a4, Bv0, ABv[4][0])
    ABv[4][1] = mm256_fmadd_ps(a4, Bv1, ABv[4][1])
    let a5 = mm256_set1_ps(packA[k * MR + 5])
    ABv[5][0] = mm256_fmadd_ps(a5, Bv0, ABv[5][0])
    ABv[5][1] = mm256_fmadd_ps(a5, Bv1, ABv[5][1])
    # Second k-iteration (unrolled)
    builtin_prefetch(addr packB[(k+2) * NR + 0], 0, 1)
    builtin_prefetch(addr packB[(k+2) * NR + 8], 0, 1)
    let Bv0b = mm256_load_ps(cast[ptr float32](packB[(k+1) * NR + 0].addr))
    let Bv1b = mm256_load_ps(cast[ptr float32](packB[(k+1) * NR + 8].addr))
    let a0b = mm256_set1_ps(packA[(k+1) * MR + 0])
    ABv[0][0] = mm256_fmadd_ps(a0b, Bv0b, ABv[0][0])
    ABv[0][1] = mm256_fmadd_ps(a0b, Bv1b, ABv[0][1])
    let a1b = mm256_set1_ps(packA[(k+1) * MR + 1])
    ABv[1][0] = mm256_fmadd_ps(a1b, Bv0b, ABv[1][0])
    ABv[1][1] = mm256_fmadd_ps(a1b, Bv1b, ABv[1][1])
    let a2b = mm256_set1_ps(packA[(k+1) * MR + 2])
    ABv[2][0] = mm256_fmadd_ps(a2b, Bv0b, ABv[2][0])
    ABv[2][1] = mm256_fmadd_ps(a2b, Bv1b, ABv[2][1])
    let a3b = mm256_set1_ps(packA[(k+1) * MR + 3])
    ABv[3][0] = mm256_fmadd_ps(a3b, Bv0b, ABv[3][0])
    ABv[3][1] = mm256_fmadd_ps(a3b, Bv1b, ABv[3][1])
    let a4b = mm256_set1_ps(packA[(k+1) * MR + 4])
    ABv[4][0] = mm256_fmadd_ps(a4b, Bv0b, ABv[4][0])
    ABv[4][1] = mm256_fmadd_ps(a4b, Bv1b, ABv[4][1])
    let a5b = mm256_set1_ps(packA[(k+1) * MR + 5])
    ABv[5][0] = mm256_fmadd_ps(a5b, Bv0b, ABv[5][0])
    ABv[5][1] = mm256_fmadd_ps(a5b, Bv1b, ABv[5][1])
    k += 2
  while k < kc:
    let Bv0 = mm256_load_ps(cast[ptr float32](packB[k * NR + 0].addr))
    let Bv1 = mm256_load_ps(cast[ptr float32](packB[k * NR + 8].addr))
    let a0 = mm256_set1_ps(packA[k * MR + 0])
    ABv[0][0] = mm256_fmadd_ps(a0, Bv0, ABv[0][0])
    ABv[0][1] = mm256_fmadd_ps(a0, Bv1, ABv[0][1])
    let a1 = mm256_set1_ps(packA[k * MR + 1])
    ABv[1][0] = mm256_fmadd_ps(a1, Bv0, ABv[1][0])
    ABv[1][1] = mm256_fmadd_ps(a1, Bv1, ABv[1][1])
    let a2 = mm256_set1_ps(packA[k * MR + 2])
    ABv[2][0] = mm256_fmadd_ps(a2, Bv0, ABv[2][0])
    ABv[2][1] = mm256_fmadd_ps(a2, Bv1, ABv[2][1])
    let a3 = mm256_set1_ps(packA[k * MR + 3])
    ABv[3][0] = mm256_fmadd_ps(a3, Bv0, ABv[3][0])
    ABv[3][1] = mm256_fmadd_ps(a3, Bv1, ABv[3][1])
    let a4 = mm256_set1_ps(packA[k * MR + 4])
    ABv[4][0] = mm256_fmadd_ps(a4, Bv0, ABv[4][0])
    ABv[4][1] = mm256_fmadd_ps(a4, Bv1, ABv[4][1])
    let a5 = mm256_set1_ps(packA[k * MR + 5])
    ABv[5][0] = mm256_fmadd_ps(a5, Bv0, ABv[5][0])
    ABv[5][1] = mm256_fmadd_ps(a5, Bv1, ABv[5][1])
    k += 1

  for i in 0 ..< MR:
    mm256_storeu_ps(cast[ptr float32](AB[i][0].addr), ABv[i][0])
    mm256_storeu_ps(cast[ptr float32](AB[i][8].addr), ABv[i][1])

{.pop.}

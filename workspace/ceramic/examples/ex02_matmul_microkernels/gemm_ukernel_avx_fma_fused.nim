## SIMD micro-kernel with fused epilogue — AVX+FMA (float32)
##
## Fuses the micro-kernel and C-store into one pass.
## For use by ex02a (hand-tuned) which bypasses the scalar AB array.
## K-loop not unrolled (the fused variant is for demonstration/alternative path).

{.push localpassC: "-mavx -mfma".}

import workspace/cpuplatforms/x86/simd_x86

proc gemm_ukernel_avx_fma_fused*[MR, NR: static int](
    packA, packB: ptr UncheckedArray[float32];
    C: ptr UncheckedArray[float32]; cStride: int;
    kc: int; alpha, beta: float32) =
  ## Manually unrolled for MR=6 with interleaved broadcasts.
  const NbVecs = NR div 8
  var ABv {.noInit.}: array[MR, array[NbVecs, m256]]
  for i in 0 ..< MR:
    for j in 0 ..< NbVecs:
      ABv[i][j] = mm256_setzero_ps()

  for k in 0 ..< kc:
    let Bv0 = mm256_load_ps(cast[ptr float32](packB[k * NR + 0].addr))
    let Bv1 = mm256_load_ps(cast[ptr float32](packB[k * NR + 8].addr))

    # Manually unrolled MR=6 with interleaved broadcasts
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

  # Fused epilogue: write C directly from SIMD regs
  if beta == 0.0'f32:
    for i in 0 ..< MR:
      mm256_storeu_ps(cast[ptr float32](C[i * cStride + 0].addr),
        if alpha == 1.0'f32: ABv[i][0] else: mm256_mul_ps(ABv[i][0], mm256_set1_ps(alpha)))
      mm256_storeu_ps(cast[ptr float32](C[i * cStride + 8].addr),
        if alpha == 1.0'f32: ABv[i][1] else: mm256_mul_ps(ABv[i][1], mm256_set1_ps(alpha)))
  elif beta == 1.0'f32:
    let alphaV = mm256_set1_ps(alpha)
    for i in 0 ..< MR:
      let Cv0 = mm256_load_ps(cast[ptr float32](C[i * cStride + 0].addr))
      let Cv1 = mm256_load_ps(cast[ptr float32](C[i * cStride + 8].addr))
      mm256_storeu_ps(cast[ptr float32](C[i * cStride + 0].addr), mm256_fmadd_ps(ABv[i][0], alphaV, Cv0))
      mm256_storeu_ps(cast[ptr float32](C[i * cStride + 8].addr), mm256_fmadd_ps(ABv[i][1], alphaV, Cv1))
  else:
    let alphaV = mm256_set1_ps(alpha)
    let betaV = mm256_set1_ps(beta)
    for i in 0 ..< MR:
      let Cv0 = mm256_load_ps(cast[ptr float32](C[i * cStride + 0].addr))
      let Cv1 = mm256_load_ps(cast[ptr float32](C[i * cStride + 8].addr))
      let Cv0_s = mm256_mul_ps(Cv0, betaV)
      let Cv1_s = mm256_mul_ps(Cv1, betaV)
      mm256_storeu_ps(cast[ptr float32](C[i * cStride + 0].addr), mm256_fmadd_ps(ABv[i][0], alphaV, Cv0_s))
      mm256_storeu_ps(cast[ptr float32](C[i * cStride + 8].addr), mm256_fmadd_ps(ABv[i][1], alphaV, Cv1_s))

{.pop.}

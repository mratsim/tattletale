## Seeded fuzzer for the SME2 GEMM micro-kernels in `gemm_ukernel_arm64_sme2.nim`.
##
## Stress-tests the four kernels plus the `gemmUkernelSme` dispatcher against a
## scalar reference: seeded random A/B, kc oddments (the `% 4` blocks/tail and
## `% 16` A-pack edges), every alpha/beta combination, ReLU on and off.
##
## aarch64-only: the kernels require ARMv9.2 with FEAT_SME2 and the
## `-march=armv9-a+sme2` flag the kernel module sets via localpassC.
##
## Usage:
##   nim cpp -r -d:release --hints:off --warnings:off \
##     --outdir:build/wip/fuzzer --nimcache:nimcache/wip/fuzzer \
##     workspace/ceramic/examples/ex02_matmul_microkernels/gemm_ukernel_arm64_sme2_fuzzer.nim

import std/[random, strformat]
import workspace/ceramic/examples/ex02_matmul_microkernels/gemm_ukernel_arm64_sme2

proc gemmReference(
    M, N, K: int, A, B: ptr UncheckedArray[float32],
    alpha, beta: float32, relu: bool,
    C: var openArray[float32]) =
  ## Scalar reference: `C[i*N+j] = beta*C[i*N+j] + alpha*f(acc)` with
  ## `f` = identity or ReLU and `acc = Σ_k A[i*K+k] * B[k*N+j]`.
  ## A `beta == 0` call writes `alpha*f(acc)` without reading C, matching
  ## the kernels' betaZero path.
  ##
  ## Expected input:
  ##   - A: M×K row-major f32, B: K×N row-major f32
  ##   - C: M×N row-major f32, prior values read when `beta != 0`
  ##
  ## Output: C updated in place.
  for i in 0 ..< M:
    for j in 0 ..< N:
      var acc = 0.0'f32
      for k in 0 ..< K:
        acc += A[i * K + k] * B[k * N + j]
      let f = if relu and acc < 0.0'f32: 0.0'f32 else: acc
      let ci = i * N + j
      C[ci] = if beta == 0.0'f32: alpha * f
              else: beta * C[ci] + alpha * f

proc packTiles(
    MR, NR, K: int, A, B: ptr UncheckedArray[float32],
    packA, packB: ptr UncheckedArray[float32]) =
  ## Packs A/B into the kernels' `(ir, kc, MR/NR)` layout, the
  ## `gemmUkernelSme` documented contract: `packA[k*MR + r] = A[r*K + k]`,
  ## `packB[k*NR + j] = B[k*NR + j]`.
  ##
  ## Expected input:
  ##   - A: MR×K row-major f32, B: K×NR row-major f32
  ##   - packA: K*MR f32, packB: K*NR f32
  ##
  ## Output: packed buffers filled.
  for k in 0 ..< K:
    for r in 0 ..< MR:
      packA[k * MR + r] = A[r * K + k]
    for j in 0 ..< NR:
      packB[k * NR + j] = B[k * NR + j]

proc closeEnough(C, refC: openArray[float32]): tuple[ok: bool, maxAbs, maxRel: float32] =
  ## Kernel-vs-reference comparison: `maxAbs <= 1e-4` and `maxRel <= 1e-1`
  ## (per-element relative with a 1e-12 floor).
  ##
  ## atol 1e-4 is the primary gate: the full grid's worst absolute error
  ## is 8.6e-6 (f32 accumulation order, verified 2 ulps from f64-exact),
  ## a 12x margin. rtol 1e-1 is secondary: per-cell relative amplifies
  ## on near-zero cells (a 1-ulp diff on a 2e-4 cancellation cell reads
  ## ~1e-2), so only genuinely wrong cells (real bugs) trip it.
  doAssert C.len == refC.len
  var maxAbs = 0.0'f32
  var maxRel = 0.0'f32
  for i in 0 ..< C.len:
    let d = abs(C[i] - refC[i])
    maxAbs = max(maxAbs, d)
    maxRel = max(maxRel, d / max(abs(C[i]), max(abs(refC[i]), 1e-12'f32)))
  (maxAbs <= 1e-4'f32 and maxRel <= 1e-1'f32, maxAbs, maxRel)

proc main() =
  const
    Ks = [1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65,
          127, 128, 129]
    AlphaBeta = [(1.0'f32, 0.0'f32), (0.5'f32, 2.0'f32), (0.0'f32, 0.0'f32),
                 (1.0'f32, 1.0'f32), (2.0'f32, 0.5'f32), (-0.5'f32, 1.5'f32)]
  var totalCases = 0
  var worstAbs = 0.0'f32
  var worstRel = 0.0'f32
  var worstDesc = ""
  var worstSeed = -1

  for seed in 0 ..< 8:
    var rng = initRand(seed)
    var seedCases = 0
    var seedWorst = 0.0'f32
    for K in Ks:
      # 16×16 AB-store kernel through the dispatcher: AB = A·B.
      block:
        const MR = 16
        const NR = 16
        var A = newSeq[float32](MR * K)
        var B = newSeq[float32](K * NR)
        for i in 0 ..< A.len: A[i] = rng.rand(2.0'f32) - 1.0'f32
        for i in 0 ..< B.len: B[i] = rng.rand(2.0'f32) - 1.0'f32
        var packA = newSeq[float32](K * MR)
        var packB = newSeq[float32](K * NR)
        packTiles(MR, NR, K,
                  cast[ptr UncheckedArray[float32]](addr A[0]),
                  cast[ptr UncheckedArray[float32]](addr B[0]),
                  cast[ptr UncheckedArray[float32]](addr packA[0]),
                  cast[ptr UncheckedArray[float32]](addr packB[0]))
        var AB: array[MR, array[NR, float32]]
        gemmUkernelSme[MR, NR](
          cast[ptr UncheckedArray[float32]](addr packA[0]),
          cast[ptr UncheckedArray[float32]](addr packB[0]),
          AB, K)
        var refC = newSeq[float32](MR * NR)
        gemmReference(MR, NR, K,
                      cast[ptr UncheckedArray[float32]](addr A[0]),
                      cast[ptr UncheckedArray[float32]](addr B[0]),
                      1.0'f32, 0.0'f32, false, refC)
        var flat: array[MR * NR, float32]
        for i in 0 ..< MR:
          for j in 0 ..< NR:
            flat[i * NR + j] = AB[i][j]
        let (ok, maxAbs, maxRel) = closeEnough(flat, refC)
        totalCases.inc
        seedCases.inc
        seedWorst = max(seedWorst, maxAbs)
        doAssert ok, &"seed {seed} smeGemmUkernel16x16 K={K}: maxAbs={maxAbs:.3e} maxRel={maxRel:.3e}"
        if maxAbs > worstAbs:
          worstAbs = maxAbs
          worstRel = maxRel
          worstDesc = "smeGemmUkernel16x16"
          worstSeed = seed
      # 32×32 AB-store kernel through the dispatcher: AB = A·B.
      block:
        const MR = 32
        const NR = 32
        var A = newSeq[float32](MR * K)
        var B = newSeq[float32](K * NR)
        for i in 0 ..< A.len: A[i] = rng.rand(2.0'f32) - 1.0'f32
        for i in 0 ..< B.len: B[i] = rng.rand(2.0'f32) - 1.0'f32
        var packA = newSeq[float32](K * MR)
        var packB = newSeq[float32](K * NR)
        packTiles(MR, NR, K,
                  cast[ptr UncheckedArray[float32]](addr A[0]),
                  cast[ptr UncheckedArray[float32]](addr B[0]),
                  cast[ptr UncheckedArray[float32]](addr packA[0]),
                  cast[ptr UncheckedArray[float32]](addr packB[0]))
        var AB: array[MR, array[NR, float32]]
        gemmUkernelSme[MR, NR](
          cast[ptr UncheckedArray[float32]](addr packA[0]),
          cast[ptr UncheckedArray[float32]](addr packB[0]),
          AB, K)
        var refC = newSeq[float32](MR * NR)
        gemmReference(MR, NR, K,
                      cast[ptr UncheckedArray[float32]](addr A[0]),
                      cast[ptr UncheckedArray[float32]](addr B[0]),
                      1.0'f32, 0.0'f32, false, refC)
        var flat: array[MR * NR, float32]
        for i in 0 ..< MR:
          for j in 0 ..< NR:
            flat[i * NR + j] = AB[i][j]
        let (ok, maxAbs, maxRel) = closeEnough(flat, refC)
        totalCases.inc
        seedCases.inc
        seedWorst = max(seedWorst, maxAbs)
        doAssert ok, &"seed {seed} smeGemmUkernel32x32 K={K}: maxAbs={maxAbs:.3e} maxRel={maxRel:.3e}"
        if maxAbs > worstAbs:
          worstAbs = maxAbs
          worstRel = maxRel
          worstDesc = "smeGemmUkernel32x32"
          worstSeed = seed
      # Fused kernels: alpha/beta/ReLU applied in streaming mode.
      for (alpha, beta) in AlphaBeta:
        for relu in [0, 1]:
          block:
            const MR = 32
            const NR = 32
            var A = newSeq[float32](MR * K)
            var B = newSeq[float32](K * NR)
            for i in 0 ..< A.len: A[i] = rng.rand(2.0'f32) - 1.0'f32
            for i in 0 ..< B.len: B[i] = rng.rand(2.0'f32) - 1.0'f32
            var packA = newSeq[float32](K * MR)
            var packB = newSeq[float32](K * NR)
            packTiles(MR, NR, K,
                      cast[ptr UncheckedArray[float32]](addr A[0]),
                      cast[ptr UncheckedArray[float32]](addr B[0]),
                      cast[ptr UncheckedArray[float32]](addr packA[0]),
                      cast[ptr UncheckedArray[float32]](addr packB[0]))
            var C = newSeq[float32](MR * NR)
            for i in 0 ..< C.len: C[i] = rng.rand(2.0'f32) - 1.0'f32
            var refC = C
            smeGemmUkernel32x32Epi(
              cast[ptr float32](addr packA[0]),
              cast[ptr float32](addr packB[0]),
              addr C[0], cint(NR), cint(K),
              alpha, beta, cint(relu),
              cint(alpha == 1.0'f32), cint(beta == 0.0'f32),
              cint(beta == 1.0'f32))
            gemmReference(MR, NR, K,
                          cast[ptr UncheckedArray[float32]](addr A[0]),
                          cast[ptr UncheckedArray[float32]](addr B[0]),
                          alpha, beta, relu == 1, refC)
            let (ok, maxAbs, maxRel) = closeEnough(C, refC)
            totalCases.inc
            seedCases.inc
            seedWorst = max(seedWorst, maxAbs)
            doAssert ok, &"seed {seed} smeGemmUkernel32x32Epi K={K} " &
              &"alpha={alpha} beta={beta} relu={relu}: " &
              &"maxAbs={maxAbs:.3e} maxRel={maxRel:.3e}"
            if maxAbs > worstAbs:
              worstAbs = maxAbs
              worstRel = maxRel
              worstDesc = "smeGemmUkernel32x32Epi"
              worstSeed = seed
          block:
            const MR = 32
            const NR = 32
            var A = newSeq[float32](MR * K)
            var B = newSeq[float32](K * NR)
            for i in 0 ..< A.len: A[i] = rng.rand(2.0'f32) - 1.0'f32
            for i in 0 ..< B.len: B[i] = rng.rand(2.0'f32) - 1.0'f32
            var packA = newSeq[float32](K * MR)
            var packB = newSeq[float32](K * NR)
            packTiles(MR, NR, K,
                      cast[ptr UncheckedArray[float32]](addr A[0]),
                      cast[ptr UncheckedArray[float32]](addr B[0]),
                      cast[ptr UncheckedArray[float32]](addr packA[0]),
                      cast[ptr UncheckedArray[float32]](addr packB[0]))
            var C = newSeq[float32](MR * NR)
            for i in 0 ..< C.len: C[i] = rng.rand(2.0'f32) - 1.0'f32
            var refC = C
            smeGemmUkernel32x32EpiDv(
              cast[ptr float32](addr packA[0]),
              cast[ptr float32](addr packB[0]),
              addr C[0], cint(NR), cint(K),
              alpha, beta, cint(relu),
              cint(alpha == 1.0'f32), cint(beta == 0.0'f32),
              cint(beta == 1.0'f32))
            gemmReference(MR, NR, K,
                          cast[ptr UncheckedArray[float32]](addr A[0]),
                          cast[ptr UncheckedArray[float32]](addr B[0]),
                          alpha, beta, relu == 1, refC)
            let (ok, maxAbs, maxRel) = closeEnough(C, refC)
            totalCases.inc
            seedCases.inc
            seedWorst = max(seedWorst, maxAbs)
            doAssert ok, &"seed {seed} smeGemmUkernel32x32EpiDv K={K} " &
              &"alpha={alpha} beta={beta} relu={relu}: " &
              &"maxAbs={maxAbs:.3e} maxRel={maxRel:.3e}"
            if maxAbs > worstAbs:
              worstAbs = maxAbs
              worstRel = maxRel
              worstDesc = "smeGemmUkernel32x32EpiDv"
              worstSeed = seed
    echo &"  seed {seed}: {seedCases} cases, worst maxAbs {seedWorst:.3e}"

  echo ""
  echo &"Fuzzer summary: {totalCases} cases, all passed."
  echo &"  worst maxAbs {worstAbs:.3e} (maxRel {worstRel:.3e}) in {worstDesc}, seed {worstSeed}"
  echo "  atol 1e-4 (primary), rtol 1e-1 (near-zero cells) vs the scalar reference. Deterministic (seeded)."

main()

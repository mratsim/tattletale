## Shared oracle harness for the manual_*_cuda GPU tests.
##
## The per-arch test files (manual_sm80_*, manual_sm86_*, ...) differ only in
## the atom + the driver kernel; the fixture generation, the tf32 reference
## oracle, the execute/verify loop and the report are factored here so each
## new SM is just `const atom = ...` + the driver funcs + the cuda block +
## a 4-line main.
##
## Kernel-name conventions (the test files' cuda blocks must name them so):
##   microtile: mmaMicrotileKernel / mmaMicrotileExplicitKernel (one module)
##   ukernel:   gemmUkernelKernel
##   tiled:     gemmTiledKernel


import std/[random, strformat, math, typetraits]
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/tensors
import workspace/ceramic/src/atoms
import workspace/ceramic/src/kernel_gemm_gpu
import workspace/crucible

# ═════════════════════════════════════════════════════════════════════════
#  Low-level primitives — the elementwise/compare helpers the oracles and
#  drivers build on (kept at the top: no forward-declaration dance).
# ═════════════════════════════════════════════════════════════════════════

proc fmaf(a, b, c: float32): float32 {.importc: "fmaf", header: "<math.h>".}
proc allClose(testC, refC: openArray[float32];
               M, N: int;
               context: string;
               relTol = 1e-4'f32; absTol = 1e-4'f32) =
  ## Element-wise numpy.allclose check: |test - ref| <= absTol + relTol·|ref|
  ## (single condition, NaN never equal) — loud doAssert on the first
  ## failure. Defaults are loose for the tf32 oracle domain; the current
  ## gemm tests pass with maxAbsErr = 0 (fixture-exact), but the assertion
  ## is a tolerance — tighten per test if needed.
  var maxAbsErr: float32 = 0.0'f32
  for m in 0 ..< M:
    for n in 0 ..< N:
      let refVal = refC[m + n * M]
      let testVal = testC[m + n * M]
      let absErr = abs(testVal - refVal)
      if absErr > maxAbsErr: maxAbsErr = absErr
      doAssert absErr <= absTol + relTol * abs(refVal),
        &"{context} [{m},{n}]: got {testVal}, expected {refVal}, absErr={absErr}"
  echo &"    PASS (maxAbsErr={maxAbsErr:.2e})"

# ═════════════════════════════════════════════════════════════════════════
#  Reference oracles — the naive whole-tile GEMMs the GPU kernels are
#  tested against. gemm_ref is the layout-generic outer product over
#  TensorViews; gemm_tf32_ref is the flat-array tf32-exact oracle
#  (experiment_testutils convention).
# ═════════════════════════════════════════════════════════════════════════
#  gemm_ref(C, A, B) — whole-tile reference (outer product)
# ═════════════════════════════════════════════════════════════════════════

proc gemm_ref*[T, ShA, StA, ShB, StB, ShC, StC](
    C: var (TensorView[T, ShC, StC] or Tensor[T, ShC, StC]),
    A: TensorView[T, ShA, StA] or Tensor[T, ShA, StA],
    B: TensorView[T, ShB, StB] or Tensor[T, ShB, StB]) =
  ## Reference fragment gemm: C[m,n] += A[m,k] * B[n,k] (outer product).
  ## Correctness oracle for the GPU kernels — not a performance kernel.
  ## (Moved here from kernel_gemm_gpu.nim 2026-08-10 with the
  ## gemm_fragment → gemm_atom rename; now a proc, not a template.)
  const
    M = ShC.default[0]
    N = ShC.default[1]
    K = ShA.default[1]
  when typeof(ShA.default[0]) isnot typeof(M):
    {.error: "gemm_ref: A mode 0 (M) != C mode 0".}
  when typeof(ShB.default[0]) isnot typeof(N):
    {.error: "gemm_ref: B mode 0 (N) != C mode 1".}
  when typeof(ShA.default[1]) isnot typeof(K):
    {.error: "gemm_ref: A mode 1 (K) != B mode 1".}
  for k in 0 ..< K:
    for m in 0 ..< M:
      for n in 0 ..< N:
        C[m, n] += A[m, k] * B[n, k]

# ═════════════════════════════════════════════════════════════════════════
#  tf32 fixture machinery — the bit-exact test domain (0..15, tf32ified)
# ═════════════════════════════════════════════════════════════════════════

func tf32ify(x: float32): uint32 =
  ## f32 → tf32 bit pattern: truncate the low 13 mantissa bits (round-toward-zero).
  ## Deliberately NOT cvt.rna (round-to-nearest-away, CUTLASS' f32→tf32
  ## conversion): the two agree on the small-integer fixture domain, and RZ
  ## keeps the bit pattern a pure mask. Revisit (cvt.rna) when a production
  ## f32→tf32 conversion path is added.
  (cast[uint32](x)) and 0xFFFFE000'u32

proc tf32Fixture(rng: var Rand; M, K: int): seq[uint32] =
  ## A random (M, K) col-major tf32 fixture in the f32 domain -15..15.
  ## The symmetric domain is pinned so every |product| (≤ 15·15 = 225),
  ## |partial sum| (≤ K·225) and epilogue is exactly representable in f32's
  ## 24-bit mantissa — this is what makes the gemm tests bit-exact regardless
  ## of the mma pipe's internal accumulation order (see gemm_tf32_ref).
  ## Negatives are required so epilogues with a clamp (EpiReLU) see both
  ## signs of the accumulator; a dropped clamp would then fail the test.
  doAssert K * 15 * 15 < 1 shl 24,
    "tf32Fixture: K·15² ≥ 2^24 — partial sums leave the f32 exact-representable" &
    " domain; the oracle would no longer be bit-exact"
  result = newSeq[uint32](M * K)
  for i in 0 ..< result.len:
    result[i] = tf32ify(float32(rng.rand(-15 .. 15)))

proc gemm_tf32_ref*(C: var openArray[float32];
                    A, B: openArray[uint32];
                    M, N, K: int;
                    cInit: float32) =
  ## C[m,n] = cInit + Σ_k tf32(A[m,k]) · tf32(B[n,k]) — the naive O(M·N·K)
  ## triple loop (experiment_testutils convention).
  ##
  ## Bit-exactness does NOT come from the fmaf/ascending-k accumulation:
  ## mma.sync's internal K=8 dot-product order is an undocumented hardware
  ## detail. It comes from the fixture domain (0..15, tf32ified): every
  ## product and partial sum is exactly representable in f32, so ANY
  ## accumulation order — fmaf, mul+add, or the mma adder tree — yields the
  ## identical exact result. fmaf is used merely for convenience; do NOT
  ## extend this oracle to non-exact ranges (e.g. random floats) without
  ## switching the comparison to a tolerance check.
  for m in 0 ..< M:
    for n in 0 ..< N:
      var sum = cInit
      for k in 0 ..< K:
        let av = cast[float32](A[m + k * M])
        let bv = cast[float32](B[n + k * N])
        sum = fmaf(av, bv, sum)
      C[m + n * M] = sum

# ═════════════════════════════════════════════════════════════════════════
#  Backend engines — the HwEngine API
#
# Kernel-name conventions (the test files' cuda/opencl blocks must name them so):
#   microtile: mmaMicrotileKernel / mmaMicrotileExplicitKernel (one module)
#   ukernel:   gemmUkernelKernel
#   tiled:     gemmTiledKernel
#
# ═════════════════════════════════════════════════════════════════════════
#  Per-kernel test drivers
# ═════════════════════════════════════════════════════════════════════════

proc microtileFixtures(atom: static MmaAtom; rng: var Rand): tuple[A, B: seq[uint32]] =
  ## One microtile trial's tf32 fixtures — the shared oracle. The NVRTC and
  ## OpenCL drivers both call this (and verifyMicrotile); they differ ONLY
  ## in the launch. The _cuda/_opencl files must not touch tf32Fixture or
  ## gemm_tf32_ref directly.
  const M = atom.mnk.m
  const N = atom.mnk.n
  const K = atom.mnk.k
  result.A = tf32Fixture(rng, M, K)
  result.B = tf32Fixture(rng, N, K)

proc verifyMicrotile(atom: static MmaAtom; trial: int;
                      gpuC: openArray[float32]; A, B: openArray[uint32];
                      cInit: float32; context: string) =
  ## One microtile trial's reference + comparison — the shared oracle.
  ## gpuC vs gemm_tf32_ref via allClose (bit-exact in practice: the fixture
  ## domain is exact-representable).
  const M = atom.mnk.m
  const N = atom.mnk.n
  const K = atom.mnk.k
  var refC = newSeq[float32](M * N)
  refC.gemm_tf32_ref(A, B, M, N, K, cInit)
  allClose(gpuC, refC, M, N, context)

proc testMicrotile*[E](engine: var E; atom: static MmaAtom; label: string) =
  ## One register-level MMA on the atom's own tile: C(M×N) = A(M×K)·B(N×K).
  ## Runs both the 4-arg (in-place) and 5-arg (explicit, cFrag = 1.0) forms,
  ## 16 trials each, bit-exact vs the tf32 reference.
  const
    M = atom.mnk.m
    N = atom.mnk.n
  var rng = initRand(0xC0FFEE)
  for trial in 0 ..< 16:
    let (A, B) = microtileFixtures(atom, rng)

    # in-place (4-arg)
    var gpuC = newSeq[float32](M * N)
    engine.run<<(1, toIntVal(atom.threadCount(opA)))>>("mmaMicrotileKernel", gpuC, (A, B))
    verifyMicrotile(atom, trial, gpuC, A, B, 0.0'f32, "in-place trial " & $trial)

    # explicit-output (5-arg), cFrag = 1.0
    var gpuD = newSeq[float32](M * N)
    engine.run<<(1, toIntVal(atom.threadCount(opA)))>>("mmaMicrotileExplicitKernel", gpuD, (A, B))
    verifyMicrotile(atom, trial, gpuD, A, B, 1.0'f32, "explicit trial " & $trial)

  echo "  OK: m16n8k8 tf32 microtile matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 16 trials, in-place + explicit)"

proc testUkernel*[E](engine: var E; atom: static MmaAtom; label: string) =
  ## The k-loop microkernel: C(M×N) = A(M, 2·K)·B(N, 2·K) — two k_blocks,
  ## 16 trials vs the tf32 reference (exact on the tf32-exact fixture).
  const
    M = atom.mnk.m
    N = atom.mnk.n
    Ktotal = 2 * atom.mnk.k
  var rng = initRand(0xC0FFEE)
  for trial in 0 ..< 16:
    let A = tf32Fixture(rng, M, Ktotal)
    let B = tf32Fixture(rng, N, Ktotal)

    var refC = newSeq[float32](M * N)
    refC.gemm_tf32_ref(A, B, M, N, Ktotal, 0.0'f32)
    var gpuC = newSeq[float32](M * N)
    engine.run<<(1, toIntVal(atom.threadCount(opA)))>>("gemmUkernelKernel", gpuC, (A, B))
    allClose(gpuC, refC, M, N, "trial " & $trial)

  echo "  OK: m16n8k8 tf32 gemm_ukernel matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 2 k_blocks, 16 trials)"

proc testTiled*[E](engine: var E; tiled: static TiledMma; label: string) =
  ## The tiled GEMM on the (2,2,1)-tiled atom: 1×1 grid, K = TILE_K = 16
  ## (two k_blocks through gemm_ukernel), config (α, β) = (1, 0). C is
  ## NaN-prefilled: the β=0 branch must skip the C read, so a spurious
  ## read fails the check.
  const
    TILE_K = 16
    thrM = toIntVal(tiled.threadLayout.shape[0])
    thrN = toIntVal(tiled.threadLayout.shape[1])
    thrK = toIntVal(tiled.threadLayout.shape[2])
    TILE_M = thrM * tiled.atom.mnk.m
    TILE_N = thrN * tiled.atom.mnk.n
    blockSize = toIntVal(tiled.atom.threadCount(opA)) * thrM * thrN * thrK
  var rng = initRand(0xC0FFEE)
  for trial in 0 ..< 16:
    let A_gpu = tf32Fixture(rng, TILE_M, TILE_K)
    let B_gpu = tf32Fixture(rng, TILE_N, TILE_K)

    # oracle: the simple CPU triple-loop reference — C_ref = α·acc + β·C_init.
    # (α, β) = (1, 0): the β term drops — the kernel's β=0 branch must skip
    # the C read, so the NaN-prefilled gpuC stays untouched on success.
    var acc = newSeq[float32](TILE_M * TILE_N)
    acc.gemm_tf32_ref(A_gpu, B_gpu, TILE_M, TILE_N, TILE_K, 0.0'f32)
    const alpha = 1.0'f32
    const beta = 0.0'f32
    var C_ref = newSeq[float32](TILE_M * TILE_N)
    for i in 0 ..< TILE_M * TILE_N:
      C_ref[i] = alpha * acc[i]

    var gpuC = newSeq[float32](TILE_M * TILE_N)
    for i in 0 ..< TILE_M * TILE_N:
      gpuC[i] = 0x7FC00000'f32    # NaN sentinel — a spurious C read fails
    engine.run<<(1, blockSize)>>("gemmTiledKernel", gpuC,
               (A_gpu, B_gpu, alpha, beta))
    allClose(gpuC, C_ref, TILE_M, TILE_N, "trial " & $trial)

  echo "  OK: gemm_tiled K=16 (2 k_blocks) matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 16 trials, (1,0), NaN C)"

proc testTiledMultiBlock*[E](engine: var E; tiled: static TiledMma; label: string) =
  ## The tiled GEMM with K = 32 (TILE_K = 32): four k_blocks accumulated
  ## through one gemm_ukernel call — the full-K fragment must be gathered
  ## completely before the k_block loop reads it.
  const
    TILE_K = 32
    thrM = toIntVal(tiled.threadLayout.shape[0])
    thrN = toIntVal(tiled.threadLayout.shape[1])
    thrK = toIntVal(tiled.threadLayout.shape[2])
    TILE_M = thrM * tiled.atom.mnk.m
    TILE_N = thrN * tiled.atom.mnk.n
    blockSize = toIntVal(tiled.atom.threadCount(opA)) * thrM * thrN * thrK
  var rng = initRand(0xF1F1)
  for trial in 0 ..< 16:
    let A_gpu = tf32Fixture(rng, TILE_M, TILE_K)
    let B_gpu = tf32Fixture(rng, TILE_N, TILE_K)

    var acc = newSeq[float32](TILE_M * TILE_N)
    acc.gemm_tf32_ref(A_gpu, B_gpu, TILE_M, TILE_N, TILE_K, 0.0'f32)
    const alpha = 1.0'f32
    const beta = 0.0'f32
    var C_ref = newSeq[float32](TILE_M * TILE_N)
    for i in 0 ..< TILE_M * TILE_N:
      C_ref[i] = alpha * acc[i]

    var gpuC = newSeq[float32](TILE_M * TILE_N)
    for i in 0 ..< TILE_M * TILE_N:
      gpuC[i] = 0x7FC00000'f32    # NaN sentinel — a spurious C read fails
    engine.run<<(1, blockSize)>>("gemmTiledKernelK32", gpuC,
               (A_gpu, B_gpu, alpha, beta))
    allClose(gpuC, C_ref, TILE_M, TILE_N, "trial " & $trial)

  echo "  OK: gemm_tiled K=32 (4 k_blocks) matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 16 trials, (1,0), NaN C)"

proc testGemmCta*[E](engine: var E; tiled: static TiledMma; label: string) =
  ## C(64×32) = α·A(64×32)·B(32×32) + β·C over a 2×2 CTA grid: each CTA
  ## computes its (mCTA, nCTA) tile of the problem GEMM. (α, β) = (1, 0),
  ## C NaN-prefilled (a spurious C read fails), 16 trials, bit-exact vs
  ## the tf32 reference.
  const
    M = 64
    N = 32
    K = 32
    TILE_M = 32
    TILE_N = 16
    TILE_K = 32
    thrM = toIntVal(tiled.threadLayout.shape[0])
    thrN = toIntVal(tiled.threadLayout.shape[1])
    thrK = toIntVal(tiled.threadLayout.shape[2])
    blockSize = toIntVal(tiled.atom.threadCount(opA)) * thrM * thrN * thrK
  const alpha = 1.0'f32
  const beta = 0.0'f32
  var rng = initRand(0xC0FFEE)

  for trial in 0 ..< 16:
    let A_gpu = tf32Fixture(rng, M, K)
    let B_gpu = tf32Fixture(rng, N, K)

    var acc = newSeq[float32](M * N)
    acc.gemm_tf32_ref(A_gpu, B_gpu, M, N, K, 0.0'f32)
    var C_ref = newSeq[float32](M * N)
    for i in 0 ..< M * N:
      C_ref[i] = alpha * acc[i]

    var gpuC = newSeq[float32](M * N)
    for i in 0 ..< M * N:
      gpuC[i] = 0x7FC00000'f32    # NaN sentinel, a spurious C read fails
    engine.run<<((M div TILE_M) * (N div TILE_N), blockSize)>>("gemmCtaKernel", gpuC,
               (A_gpu, B_gpu, alpha, beta))
    allClose(gpuC, C_ref, M, N, "gemm_cta trial " & $trial)

  echo "  OK: gemm_cta M=64 N=32 K=32 tile (32,16,32) matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 2x2 CTA grid, 128 threads, 16 trials, (1,0), NaN C)"

proc testGemmCtaBeta*[E](engine: var E; tiled: static TiledMma; label: string) =
  ## Same 2×2 grid with (α, β) = (1, 1): C is pre-loaded from the fixture
  ## domain, verify D = α·AB + β·C_old elementwise.
  const
    M = 64
    N = 32
    K = 32
    TILE_M = 32
    TILE_N = 16
    TILE_K = 32
    thrM = toIntVal(tiled.threadLayout.shape[0])
    thrN = toIntVal(tiled.threadLayout.shape[1])
    thrK = toIntVal(tiled.threadLayout.shape[2])
    blockSize = toIntVal(tiled.atom.threadCount(opA)) * thrM * thrN * thrK
  const alpha = 1.0'f32
  const beta = 1.0'f32
  var rng = initRand(0xBEEF)

  for trial in 0 ..< 16:
    let A_gpu = tf32Fixture(rng, M, K)
    let B_gpu = tf32Fixture(rng, N, K)
    var C_init = newSeq[float32](M * N)
    for i in 0 ..< M * N:
      C_init[i] = float32(rng.rand(0 .. 15))   # exact-representable domain

    var acc = newSeq[float32](M * N)
    acc.gemm_tf32_ref(A_gpu, B_gpu, M, N, K, 0.0'f32)
    var C_ref = newSeq[float32](M * N)
    for i in 0 ..< M * N:
      C_ref[i] = alpha * acc[i] + beta * C_init[i]

    var gpuC = C_init
    engine.run<<((M div TILE_M) * (N div TILE_N), blockSize)>>("gemmCtaKernel", gpuC,
               (A_gpu, B_gpu, alpha, beta))
    allClose(gpuC, C_ref, M, N, "gemm_cta beta trial " & $trial)

  echo "  OK: gemm_cta (1,1) matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 2x2 CTA grid, C pre-loaded, 16 trials)"

proc testGemmCtaK64*[E](engine: var E; tiled: static TiledMma; label: string) =
  ## C(64×32) = α·A(64×64)·B(32×64) + β·C over a 2×2 CTA grid with
  ## TWO k-tiles: K = 64 = 2·BLK_K with BLK_K = 32, so gemm_cta slices
  ## each full-K CTA tile into two gemm_tiled k-tile passes accumulated
  ## into the same dFrag, then runs the epilogue once. (α, β) = (1, 0),
  ## C NaN-prefilled (a spurious C read fails), 16 trials, bit-exact vs
  ## the tf32 reference.
  const
    M = 64
    N = 32
    K = 64
    TILE_M = 32
    TILE_N = 16
    BLK_K = 32
    thrM = toIntVal(tiled.threadLayout.shape[0])
    thrN = toIntVal(tiled.threadLayout.shape[1])
    thrK = toIntVal(tiled.threadLayout.shape[2])
    blockSize = toIntVal(tiled.atom.threadCount(opA)) * thrM * thrN * thrK
  const alpha = 1.0'f32
  const beta = 0.0'f32
  var rng = initRand(0x2B17)

  for trial in 0 ..< 16:
    let A_gpu = tf32Fixture(rng, M, K)
    let B_gpu = tf32Fixture(rng, N, K)

    var acc = newSeq[float32](M * N)
    acc.gemm_tf32_ref(A_gpu, B_gpu, M, N, K, 0.0'f32)
    var C_ref = newSeq[float32](M * N)
    for i in 0 ..< M * N:
      C_ref[i] = alpha * acc[i]

    var gpuC = newSeq[float32](M * N)
    for i in 0 ..< M * N:
      gpuC[i] = 0x7FC00000'f32    # NaN sentinel, a spurious C read fails
    engine.run<<((M div TILE_M) * (N div TILE_N), blockSize)>>("gemmCtaK64Kernel", gpuC,
               (A_gpu, B_gpu, alpha, beta))
    allClose(gpuC, C_ref, M, N, "gemm_cta K=64 trial " & $trial)

  echo "  OK: gemm_cta M=64 N=32 K=64 (2 k-tiles, BLK_K=32) matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 2x2 CTA grid, 128 threads, 16 trials, (1,0), NaN C)"

proc testGemmCtaSingle*[E](engine: var E; tiled: static TiledMma; label: string) =
  ## M=32, N=16, K=32, tile (32, 16, 32): a 1×1 CTA grid for variety.
  ## (α, β) = (1, 0), NaN-prefilled C.
  const
    M = 32
    N = 16
    K = 32
    TILE_M = 32
    TILE_N = 16
    TILE_K = 32
    thrM = toIntVal(tiled.threadLayout.shape[0])
    thrN = toIntVal(tiled.threadLayout.shape[1])
    thrK = toIntVal(tiled.threadLayout.shape[2])
    blockSize = toIntVal(tiled.atom.threadCount(opA)) * thrM * thrN * thrK
  const alpha = 1.0'f32
  const beta = 0.0'f32
  var rng = initRand(0xF1F1)

  for trial in 0 ..< 16:
    let A_gpu = tf32Fixture(rng, M, K)
    let B_gpu = tf32Fixture(rng, N, K)

    var acc = newSeq[float32](M * N)
    acc.gemm_tf32_ref(A_gpu, B_gpu, M, N, K, 0.0'f32)
    var C_ref = newSeq[float32](M * N)
    for i in 0 ..< M * N:
      C_ref[i] = alpha * acc[i]

    var gpuC = newSeq[float32](M * N)
    for i in 0 ..< M * N:
      gpuC[i] = 0x7FC00000'f32
    engine.run<<(1, blockSize)>>("gemmCtaKernelSingle", gpuC,
               (A_gpu, B_gpu, alpha, beta))
    allClose(gpuC, C_ref, M, N, "gemm_cta single trial " & $trial)

  echo "  OK: gemm_cta M=32 N=16 K=32 tile (32,16,32) matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 1x1 CTA grid, 128 threads, 16 trials, (1,0), NaN C)"

proc testGemmCtaIdentity*[E](engine: var E; tiled: static TiledMma; label: string) =
  ## The EpiIdentity grid epilogue: D = AB over the same 2×2 CTA grid.
  ## D is NaN-prefilled, a dropped store leaves NaN and fails the check.
  const
    M = 64
    N = 32
    K = 32
    TILE_M = 32
    TILE_N = 16
    TILE_K = 32
    thrM = toIntVal(tiled.threadLayout.shape[0])
    thrN = toIntVal(tiled.threadLayout.shape[1])
    thrK = toIntVal(tiled.threadLayout.shape[2])
    blockSize = toIntVal(tiled.atom.threadCount(opA)) * thrM * thrN * thrK
  var rng = initRand(0x10EA)
  for trial in 0 ..< 16:
    let A_gpu = tf32Fixture(rng, M, K)
    let B_gpu = tf32Fixture(rng, N, K)

    var acc = newSeq[float32](M * N)
    acc.gemm_tf32_ref(A_gpu, B_gpu, M, N, K, 0.0'f32)
    var C_ref = newSeq[float32](M * N)
    for i in 0 ..< M * N:
      C_ref[i] = acc[i]

    var gpuC = newSeq[float32](M * N)
    for i in 0 ..< M * N:
      gpuC[i] = 0x7FC00000'f32
    engine.run<<((M div TILE_M) * (N div TILE_N), blockSize)>>("gemmCtaIdentityKernel", gpuC,
               (A_gpu, B_gpu))
    allClose(gpuC, C_ref, M, N, "gemm_cta identity trial " & $trial)

  echo "  OK: gemm_cta identity (EpiIdentity, D = AB) matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 2x2 CTA grid, 128 threads, 16 trials, NaN D)"

proc testGemmCtaReLU*[E](engine: var E; tiled: static TiledMma; label: string) =
  ## The EpiReLU grid epilogue: D = max(0, AB) over the same 2×2 CTA grid.
  ## The fixture domain -15..15 makes AB span negative values, so the clamp
  ## is exercised; the NaN-prefilled D still catches dropped stores.
  const
    M = 64
    N = 32
    K = 32
    TILE_M = 32
    TILE_N = 16
    TILE_K = 32
    thrM = toIntVal(tiled.threadLayout.shape[0])
    thrN = toIntVal(tiled.threadLayout.shape[1])
    thrK = toIntVal(tiled.threadLayout.shape[2])
    blockSize = toIntVal(tiled.atom.threadCount(opA)) * thrM * thrN * thrK
  var rng = initRand(0x2E4A)
  for trial in 0 ..< 16:
    let A_gpu = tf32Fixture(rng, M, K)
    let B_gpu = tf32Fixture(rng, N, K)

    var acc = newSeq[float32](M * N)
    acc.gemm_tf32_ref(A_gpu, B_gpu, M, N, K, 0.0'f32)
    var C_ref = newSeq[float32](M * N)
    for i in 0 ..< M * N:
      C_ref[i] = max(acc[i], 0.0'f32)

    var gpuC = newSeq[float32](M * N)
    for i in 0 ..< M * N:
      gpuC[i] = 0x7FC00000'f32
    engine.run<<((M div TILE_M) * (N div TILE_N), blockSize)>>("gemmCtaReLUKernel", gpuC,
               (A_gpu, B_gpu))
    allClose(gpuC, C_ref, M, N, "gemm_cta relu trial " & $trial)

  echo "  OK: gemm_cta relu (EpiReLU, D = max(0, AB)) matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 2x2 CTA grid, 128 threads, 16 trials, NaN D)"

proc testGemmCtaBias*[E](engine: var E; tiled: static TiledMma; label: string) =
  ## The EpiAddBias grid epilogue: D = AB + bias over the same 2×2 CTA
  ## grid. The bias is a (N,) column vector; the kernel builds a stride-0
  ## broadcast view over the output fragment's shape, so every row of a
  ## column reads the same bias element. The fixture domain keeps D
  ## exact-representable.
  const
    M = 64
    N = 32
    K = 32
    TILE_M = 32
    TILE_N = 16
    TILE_K = 32
    thrM = toIntVal(tiled.threadLayout.shape[0])
    thrN = toIntVal(tiled.threadLayout.shape[1])
    thrK = toIntVal(tiled.threadLayout.shape[2])
    blockSize = toIntVal(tiled.atom.threadCount(opA)) * thrM * thrN * thrK
  var rng = initRand(0x4B1A5E)
  for trial in 0 ..< 16:
    let A_gpu = tf32Fixture(rng, M, K)
    let B_gpu = tf32Fixture(rng, N, K)
    var bias = newSeq[float32](N)
    for j in 0 ..< N:
      bias[j] = float32(rng.rand(0 .. 15))

    var acc = newSeq[float32](M * N)
    acc.gemm_tf32_ref(A_gpu, B_gpu, M, N, K, 0.0'f32)
    var C_ref = newSeq[float32](M * N)
    for i in 0 ..< M * N:
      C_ref[i] = acc[i] + bias[i div M]   # col-major: column of flat i

    var gpuC = newSeq[float32](M * N)
    for i in 0 ..< M * N:
      gpuC[i] = 0x7FC00000'f32
    engine.run<<((M div TILE_M) * (N div TILE_N), blockSize)>>("gemmCtaBiasKernel", gpuC,
               (A_gpu, B_gpu, bias))
    allClose(gpuC, C_ref, M, N, "gemm_cta bias trial " & $trial)

  echo "  OK: gemm_cta bias (EpiAddBias, D = AB + bias) matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 2x2 CTA grid, 128 threads, 16 trials, column broadcast, NaN D)"

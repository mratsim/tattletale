## Shared oracle harness for the manual_*_cuda GPU tests.
##
## The per-arch test files (manual_sm80_*, ...) differ only in
## the atom + the driver kernel. The fixture generation, the tf32 reference
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
  ## failure. Defaults are loose for the tf32 oracle domain. The current
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
#  TensorViews. gemm_tf32_ref is the flat-array tf32-exact oracle
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
  ## gemm_fragment → gemm_atom rename, now a proc, not a template.)
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
  ## signs of the accumulator. A dropped clamp would then fail the test.
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
  ## identical exact result. fmaf is used merely for convenience. Do not
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
  ## OpenCL drivers both call this (and verifyMicrotile). They differ only in the launch.
  ## The _cuda/_opencl files must not touch tf32Fixture or
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
    thrM = tiled.thrM
    thrN = tiled.thrN
    thrK = tiled.thrK
    TILE_M = thrM * tiled.atom.mnk.m
    TILE_N = thrN * tiled.atom.mnk.n
    blockSize = tiled.threadCount()
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
  ## through one gemm_ukernel call: the full-K fragment must be copied
  ## from the staged smem tile completely before the k_block loop reads it.
  const
    TILE_K = 32
    thrM = tiled.thrM
    thrN = tiled.thrN
    thrK = tiled.thrK
    TILE_M = thrM * tiled.atom.mnk.m
    TILE_N = thrN * tiled.atom.mnk.n
    blockSize = tiled.threadCount()
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
    thrM = tiled.thrM
    thrN = tiled.thrN
    thrK = tiled.thrK
    blockSize = tiled.threadCount()
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

  echo "  OK: gemm_cta M=64 N=32 K=32 tile (32,16,32) matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 1x1 CTA grid, 256 threads, 16 trials, (1,0), NaN C)"

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
    thrM = tiled.thrM
    thrN = tiled.thrN
    thrK = tiled.thrK
    blockSize = tiled.threadCount()
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

  echo "  OK: gemm_cta (1,1) matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 1x1 CTA grid, C pre-loaded, 16 trials)"

proc testGemmCtaK64*[E](engine: var E; tiled: static TiledMma; label: string) =
  ## C(64×32) = α·A(64×64)·B(32×64) + β·C over a 2×2 CTA grid with
  ## TWO k-tiles: K = 64 = 2·tileK with tileK = 32, so gemm_cta slices
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
    tileK = 32
    thrM = tiled.thrM
    thrN = tiled.thrN
    thrK = tiled.thrK
    blockSize = tiled.threadCount()
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

  echo "  OK: gemm_cta M=64 N=32 K=64 (2 k-tiles, tileK=32) matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 1x1 CTA grid, 256 threads, 16 trials, (1,0), NaN C)"

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
    thrM = tiled.thrM
    thrN = tiled.thrN
    thrK = tiled.thrK
    blockSize = tiled.threadCount()
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
    thrM = tiled.thrM
    thrN = tiled.thrN
    thrK = tiled.thrK
    blockSize = tiled.threadCount()
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

  echo "  OK: gemm_cta identity (EpiIdentity, D = AB) matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 1x1 CTA grid, 256 threads, 16 trials, NaN D)"

proc testGemmCtaReLU*[E](engine: var E; tiled: static TiledMma; label: string) =
  ## The EpiReLU grid epilogue: D = max(0, AB) over the same 2×2 CTA grid.
  ## The fixture domain -15..15 makes AB span negative values, so the clamp
  ## is exercised. The NaN-prefilled D still catches dropped stores.
  const
    M = 64
    N = 32
    K = 32
    TILE_M = 32
    TILE_N = 16
    TILE_K = 32
    thrM = tiled.thrM
    thrN = tiled.thrN
    thrK = tiled.thrK
    blockSize = tiled.threadCount()
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

  echo "  OK: gemm_cta relu (EpiReLU, D = max(0, AB)) matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 1x1 CTA grid, 256 threads, 16 trials, NaN D)"

proc testGemmCtaBias*[E](engine: var E; tiled: static TiledMma; label: string) =
  ## The EpiAddBias grid epilogue: D = AB + bias over the same 2×2 CTA
  ## grid. The bias is a (N,) column vector. The kernel builds a stride-0
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
    thrM = tiled.thrM
    thrN = tiled.thrN
    thrK = tiled.thrK
    blockSize = tiled.threadCount()
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

  echo "  OK: gemm_cta bias (EpiAddBias, D = AB + bias) matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 1x1 CTA grid, 256 threads, 16 trials, column broadcast, NaN D)"

proc testGemmCtaDynamic*[E](engine: var E; tiled: static TiledMma;
                            label, kernelName: string;
                            M, N, K, ldA, ldB, ldC, kView: static int;
                            alpha = 1.0'f32, beta = 0.0'f32) =
  ## C(M×N) = α·A(M×K)·B(N×K) + β·C with runtime M/N/K and runtime
  ## leading strides: the kernel receives M, N, K, ldA, ldB, ldC as
  ## arguments and builds the problem views from them. gemm_cta covers
  ## the ceil(M/tileM) × ceil(N/tileN) CTA grid. The buffers are padded
  ## to the leading strides and to the ALLOCATED K (kView): the fixture
  ## fills the K problem columns at the strided offsets, and the columns
  ## K ..< kView are NaN-padded (0x7FC00000): a false-positive load
  ## in the ragged-K residue reads NaN into the accumulator and allClose
  ## fails. The view K is kView (the kernel literal), the problem K is
  ## runtime (at most kView).
  ## A shape with M mod tileM != 0 or N mod tileN != 0 exercises the
  ## ragged boundary predication, where the masked load zero-fills
  ## outside the problem and only the valid elements are stored.
  ## K mod tileK != 0 exercises the ragged-K residue, the last k-tile
  ## partial with its k >= validK coordinates zero-filled at the load.
  ## Two checks guard the predication itself:
  ## the pad region of C (rows m >= M, the guard columns n >= N) must
  ## stay NaN after the run, so a store-mask false positive fails the
  ## test instead of landing invisibly.
  ##   - beta == 0: C is NaN-prefilled, a spurious C read fails
  ##   - beta != 0: C's valid region is prefilled with exact small ints,
  ##     the reference adds β·C, and the masked-off C reads are skipped
  ##     (the mask wraps the read)
  ## Bit-exact vs the tf32 reference.
  const
    TILE_K = 32                     # the k-tile depth
    thrM = tiled.thrM
    thrN = tiled.thrN
    thrK = tiled.thrK
    # the tile dims follow from the thread layout's exact coverage, the
    # same contract gemm_tiled asserts. The CUDA/OpenCL kernel strings
    # hardcode the matching literals (pinned by the manual tests)
    TILE_M = thrM * toIntVal(tiled.atom.mnk.m)
    TILE_N = thrN * toIntVal(tiled.atom.mnk.n)
    blockSize = tiled.threadCount()
  static:
    doAssert TILE_K mod (thrK * toIntVal(tiled.atom.mnk.k)) == 0,
      "testGemmCtaDynamic: the k-tile depth (" & $TILE_K &
      ") must be a multiple of thrK·atomK"
  # kView and K are runtime params: their contract is checked at runtime
  # (the kernel literal is pinned by the manual tests' static asserts)
  doAssert kView mod TILE_K == 0,
    "testGemmCtaDynamic: the allocated K (" & $kView &
    ") must be a multiple of the k-tile depth (" & $TILE_K &
    ") (the view K must tile evenly, gemm_cta's static contract)"
  doAssert K <= kView,
    "testGemmCtaDynamic: the problem K (" & $K &
    ") must not exceed the allocated K (" & $kView & ")"
  let gridM = (M + TILE_M - 1) div TILE_M
  let gridN = (N + TILE_N - 1) div TILE_N
  var rng = initRand(0xC0FFEE + M * 131 + N * 17 + K * 11 + ldA * 7)
  for trial in 0 ..< 16:
    let A_fixture = tf32Fixture(rng, M, K)
    let B_fixture = tf32Fixture(rng, N, K)

    # padded buffers: the fixture at the strided offsets, the pad rows
    # never read (the views cover only the problem M/N), the K-pad
    # columns K ..< kView NaN: a ragged-K false positive reads NaN
    # into the accumulator
    var A_gpu = newSeq[uint32](ldA * kView)
    var B_gpu = newSeq[uint32](ldB * kView)
    for m in 0 ..< ldA * kView:
      A_gpu[m] = 0x7FC00000'u32
    for m in 0 ..< ldB * kView:
      B_gpu[m] = 0x7FC00000'u32
    for k in 0 ..< K:
      for m in 0 ..< M:
        A_gpu[m + k * ldA] = A_fixture[m + k * M]
      for n in 0 ..< N:
        B_gpu[n + k * ldB] = B_fixture[n + k * N]

    var acc = newSeq[float32](M * N)
    acc.gemm_tf32_ref(A_fixture, B_fixture, M, N, K, 0.0'f32)
    # the C fixture: exact small ints, used by the β != 0 reference and
    # the C prefill (the valid region only, the pad stays NaN)
    var C_fixture = newSeq[float32](M * N)
    for i in 0 ..< M * N:
      C_fixture[i] = float32((i mod M) * 3 + (i div M) * 7 + 1)
    var C_ref = newSeq[float32](M * N)
    for i in 0 ..< M * N:
      C_ref[i] = alpha * acc[i] + beta * C_fixture[i]

    # TILE_N guard columns: a store-mask false positive in n >= N lands
    # here instead of past the buffer (ragged-N overshoot can reach
    # n = N + TILE_N - 1)
    var gpuC = newSeq[float32](ldC * (N + TILE_N))
    for i in 0 ..< gpuC.len:
      gpuC[i] = 0x7FC00000'f32    # NaN sentinel, a spurious C read fails
    if beta != 0.0'f32:
      for n in 0 ..< N:
        for m in 0 ..< M:
          gpuC[m + n * ldC] = C_fixture[m + n * M]
    engine.run<<(gridM * gridN, blockSize)>>(kernelName, gpuC,
               (A_gpu, B_gpu, int32(M), int32(N), int32(K), int32(ldA), int32(ldB), int32(ldC),
                alpha, beta))

    # repack the ldC-strided result into a compact (M, N) array
    var compact = newSeq[float32](M * N)
    for n in 0 ..< N:
      for m in 0 ..< M:
        compact[m + n * M] = gpuC[m + n * ldC]
    allClose(compact, C_ref, M, N,
             "gemm_cta dynamic M=" & $M & " N=" & $N & " K=" & $K & " trial " & $trial)

    # the pad region must be untouched: any element outside the valid
    # (M, N) range still holding the NaN sentinel proves the store mask
    # skipped it (a false positive writes the pad or a guard column)
    for n in 0 ..< N + TILE_N:
      for m in 0 ..< ldC:
        if m >= M or n >= N:
          doAssert isNaN(gpuC[m + n * ldC]),
            "gemm_cta dynamic: a store wrote past the valid (M, N) range" &
            " (m=" & $m & ", n=" & $n & "), the store mask has a false positive"

  echo "  OK: gemm_cta dynamic M=" & $M & " N=" & $N & " K=" & $K &
    " kView=" & $kView &
    " ldA=" & $ldA & " ldB=" & $ldB & " ldC=" & $ldC &
    " (α=" & $alpha & ", β=" & $beta & ") matches reference within 1e-4" &
    " (tf32-exact fixture, ", label, " atom, " & $gridM & "x" & $gridN &
    " CTA grid, 16 trials, K-pad + pad region verified NaN)"

# ═════════════════════════════════════════════════════════════════════════
#  gemm_gpu oracles (Level 5)
# ═════════════════════════════════════════════════════════════════════════
#
#  The gemm_gpu kernels are one-line entry points: they build the problem views
#  over the raw buffers and hand them to gemm_gpu, which derives the tma policy,
#  the CTA position (blockIdx.x/y) and the per-thread epilogue shard internally.
#  Launched over a 2D grid.

proc testGemmGpu*[E](engine: var E; tiled: static TiledMma; label: string) =
  ## gemm_gpu with EpiAXPBY (1, 0): C(32×32) = A(32×32)·B(32×32) over a 1×1 CTA grid,
  ## 16 trials, bit-exact vs the tf32 reference.
  ## C is NaN-prefilled, a spurious C read fails.
  const
    M = 32
    N = 32
    K = 32
    TILE_M = tiled.thrM * tiled.atom.mnk.m
    TILE_N = tiled.thrN * tiled.atom.mnk.n
    blockSize = tiled.threadCount()
  const alpha = 1.0'f32
  const beta = 0.0'f32
  var rng = initRand(0xA11CE)
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
    engine.run<<((M div TILE_M, N div TILE_N), blockSize)>>("gemmGpuKernel", gpuC,
               (A_gpu, B_gpu, alpha, beta))
    allClose(gpuC, C_ref, M, N, "gemm_gpu trial " & $trial)
  echo "  OK: gemm_gpu M=32 N=32 K=32 matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 1x1 CTA grid, 256 threads, 16 trials, (1,0), NaN C)"

proc testGemmGpuBeta*[E](engine: var E; tiled: static TiledMma; label: string) =
  ## Same 1×1 grid with (α, β) = (1, 1): C is pre-loaded from the fixture domain,
  ## verify D = α·AB + β·C_old elementwise.
  const
    M = 32
    N = 32
    K = 32
    TILE_M = tiled.thrM * tiled.atom.mnk.m
    TILE_N = tiled.thrN * tiled.atom.mnk.n
    blockSize = tiled.threadCount()
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
    engine.run<<((M div TILE_M, N div TILE_N), blockSize)>>("gemmGpuKernel", gpuC,
               (A_gpu, B_gpu, alpha, beta))
    allClose(gpuC, C_ref, M, N, "gemm_gpu beta trial " & $trial)
  echo "  OK: gemm_gpu (1,1) matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 1x1 CTA grid, C pre-loaded, 16 trials)"

proc testGemmGpuK64*[E](engine: var E; tiled: static TiledMma; label: string) =
  ## gemm_gpu with K = 64: two k-tiles of depth 32 through gemm_cta's k-tile loop,
  ## accumulated into one fragment, one epilogue pass, over a 1×1 CTA grid.
  const
    M = 32
    N = 32
    K = 64
    TILE_M = tiled.thrM * tiled.atom.mnk.m
    TILE_N = tiled.thrN * tiled.atom.mnk.n
    blockSize = tiled.threadCount()
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
      gpuC[i] = 0x7FC00000'f32
    engine.run<<((M div TILE_M, N div TILE_N), blockSize)>>("gemmGpuK64Kernel", gpuC,
               (A_gpu, B_gpu, alpha, beta))
    allClose(gpuC, C_ref, M, N, "gemm_gpu K=64 trial " & $trial)
  echo "  OK: gemm_gpu M=32 N=32 K=64 (2 k-tiles, tileK=32) matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 1x1 CTA grid, 256 threads, 16 trials, (1,0), NaN C)"

proc testGemmGpuIdentity*[E](engine: var E; tiled: static TiledMma; label: string) =
  ## gemm_gpu with EpiIdentity: D = AB over the same 1×1 CTA grid.
  ## D is NaN-prefilled, a dropped store leaves NaN and fails the check.
  const
    M = 32
    N = 32
    K = 32
    TILE_M = tiled.thrM * tiled.atom.mnk.m
    TILE_N = tiled.thrN * tiled.atom.mnk.n
    blockSize = tiled.threadCount()
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
    engine.run<<((M div TILE_M, N div TILE_N), blockSize)>>("gemmGpuIdentityKernel", gpuC,
               (A_gpu, B_gpu))
    allClose(gpuC, C_ref, M, N, "gemm_gpu identity trial " & $trial)
  echo "  OK: gemm_gpu identity (EpiIdentity, D = AB) matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 1x1 CTA grid, 256 threads, 16 trials, NaN D)"

proc testGemmGpuReLU*[E](engine: var E; tiled: static TiledMma; label: string) =
  ## gemm_gpu with EpiReLU: D = max(0, AB) over the same 1×1 CTA grid.
  ## The fixture domain -15..15 makes AB span negative values.
  const
    M = 32
    N = 32
    K = 32
    TILE_M = tiled.thrM * tiled.atom.mnk.m
    TILE_N = tiled.thrN * tiled.atom.mnk.n
    blockSize = tiled.threadCount()
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
    engine.run<<((M div TILE_M, N div TILE_N), blockSize)>>("gemmGpuReLUKernel", gpuC,
               (A_gpu, B_gpu))
    allClose(gpuC, C_ref, M, N, "gemm_gpu relu trial " & $trial)
  echo "  OK: gemm_gpu relu (EpiReLU, D = max(0, AB)) matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 1x1 CTA grid, 256 threads, 16 trials, NaN D)"

proc testGemmGpuBias*[E](engine: var E; tiled: static TiledMma; label: string) =
  ## gemm_gpu with EpiAddBias: D = AB + bias over the same 1×1 CTA grid.
  ## The bias is a (N,) column vector, the op's problem view has stride-0 rows.
  ## The shard partitions it onto the thread's fragment.
  const
    M = 32
    N = 32
    K = 32
    TILE_M = tiled.thrM * tiled.atom.mnk.m
    TILE_N = tiled.thrN * tiled.atom.mnk.n
    blockSize = tiled.threadCount()
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
    engine.run<<((M div TILE_M, N div TILE_N), blockSize)>>("gemmGpuBiasKernel", gpuC,
               (A_gpu, B_gpu, bias))
    allClose(gpuC, C_ref, M, N, "gemm_gpu bias trial " & $trial)
  echo "  OK: gemm_gpu bias (EpiAddBias, D = AB + bias) matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 1x1 CTA grid, 256 threads, 16 trials, column broadcast, NaN D)"

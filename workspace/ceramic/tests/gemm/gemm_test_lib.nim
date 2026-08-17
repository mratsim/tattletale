## Shared reference harness for the manual_*_cuda GPU tests.
##
## Per-arch test files differ only in the atom and the driver kernel.
## Fixture generation, the tf32 reference GEMM, the execute/verify loop,
## and the report live here.


import std/[random, strformat, math, typetraits]
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/tensors
import workspace/ceramic/src/atoms
import workspace/ceramic/src/kernel_gemm_gpu
import workspace/crucible

# ═════════════════════════════════════════════════════════════════════════
#  Low-level primitives: the elementwise/compare helpers the references
#  and drivers build on.
# ═════════════════════════════════════════════════════════════════════════

proc fmaf(a, b, c: float32): float32 {.importc: "fmaf", header: "<math.h>".}
proc allClose(testC, refC: openArray[float32];
               M, N: int;
               context: string;
               relTol = 1e-4'f32; absTol = 1e-4'f32) =
  ## Element-wise numpy.allclose check: |test - ref| <= absTol + relTol·|ref|,
  ## single condition, NaN never equal. Loud doAssert on the first failure.
  ## Default tolerances suit the tf32-exact fixture domain.
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
#  Reference GEMMs the GPU kernels are tested against:
#  gemm_ref, the layout-generic outer product over TensorViews, and
#  gemm_tf32_ref, the flat-array tf32-exact reference.
# ═════════════════════════════════════════════════════════════════════════

proc gemm_ref*[T, ShA, StA, ShB, StB, ShC, StC](
    C: var (TensorView[T, ShC, StC] or Tensor[T, ShC, StC]),
    A: TensorView[T, ShA, StA] or Tensor[T, ShA, StA],
    B: TensorView[T, ShB, StB] or Tensor[T, ShB, StB]) =
  ## Reference fragment gemm: C[m,n] += A[m,k] * B[n,k] (outer product).
  ## Reference GEMM for the GPU kernels, not a performance kernel.
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
#  tf32 fixture machinery: the bit-exact test domain (0..15, tf32ified)
# ═════════════════════════════════════════════════════════════════════════

func tf32ify(x: float32): uint32 =
  ## f32 → tf32 bit pattern: truncate the low 13 mantissa bits
  ## (round-toward-zero). Not cvt.rna (round-to-nearest-away, CUTLASS'
  ## f32→tf32 conversion): both agree on the small-integer fixture domain,
  ## and RZ keeps the bit pattern a pure mask.
  (cast[uint32](x)) and 0xFFFFE000'u32

proc tf32Fixture(rng: var Rand; M, K: int): seq[uint32] =
  ## A random (M, K) col-major tf32 fixture in the f32 domain -15..15.
  ## Every |product| (≤ 15·15 = 225) and |partial sum| (≤ K·225) is
  ## exactly representable in f32's 24-bit mantissa, so the gemm tests
  ## are bit-exact regardless of the mma pipe's internal accumulation
  ## order. Negatives so epilogue clamps (EpiReLU) see both signs.
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
  ## C[m,n] = cInit + Σ_k tf32(A[m,k]) · tf32(B[n,k]), the naive O(M·N·K)
  ## triple loop.
  ##
  ## Bit-exactness does not come from fmaf/ascending-k accumulation:
  ## mma.sync's internal K=8 dot-product order is an undocumented hardware
  ## detail. It comes from the fixture domain (0..15, tf32ified): every
  ## product and partial sum is exactly representable in f32, so any
  ## accumulation order (fmaf, mul+add, or the mma adder tree) yields the
  ## identical exact result. fmaf is used merely for convenience.
  ## Bit-exact only on this domain. Random floats require a tolerance
  ## comparison.
  for m in 0 ..< M:
    for n in 0 ..< N:
      var sum = cInit
      for k in 0 ..< K:
        let av = cast[float32](A[m + k * M])
        let bv = cast[float32](B[n + k * N])
        sum = fmaf(av, bv, sum)
      C[m + n * M] = sum

# ═════════════════════════════════════════════════════════════════════════
#  Per-kernel test drivers (the HwEngine API)
#
# Kernel-name conventions (the test files' cuda/opencl blocks must name them so):
#   microtile: mmaMicrotileKernel / mmaMicrotileExplicitKernel (one module)
#   warp:      gemmWarpKernel
#   tiled:     gemmTiledKernel
#
# ═════════════════════════════════════════════════════════════════════════

proc microtileFixtures(atom: static MmaAtom; rng: var Rand): tuple[A, B: seq[uint32]] =
  ## A and B tf32 fixtures: A is (M, K), B is (N, K), from the atom's mnk.
  const M = atom.mnk.m
  const N = atom.mnk.n
  const K = atom.mnk.k
  result.A = tf32Fixture(rng, M, K)
  result.B = tf32Fixture(rng, N, K)

proc verifyMicrotile(atom: static MmaAtom; trial: int;
                      gpuC: openArray[float32]; A, B: openArray[uint32];
                      cInit: float32; context: string) =
  ## Computes gemm_tf32_ref for the trial and allClose-compares gpuC against it.
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

proc testWarp*[E](engine: var E; atom: static MmaAtom; label: string) =
  ## gemm_warp k-loop: C(M×N) = A(M, 2·K)·B(N, 2·K), two k slices,
  ## 16 trials vs the tf32 reference.
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
    engine.run<<(1, toIntVal(atom.threadCount(opA)))>>("gemmWarpKernel", gpuC, (A, B))
    allClose(gpuC, refC, M, N, "trial " & $trial)

  echo "  OK: m16n8k8 tf32 gemm_warp matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 2 k slices, 16 trials)"

proc testTiled*[E](engine: var E; tiled: static TiledMma; label: string) =
  ## Tiled GEMM on the (2,2,1)-tiled atom: 1×1 grid, K = TILE_K = 16,
  ## config (α, β) = (1, 0). C is NaN-prefilled: the β=0 branch must
  ## skip the C read, so a spurious read fails the check.
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

    # Reference: C_ref = α·acc + β·C_init with (α, β) = (1, 0).
    var acc = newSeq[float32](TILE_M * TILE_N)
    acc.gemm_tf32_ref(A_gpu, B_gpu, TILE_M, TILE_N, TILE_K, 0.0'f32)
    const alpha = 1.0'f32
    const beta = 0.0'f32
    var C_ref = newSeq[float32](TILE_M * TILE_N)
    for i in 0 ..< TILE_M * TILE_N:
      C_ref[i] = alpha * acc[i]

    var gpuC = newSeq[float32](TILE_M * TILE_N)
    for i in 0 ..< TILE_M * TILE_N:
      gpuC[i] = 0x7FC00000'f32    # NaN sentinel, a spurious C read fails
    engine.run<<(1, blockSize)>>("gemmTiledKernel", gpuC,
               (A_gpu, B_gpu, alpha, beta))
    allClose(gpuC, C_ref, TILE_M, TILE_N, "trial " & $trial)

  echo "  OK: gemm_tiled K=16 (2 k slices) matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 16 trials, (1,0), NaN C)"

proc testTiledMultiBlock*[E](engine: var E; tiled: static TiledMma; label: string) =
  ## Tiled GEMM with K = 32 (TILE_K = 32): four passes over the K dimension.
  ## Requires the full-K fragment fully copied from the prepared smem tile
  ## before the loop reads it.
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
      gpuC[i] = 0x7FC00000'f32    # NaN sentinel, a spurious C read fails
    engine.run<<(1, blockSize)>>("gemmTiledKernelK32", gpuC,
               (A_gpu, B_gpu, alpha, beta))
    allClose(gpuC, C_ref, TILE_M, TILE_N, "trial " & $trial)

  echo "  OK: gemm_tiled K=32 (4 k slices) matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 16 trials, (1,0), NaN C)"

proc testGemmCta*[E](engine: var E; tiled: static TiledMma; label: string) =
  ## C(64×32) = α·A(64×32)·B(32×32) + β·C over a 2×2 CTA grid: each CTA
  ## computes its (mCTA, nCTA) tile of the input GEMM. (α, β) = (1, 0),
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
  ## C(64×32) = α·A(64×32)·B(32×32) + β·C over a 2×2 CTA grid with
  ## (α, β) = (1, 1): C_init pre-loaded from the fixture domain, verify
  ## D = α·AB + β·C_init elementwise.
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
  ## K = 64 = 2·32: two tileK-sized slices of K accumulated into one
  ## fragment, then one epilogue pass. (α, β) = (1, 0), C NaN-prefilled
  ## (a spurious C read fails), 16 trials, bit-exact vs the tf32 reference.
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
  ## C(32×16) = α·A(32×32)·B(16×32) + β·C over a 1×1 CTA grid, tile
  ## (32, 16, 32). (α, β) = (1, 0), NaN-prefilled C.
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
  ## EpiIdentity grid epilogue: D = AB over a 2×2 CTA grid.
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
  ## EpiReLU grid epilogue: D = max(0, AB) over a 2×2 CTA grid.
  ## Fixture domain -15..15 makes AB span negative values, so the clamp
  ## is exercised. NaN-prefilled D catches dropped stores.
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
  ## EpiAddBias grid epilogue: D = AB + bias over a 2×2 CTA grid.
  ## Bias is a (N,) column vector broadcast over the rows.
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
  ## leading strides, launched over the ceil(M/tileM) × ceil(N/tileN)
  ## CTA grid. Buffers are padded to the leading strides and to the
  ## allocated K (kView): the fixture fills the K input columns at the
  ## strided offsets, and columns K ..< kView are NaN-padded
  ## (0x7FC00000), so a false-positive load in the ragged-K residue
  ## reads NaN into the accumulator and allClose fails. The view K is kView
  ## (the kernel literal). The input's K is runtime, at most kView.
  ##
  ## M mod tileM != 0 or N mod tileN != 0 exercises the ragged boundary
  ## predication: the masked load zero-fills outside the input and only
  ## the valid elements are stored. K mod tileK != 0 exercises the
  ## ragged-K residue, the last tileK-sized slice of K,
  ## whose k >= validK coordinates are zero-filled at the load.
  ##
  ## Checks on the predication:
  ## - the pad region of C (rows m >= M, guard columns n >= N)
  ##   stays NaN after the run, so a store-mask false positive fails the test instead of landing invisibly.
  ## - beta == 0: C NaN-prefilled, a spurious C read fails
  ## - beta != 0: C's valid region prefilled with exact small ints, the
  ##   reference adds β·C, and the mask wraps the C read
  ##   (skips the masked-off reads)
  ##
  ## Bit-exact vs the tf32 reference.
  const
    TILE_K = 32                     # depth of one tileK-sized slice of K
    thrM = tiled.thrM
    thrN = tiled.thrN
    thrK = tiled.thrK
    # tile dims follow from the thread layout times the atom
    TILE_M = thrM * toIntVal(tiled.atom.mnk.m)
    TILE_N = thrN * toIntVal(tiled.atom.mnk.n)
    blockSize = tiled.threadCount()
  static:
    doAssert TILE_K mod (thrK * toIntVal(tiled.atom.mnk.k)) == 0,
      "testGemmCtaDynamic: the k-tile depth (" & $TILE_K &
      ") must be a multiple of thrK·atomK"
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
    # C fixture: exact small ints for the β != 0 reference and the C prefill
    var C_fixture = newSeq[float32](M * N)
    for i in 0 ..< M * N:
      C_fixture[i] = float32((i mod M) * 3 + (i div M) * 7 + 1)
    var C_ref = newSeq[float32](M * N)
    for i in 0 ..< M * N:
      C_ref[i] = alpha * acc[i] + beta * C_fixture[i]

    # TILE_N guard columns: a store-mask false positive in n >= N lands
    # here instead of past the buffer (ragged-N overshoot reaches
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

    var compact = newSeq[float32](M * N)
    for n in 0 ..< N:
      for m in 0 ..< M:
        compact[m + n * M] = gpuC[m + n * ldC]
    allClose(compact, C_ref, M, N,
             "gemm_cta dynamic M=" & $M & " N=" & $N & " K=" & $K & " trial " & $trial)

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
#  gemm_kernel tests
# ═════════════════════════════════════════════════════════════════════════
#
#  The test kernels build the input views over the raw buffers and hand them
#  to gemm_kernel, which derives the tma policy, the CTA position
#  (blockIdx.x/y), and the per-thread epilogue shard internally.
#  Launched over a 2D grid.

proc testGemmKernel*[E](engine: var E; tiled: static TiledMma; label: string) =
  ## gemm_kernel with EpiAXPBY (1, 0): C(32×32) = A(32×32)·B(32×32) over a 1×1 CTA grid,
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
    engine.run<<((M div TILE_M, N div TILE_N), blockSize)>>("gemmKernelAXPBY", gpuC,
               (A_gpu, B_gpu, alpha, beta))
    allClose(gpuC, C_ref, M, N, "gemm_kernel trial " & $trial)
  echo "  OK: gemm_kernel M=32 N=32 K=32 matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 1x1 CTA grid, 256 threads, 16 trials, (1,0), NaN C)"

proc testGemmKernelBeta*[E](engine: var E; tiled: static TiledMma; label: string) =
  ## gemm_kernel with EpiAXPBY (α, β) = (1, 1): C(32×32) = A(32×32)·B(32×32)
  ## over a 1×1 CTA grid, C_init pre-loaded from the fixture domain, verify
  ## D = α·AB + β·C_init elementwise.
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
    engine.run<<((M div TILE_M, N div TILE_N), blockSize)>>("gemmKernelAXPBY", gpuC,
               (A_gpu, B_gpu, alpha, beta))
    allClose(gpuC, C_ref, M, N, "gemm_kernel beta trial " & $trial)
  echo "  OK: gemm_kernel (1,1) matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 1x1 CTA grid, C pre-loaded, 16 trials)"

proc testGemmKernelK64*[E](engine: var E; tiled: static TiledMma; label: string) =
  ## gemm_kernel with K = 64: two tileK-sized slices of K (depth 32),
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
    engine.run<<((M div TILE_M, N div TILE_N), blockSize)>>("gemmKernelK64", gpuC,
               (A_gpu, B_gpu, alpha, beta))
    allClose(gpuC, C_ref, M, N, "gemm_kernel K=64 trial " & $trial)
  echo "  OK: gemm_kernel M=32 N=32 K=64 (2 k-tiles, tileK=32) matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 1x1 CTA grid, 256 threads, 16 trials, (1,0), NaN C)"

proc testGemmKernelIdentity*[E](engine: var E; tiled: static TiledMma; label: string) =
  ## gemm_kernel with EpiIdentity: D = AB over a 1×1 CTA grid.
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
    engine.run<<((M div TILE_M, N div TILE_N), blockSize)>>("gemmKernelIdentity", gpuC,
               (A_gpu, B_gpu))
    allClose(gpuC, C_ref, M, N, "gemm_kernel identity trial " & $trial)
  echo "  OK: gemm_kernel identity (EpiIdentity, D = AB) matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 1x1 CTA grid, 256 threads, 16 trials, NaN D)"

proc testGemmKernelReLU*[E](engine: var E; tiled: static TiledMma; label: string) =
  ## gemm_kernel with EpiReLU: D = max(0, AB) over a 1×1 CTA grid.
  ## Fixture domain -15..15 makes AB span negative values.
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
    engine.run<<((M div TILE_M, N div TILE_N), blockSize)>>("gemmKernelReLU", gpuC,
               (A_gpu, B_gpu))
    allClose(gpuC, C_ref, M, N, "gemm_kernel relu trial " & $trial)
  echo "  OK: gemm_kernel relu (EpiReLU, D = max(0, AB)) matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 1x1 CTA grid, 256 threads, 16 trials, NaN D)"

proc testGemmKernelBias*[E](engine: var E; tiled: static TiledMma; label: string) =
  ## gemm_kernel with EpiAddBias: D = AB + bias over a 1×1 CTA grid.
  ## Bias is a (N,) column vector (stride-0 rows in the op's input view),
  ## sharded onto the thread's fragment.
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
    engine.run<<((M div TILE_M, N div TILE_N), blockSize)>>("gemmKernelBias", gpuC,
               (A_gpu, B_gpu, bias))
    allClose(gpuC, C_ref, M, N, "gemm_kernel bias trial " & $trial)
  echo "  OK: gemm_kernel bias (EpiAddBias, D = AB + bias) matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 1x1 CTA grid, 256 threads, 16 trials, column broadcast, NaN D)"

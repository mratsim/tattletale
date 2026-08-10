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

import std/[random, strformat, math]
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/tensors
import workspace/ceramic/src/atoms
import workspace/ceramic/src/kernel_gemm_gpu
import workspace/crucible/src/codegen/nvrtc

func tf32ify*(x: float32): uint32 =
  ## f32 → tf32 bit pattern: truncate the low 13 mantissa bits (round-toward-zero).
  ## Deliberately NOT cvt.rna (round-to-nearest-away, CUTLASS' f32→tf32
  ## conversion): the two agree on the small-integer fixture domain, and RZ
  ## keeps the bit pattern a pure mask. Revisit (cvt.rna) when a production
  ## f32→tf32 conversion path is added.
  (cast[uint32](x)) and 0xFFFFE000'u32

proc tf32Fixture*(rng: var Rand; M, K: int): seq[uint32] =
  ## A random (M, K) col-major tf32 fixture in the f32 domain 0..15.
  ## The domain is pinned so every product (≤ 15·15 = 225), partial sum
  ## (≤ K·225) and epilogue is exactly representable in f32's 24-bit
  ## mantissa — this is what makes the gemm tests bit-exact regardless of
  ## the mma pipe's internal accumulation order (see tf32Reference).
  doAssert K * 15 * 15 < 1 shl 24,
    "tf32Fixture: K·15² ≥ 2^24 — partial sums leave the f32 exact-representable" &
    " domain; the oracle would no longer be bit-exact"
  result = newSeq[uint32](M * K)
  for i in 0 ..< result.len:
    result[i] = tf32ify(float32(rng.rand(0 .. 15)))

proc fmaf(a, b, c: float32): float32 {.importc: "fmaf", header: "<math.h>".}

proc tf32Reference*(C: var openArray[float32];
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

proc allClose*(testC, refC: openArray[float32];
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
#  Per-kernel test drivers
# ═════════════════════════════════════════════════════════════════════════

proc testMicrotile*(nv: var NVRTC; atom: static MmaAtom; label: string) =
  ## One register-level MMA on the atom's own tile: C(M×N) = A(M×K)·B(N×K).
  ## Runs both the 4-arg (in-place) and 5-arg (explicit, cFrag = 1.0) forms,
  ## 16 trials each, bit-exact vs the tf32 reference.
  const
    M = atom.mnk.m
    N = atom.mnk.n
    K = atom.mnk.k
  var rng = initRand(0xC0FFEE)
  for trial in 0 ..< 16:
    let A = tf32Fixture(rng, M, K)
    let B = tf32Fixture(rng, N, K)

    # in-place (4-arg)
    var refC = newSeq[float32](M * N)
    refC.tf32Reference(A, B, M, N, K, 0.0'f32)
    var gpuC = newSeq[float32](M * N)
    nv.execute("mmaMicrotileKernel", dim3(1), dim3(toIntVal(atom.threadCount(opA))), gpuC, (A, B))
    allClose(gpuC, refC, M, N, "in-place trial " & $trial)

    # explicit-output (5-arg), cFrag = 1.0
    refC.tf32Reference(A, B, M, N, K, 1.0'f32)
    var gpuD = newSeq[float32](M * N)
    nv.execute("mmaMicrotileExplicitKernel", dim3(1), dim3(toIntVal(atom.threadCount(opA))), gpuD, (A, B))
    allClose(gpuD, refC, M, N, "explicit trial " & $trial)

  echo "  OK — m16n8k8 tf32 microtile matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 16 trials, in-place + explicit)"

proc testUkernel*(nv: var NVRTC; atom: static MmaAtom; label: string) =
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
    refC.tf32Reference(A, B, M, N, Ktotal, 0.0'f32)
    var gpuC = newSeq[float32](M * N)
    nv.execute("gemmUkernelKernel", dim3(1), dim3(toIntVal(atom.threadCount(opA))), gpuC, (A, B))
    allClose(gpuC, refC, M, N, "trial " & $trial)

  echo "  OK — m16n8k8 tf32 gemm_ukernel matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 2 k_blocks, 16 trials)"

proc testTiled*(nv: var NVRTC; tiled: static TiledMma; label: string) =
  ## The tiled GEMM on the (2,2,1)-tiled atom: 1×1 grid, single k-tile
  ## (TILE_K = 16), config (α, β) = (1, 0). C is NaN-prefilled: the β=0
  ## branch must skip the C read, so a spurious read fails the check.
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
    acc.tf32Reference(A_gpu, B_gpu, TILE_M, TILE_N, TILE_K, 0.0'f32)
    const alpha = 1.0'f32
    const beta = 0.0'f32
    var C_ref = newSeq[float32](TILE_M * TILE_N)
    for i in 0 ..< TILE_M * TILE_N:
      C_ref[i] = alpha * acc[i]

    var gpuC = newSeq[float32](TILE_M * TILE_N)
    for i in 0 ..< TILE_M * TILE_N:
      gpuC[i] = 0x7FC00000'f32    # NaN sentinel — a spurious C read fails
    nv.execute("gemmTiledKernel", dim3(1), dim3(blockSize), gpuC,
               (A_gpu, B_gpu, alpha, beta))
    allClose(gpuC, C_ref, TILE_M, TILE_N, "trial " & $trial)

  echo "  OK — gemm_tiled 1×1 single-k-tile matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 16 trials, (1,0), NaN C)"

proc testTiledMultiBlock*(nv: var NVRTC; tiled: static TiledMma; label: string) =
  ## The tiled GEMM with TWO k-tiles (K=32, BLK_K=16): the k_tile loop runs
  ## twice, each block accumulating its own 16-deep slice into cFrag. This
  ## pins F1 — a fragment shaped from the full-K partition would make
  ## gemm_ukernel read uninitialized registers on the second block.
  const
    TILE_K = 32
    BLK_K = 16
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
    acc.tf32Reference(A_gpu, B_gpu, TILE_M, TILE_N, TILE_K, 0.0'f32)
    const alpha = 1.0'f32
    const beta = 0.0'f32
    var C_ref = newSeq[float32](TILE_M * TILE_N)
    for i in 0 ..< TILE_M * TILE_N:
      C_ref[i] = alpha * acc[i]

    var gpuC = newSeq[float32](TILE_M * TILE_N)
    for i in 0 ..< TILE_M * TILE_N:
      gpuC[i] = 0x7FC00000'f32    # NaN sentinel — a spurious C read fails
    nv.execute("gemmTiledKernelK32", dim3(1), dim3(blockSize), gpuC,
               (A_gpu, B_gpu, alpha, beta))
    allClose(gpuC, C_ref, TILE_M, TILE_N, "trial " & $trial)

  echo "  OK — gemm_tiled 2 k-tiles (K=32, BLK_K=16) matches reference within 1e-4 (tf32-exact fixture, ", label, " atom, 16 trials, (1,0), NaN C)"

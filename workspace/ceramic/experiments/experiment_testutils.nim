## experiments_testutils: shared test infrastructure for ceramic experiments
##
## Contains naive reference implementations and validation helpers
## that are reused across all experiment files.

import std/[math, strformat, strutils, random]
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/runtime/engines
# The legacy NVRTC driver (initNvrtc/execute) is not re-exported by engines.nim
# anymore (clean engine API only) — import it directly; order matters:
# engines must be processed first so its `import ./engines/nvrtc {.all.}`
# sees a fully-processed nvrtc module (the engines ↔ nvrtc circular import
# only compiles in that direction); the direct import below is then cached.
import workspace/crucible/src/runtime/engines/nvrtc
# TODO(engine): this benchmark harness launches with a 2D grid
# (dim3(num_cta_m, num_cta_n)) — the 1D engine LaunchConfig cannot express
# it, so it stays on the internal NVRTC execute path on purpose.
import workspace/crucible/src/runtime/exec/cuda_runtime

# ═══════════════════════════════════════════════════════════════════════════
#  XOR hash — exact bit-level fingerprint
# ═══════════════════════════════════════════════════════════════════════════

proc xorHash(data: openArray[float32]): uint32 =
  result = 0
  for i in 0 ..< data.len:
    result = result xor cast[uint32](data[i])

# ═════════════════════════════════════════════════════════════════════════
#  Naive matmul (reference implementation)
# ═════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════

proc fmaf(a, b, c: float32): float32 {.importc: "fmaf", header: "<math.h>".}

proc naive_matmul[T](
    C: var openArray[T],
    A: openArray[T],
    B: openArray[T],
    M, N, K: int) =
  ## Naive O(M*N*K) matmul: C = A @ B.
  ## A is (M,K), B is (N, K), C is (M,N).  # CuTe convention: (M,K) x (N,K) => (M,N)
  ## Accumulates with fmaf in ascending-k order so the reference is
  ## bit-compatible with the GPU kernel (NVRTC compiles with fmad=true;
  ## a separate mul+add reference drifts by ulps and fails absTol=1e-4).
  doAssert A.len == M*K
  doAssert B.len == K*N
  doAssert C.len == M*N
  for m in 0 ..< M:
    for n in 0 ..< N:
      var sum: T = 0   # accumulator follows T (fmaf only for float32)
      for k in 0 ..< K:
        when T is float32:
          sum = fmaf(A[m + k * M], B[n + k * N], sum)
        else:
          sum += A[m + k * M] * B[n + k * N]
      C[m + n * M] = sum

# ═════════════════════════════════════════════════════════════════════════
#  Validation
# ═════════════════════════════════════════════════════════════════════════

proc allClose[T](testC: Tensor[T, _, _], refC: Tensor[T, _, _];
                    rel_tol: T = T(1e-5); abs_tol: T = T(1e-8)) =
  ## Compare two tensors element-wise with numpy.allclose semantics:
  ## |test - ref| <= abs_tol + rel_tol·|ref| (single condition, NaN never
  ## equal). Uses doAssert for immediate loud breakage.
  let M = testC.shape[0].toIntVal()
  let N = testC.shape[1].toIntVal()
  var maxAbsErr: T = T(0)
  for m in 0 ..< M:
    for n in 0 ..< N:
      let refVal = refC[m, n]
      let testVal = testC[m, n]
      let absErr = abs(testVal - refVal)
      if absErr > maxAbsErr: maxAbsErr = absErr
      doAssert absErr <= abs_tol + rel_tol * abs(refVal),
        &"FAIL at [{m},{n}]: got {testVal}, expected {refVal}, absErr={absErr}"
  echo &"    PASS (maxAbsErr={maxAbsErr:.2e})"

proc make_test_matrix(rng: var Rand; rows, cols: int): seq[float32] =
  ## Create rows×cols matrix with values in [0, 1] from seeded RNG.
  result = newSeq[float32](rows * cols)
  for i in 0 ..< result.len:
    result[i] = rng.rand(1.0'f32)

proc run_gemm_and_validate_colmajor*(
    kernelSource: string;
    kernelName: string) =
  ## Validate a GEMM kernel using column-major reference (CuTe convention).
  ##
  ## Test parameters are deterministically derived from the kernel identity:
  ##   seed = hash(source + name + 0x9e3779b97f4a7c15)
  ##
  ## Same kernel source + name always produces the same test suite.
  ##
  ## Kernel contract expected by this harness (shared by the CuTe tutorial
  ## sgemm_1/sgemm_2 ports):
  ##   - CTA tile 128x128x8, 256 threads, no bounds checks
  ##   - signature gemmKernel(C, A, B, M, N, K: ..., alpha, beta: float32)
  ##   - 2D grid (ceil_div(M,128), ceil_div(N,128)) with blockIdx.x/y,
  ##     threadIdx.x in [0, 256)
  ## M, N are randomized as multiples of 128, K as multiples of 8, alpha in
  ## [0.25, 4.0] — the launch config is the kernel's contract, not a knob.

  const numConfigs = 10
  const relTol = 1e-4'f32
  const absTol = 1e-4'f32

  # ── 1. Deterministic seed from kernel identity ──────────────────────
  const magic = 0x9e3779b97f4a7c15'u64
  const prime = 0x100000001b3'u64    # FNV-1a prime
  var h = magic
  for c in kernelSource:
    h = h xor uint64(c)
    h = h * prime
  for c in kernelName:
    h = h xor uint64(c)
    h = h * prime

  var rng = initRand(int64(h and 0x7FFFFFFFFFFFFFFF'u64))
  echo &"  seed=0x{h:016x}"

  # ── 2. Generate test configs ──────────────────────────────────────
  # The tutorial kernels have a FIXED CTA tile 128x128x8 with 256 threads
  # and no bounds checking, so shapes must be exact tile multiples and the
  # launch exactly 256 threads per block — those are the kernel's contract,
  # not free knobs (sgemm_1 and sgemm_2 both use these constants).
  const blkM = 128
  const blkN = 128
  const blkK = 8
  const threadsPerBlock = 256
  type TestConfig = tuple[M, N, K: int; alpha: float32]
  var configs = newSeq[TestConfig](numConfigs)
  for i in 0 ..< numConfigs:
    let M = rng.rand(1..8) * blkM          # multiple of the CTA tile
    let N = rng.rand(1..8) * blkN
    let K = rng.rand(8..128) * blkK        # multiple of the K-tile
    let alpha = rng.rand(0.25'f32..4.0'f32)
    configs[i] = (M, N, K, alpha)

  echo &"  Testing {numConfigs} configs (MxNxK  α):"
  for i, c in configs:
    echo &"    [{i+1:>2}/{numConfigs}]  {c.M:>4}x{c.N:>4}x{c.K:<4}  α={c.alpha:.3f}"

  # ── 3. Compile kernel (once) ──────────────────────────────────────
  var nv = initNvrtc(kernelSource)
  nv.compile()
  nv.getPtx()

  var overallMaxAbsErr: float32 = 0.0'f32

  # ── 4. Run each config ─────────────────────────────────────────────
  for i, (M, N, K, alpha) in configs:
    # 4a. Create fresh test data for this config
    let A = rng.make_test_matrix(M, K)
    let B = rng.make_test_matrix(N, K)

    # 4b. Compute reference result (scaled by alpha)
    var refC = newSeq[float32](M * N)
    refC.naive_matmul(A, B, M, N, K)
    if alpha != 1.0'f32:
      for j in 0 ..< M * N:
        refC[j] = alpha * refC[j]

    # 4c. Launch kernel (gpuC starts zeroed, beta=0)
    var gpuC = newSeq[float32](M * N)
    let num_cta_m = M div blkM            # exact: M is a tile multiple
    let num_cta_n = N div blkN            # exact: N is a tile multiple
    let m32 = int32(M)
    let n32 = int32(N)
    let k32 = int32(K)
    # 2D grid (cta_m, cta_n) x 256 threads — the launch contract the
    # kernel expresses via blockIdx.x/blockIdx.y and its 256-thread
    # layouts (shared by the sgemm_1/sgemm_2 ports).
    nv.execute(kernelName, dim3(num_cta_m, num_cta_n), dim3(threadsPerBlock),
               gpuC, (A, B, m32, n32, k32, alpha, 0.0'f32))

    # 4d. Validate against reference
    var cfgMaxAbs: float32 = 0.0'f32
    for j in 0 ..< M * N:
      let absErr = abs(gpuC[j] - refC[j])
      let relErr = absErr / max(1.0'f32, abs(refC[j]))
      if absErr > cfgMaxAbs: cfgMaxAbs = absErr
      doAssert relErr <= relTol and absErr <= absTol,
        &"Config [{i+1}/{numConfigs}] Element [{j mod M},{j div M}] FAIL: " &
        "gpu=" & $gpuC[j] & " ref=" & $refC[j] & " " &
        "absErr=" & $absErr & " relErr=" & $relErr

    if cfgMaxAbs > overallMaxAbsErr: overallMaxAbsErr = cfgMaxAbs
    echo &"  [{i+1:>2}/{numConfigs}]  PASS  maxAbsErr={cfgMaxAbs:.2e}"

  echo &"  ALL {numConfigs} PASS  worst maxAbsErr={overallMaxAbsErr:.2e}"

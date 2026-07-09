## experiments_testutils: shared test infrastructure for ceramic experiments
##
## Contains naive reference implementations and validation helpers
## that are reused across all experiment files.

import std/[math, strformat, strutils, random]
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/crucible/src/codegen/nvrtc

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

proc naive_matmul[T](
    C: var openArray[T],
    A: openArray[T],
    B: openArray[T],
    M, N, K: int) =
  ## Naive O(M*N*K) matmul: C = A @ B.
  ## A is (M,K), B is (N, K), C is (M,N).  # CuTe convention: (M,K) x (N,K) => (M,N)
  doAssert A.len == M*K
  doAssert B.len == K*N
  doAssert C.len == M*N
  for m in 0 ..< M:
    for n in 0 ..< N:
      var sum = 0'f32
      for k in 0 ..< K:
        sum += A[m + k * M] * B[n + k * N]
      C[m + n * M] = sum

# ═════════════════════════════════════════════════════════════════════════
#  Validation
# ═════════════════════════════════════════════════════════════════════════

proc allClose[T](testC: Tensor[T, _, _], refC: Tensor[T, _, _];
                    rel_tol: T = T(1e-5); abs_tol: T = T(1e-5)) =
  ## Compare two tensors element-wise with both relative and absolute tolerance.
  ## Equivalent to numpy.allclose. Uses doAssert for immediate loud breakage.
  let M = testC.shape[0].toIntVal()
  let N = testC.shape[1].toIntVal()
  var maxRelErr: T = T(0)
  var maxAbsErr: T = T(0)
  for m in 0 ..< M:
    for n in 0 ..< N:
      let refVal = refC[m, n]
      let testVal = testC[m, n]
      let absErr = abs(testVal - refVal)
      let relErr = absErr / max(T(1), abs(refVal))
      if relErr > maxRelErr: maxRelErr = relErr
      if absErr > maxAbsErr: maxAbsErr = absErr
      doAssert relErr <= rel_tol and absErr <= abs_tol,
        &"FAIL at [{m},{n}]: got {testVal}, expected {refVal}, absErr={absErr}, relErr={relErr}"
  echo &"    PASS (maxRelErr={maxRelErr:.2e}, maxAbsErr={maxAbsErr:.2e})"

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
  ## M, N, K, tile sizes, and threadsPerBlock are all randomized to ensure genericity

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
  type TestConfig = tuple[M, N, K, bsM, bsN, tpb: int; alpha: float32]
  var configs = newSeq[TestConfig](numConfigs)
  for i in 0 ..< numConfigs:
    let M = rng.rand(64..1024)
    let N = rng.rand(64..1024)
    let K = rng.rand(64..1024)
    let bsM = rng.rand(8..128)
    let bsN = rng.rand(8..128)
    let tpb = rng.rand(2..16) * 32
    let alpha = rng.rand(0.25'f32..4.0'f32)
    configs[i] = (M, N, K, bsM, bsN, tpb, alpha)

  echo &"  Testing {numConfigs} configs (MxNxK  bsMxbsN  tpb  α):"
  for i, c in configs:
    echo &"    [{i+1:>2}/{numConfigs}]  {c.M:>4}x{c.N:>4}x{c.K:<4}  " &
      &"bs={c.bsM:>3}x{c.bsN:<3}  tpb={c.tpb}  α={c.alpha:.3f}"

  # ── 3. Compile kernel (once) ──────────────────────────────────────
  var nv = initNvrtc(kernelSource)
  nv.compile()
  nv.getPtx()

  var overallMaxAbsErr: float32 = 0.0'f32

  # ── 4. Run each config ─────────────────────────────────────────────
  for i, (M, N, K, bsM, bsN, tpb, alpha) in configs:
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
    let num_cta_m = (M + bsM - 1) div bsM
    let num_cta_n = (N + bsN - 1) div bsN
    nv.numBlocks = int32(num_cta_m * num_cta_n)
    nv.threadsPerBlock = int32(tpb)
    let m32 = int32(M)
    let n32 = int32(N)
    let k32 = int32(K)
    nv.execute(kernelName, gpuC, (A, B, m32, n32, k32, alpha, 0.0'f32))

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

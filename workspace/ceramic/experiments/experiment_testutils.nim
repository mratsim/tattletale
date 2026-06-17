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

proc make_test_matrix(rows, cols: int; seed: int32 = 1): seq[float32] =
  ## Create rows×cols matrix with predictable values in (0, 1].
  result = newSeq[float32](rows * cols)
  let total = float32(rows * cols)
  var s = seed
  for i in 0 ..< rows:
    for j in 0 ..< cols:
      result[i * cols + j] = float32(s) / total
      s = s + 1

proc run_gemm_and_validate_colmajor*(
    kernelSource: string;
    kernelName: string;
    M, N, K: int;
    blockSizeM: int = 128;
    blockSizeN: int = 128;
    threadsPerBlock: int = 256;
    alpha: float32 = 1.0'f32;
    beta: float32 = 0.0'f32;
    relTol: float32 = 1e-4'f32;
    absTol: float32 = 1e-4'f32) =
  ## Validate a GEMM kernel using column-major reference.
  let A = make_test_matrix(M, K, 1)
  let B = make_test_matrix(N, K, 100)  # CuTe: (N,K) — N rows, K cols
  var refC = newSeq[float32](M * N)
  refC.naive_matmul(A, B, M, N, K)
  var hCinit = newSeq[float32](M * N)
  if beta != 0.0'f32:
    hCinit = make_test_matrix(M, N, 200)
  for i in 0 ..< M * N:
    refC[i] = alpha * refC[i] + beta * hCinit[i]
  var gpuC = newSeq[float32](M * N)
  for i in 0 ..< M * N:
    gpuC[i] = beta * hCinit[i]
  var nv = initNvrtc(kernelSource)
  let num_cta_m = (M + blockSizeM - 1) div blockSizeM
  let num_cta_n = (N + blockSizeN - 1) div blockSizeN
  nv.numBlocks = int32(num_cta_m * num_cta_n)
  nv.threadsPerBlock = int32(threadsPerBlock)
  nv.compile()
  nv.getPtx()
  let m32 = int32(M)
  let n32 = int32(N)
  let k32 = int32(K)
  nv.execute(kernelName, gpuC, (A, B, m32, n32, k32, alpha, beta))
  var maxAbsErr: float32 = 0.0'f32
  for i in 0 ..< M * N:
    let absErr = abs(gpuC[i] - refC[i])
    let relErr = absErr / max(1.0'f32, abs(refC[i]))
    if absErr > maxAbsErr: maxAbsErr = absErr
    doAssert relErr <= relTol and absErr <= absTol,
      "Element [" & $(i mod M) & "," & $(i div M) & "] FAIL: " &
      "gpu=" & $gpuC[i] & " ref=" & $refC[i] & " " &
      "absErr=" & $absErr & " relErr=" & $relErr
  echo "  PASS: " & $M & "x" & $N & "x" & $K & " maxAbsErr=" & $maxAbsErr
  # Print random 3x3 patch seeded from xor-hash of computed data
  let seed = xorHash(gpuC) xor xorHash(refC)
  var rng = initRand(int64(seed))
  let pr = rng.rand(max(0, M - 3))
  let pc = rng.rand(max(0, N - 3))
  echo "  3x3 patch @(" & $pr & "," & $pc & ")  hash=" & $seed & ":"
  echo "  " & repeat('-', 74)
  for dr in 0 ..< 3:
    let row = pr + dr
    stdout.write "  row " & align($row, 3) & " GPU:"
    for dc in 0 ..< 3:
      let col = pc + dc
      stdout.write &" {gpuC[row + col * M]:12.6f}"
    stdout.write "\n"
    stdout.write "          REF:"
    for dc in 0 ..< 3:
      let col = pc + dc
      stdout.write &" {refC[row + col * M]:12.6f}"
    stdout.write "\n"

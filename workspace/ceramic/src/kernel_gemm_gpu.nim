## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## GPU-suitable GEMM kernel: outer product via flat-index iteration.
##
## Implements the sgemm_1 dispatch [3] net effect:
##   C[m,n] += A[m,k] * B[n,k]
##
## Preconditions (not checked at runtime):
##   size<0>(A) == size<0>(C)  — M dimension matches
##   size<0>(B) == size<1>(C)  — N dimension matches
##   size<1>(A) == size<1>(B)  — K dimension matches

import ./int_tuples
import ./layouts
import ./tensors

template gemm*[T, ShA, StA, ShB, StB, ShC, StC](
    C: var TensorView[T, ShC, StC],
    A: TensorView[T, ShA, StA],
    B: TensorView[T, ShB, StB]) =
  ## Outer product: C[m,n] += A[m,k] * B[n,k]
  ## CuTe dispatch [3] net effect: (M,K) × (N,K) ⇒ (M,N)
  ## Acceptable on GPU, slow on CPU.
  const
    M = ShC.default[0]
    N = ShC.default[1]
    K = ShA.default[1]
  when typeof(ShA.default[0]) isnot typeof(M):
    {.error: "gemm: A mode 0 (M) != C mode 0".}
  when typeof(ShB.default[0]) isnot typeof(N):
    {.error: "gemm: B mode 0 (N) != C mode 1".}
  when typeof(ShA.default[1]) isnot typeof(K):
    {.error: "gemm: A mode 1 (K) != B mode 1".}
  for k in 0 ..< K:
    for m in 0 ..< M:
      for n in 0 ..< N:
        C[m, n] += A[m, k] * B[n, k]

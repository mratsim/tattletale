## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## GPU-suitable AXPBY kernel: epilogue via flat-index iteration.
##
##   Y = α·X + β·Y   (element-wise)
##
## Parameter order follows the function name: alpha, X, beta, Y.
## Exception to output-first convention — the name `axpby` is the mnemonic.
##
## Precondition: X and Y must have the same logical size.

import ./int_tuples
import ./layouts
import ./tensors

template axpby*[T, ShX, StX, ShY, StY](
    alpha: T,
    X: TensorView[T, ShX, StX] or Tensor[T, ShX, StX],
    beta: T,
    Y: var (TensorView[T, ShY, StY] or Tensor[T, ShY, StY])) =
  ## Y = α·X + β·Y   (element-wise)
  ## Parameter order: `axpby` → alpha, X, beta, Y
  ## CuTe: axpby(alpha, x, beta, y)
  ## Acceptable on GPU, slow on CPU.
  for i in 0 ..< size(Y.layout):
    Y(i) = alpha * X(i) + beta * Y(i)

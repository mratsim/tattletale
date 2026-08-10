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

func axpby*[T, ShX, StX, ShY, StY](
    alpha: T,
    X: TensorView[T, ShX, StX] or Tensor[T, ShX, StX],
    beta: T,
    Y: var (TensorView[T, ShY, StY] or Tensor[T, ShY, StY])) {.inline.} =
  ## Y = α·X + β·Y   (element-wise)
  ## Parameter order: `axpby` → alpha, X, beta, Y
  ## CuTe: axpby(alpha, x, beta, y)
  ##
  ## A func (not a template): alpha/beta are evaluated once, and the
  ## element loop is branch-free per call (the scale-factor dispatch is
  ## hoisted out — ex02a_matmul_handtuned's genEpilogue pattern).
  ##
  ## X and Y are zipped by their common logical size — each is indexed
  ## through its own layout (CuTe: zip(x, y) then elementwise). This is
  ## what makes the epilogue axpby(alpha, cFrag, beta, C) work: the
  ## register fragment and the gmem fragment have different layouts but
  ## equal size.
  ##
  ## Specializations (runtime branches, same op order α·X + β·Y — two
  ## multiplies then one add, no fma):
  ##   beta == 0 → the Y read is skipped entirely (Y(i) never evaluated —
  ##     a NaN-prefilled C stays untouched, mirroring the β=0 skip-read)
  ##   alpha == 1 → the α multiply is skipped
  ##   beta == 1 → the β multiply is skipped
  ##   (when alpha == 1 and beta == 1 both hold, the alpha == 1 branch
  ##   wins — the result is identical)
  static:
    doAssert toIntVal(size(X.layout)) == toIntVal(size(Y.layout)),
      "axpby: X and Y must have the same logical size — the zip loop iterates" &
      " size(Y.layout) and indexes X(i) with the same i, so an X smaller than Y" &
      " reads out of bounds past X's register array (epilogue zip contract —" &
      " RID HIDN-A-003/HIDN-B-004/HPC-A-003)"
  if beta == T(0):
    for i in 0 ..< size(Y.layout):
      Y(i) = alpha * X(i)
  elif alpha == T(1):
    for i in 0 ..< size(Y.layout):
      Y(i) = X(i) + beta * Y(i)
  elif beta == T(1):
    for i in 0 ..< size(Y.layout):
      Y(i) = alpha * X(i) + Y(i)
  else:
    for i in 0 ..< size(Y.layout):
      Y(i) = alpha * X(i) + beta * Y(i)

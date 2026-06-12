## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## GPU-suitable fill kernel: divmod-based flat-index iteration.
##
## Uses flat-index iteration (`tv(i) = val`) which calls `crd2idx`
## per element. Acceptable on GPU, slow on CPU.
##
## On CPU, use `kernel_fillwith_cpu` which uses contiguity-fused
## nimSetMem for zero-fill and nested stride-based loops otherwise.

import ./int_tuples
import ./layouts
import ./tensors

{.experimental: "callOperator".}

template fillWith*[T, Sh, St](tv: var TensorView[T, Sh, St]; val: T) =
  ## Set every logical element of `tv` to `val`.
  ## Uses flat-index iteration — acceptable on GPU, slow on CPU.
  let n = size(tv.layout)
  for i in 0 ..< n:
    tv(i) = val

template fillWith*[T, Sh, St](t: var Tensor[T, Sh, St]; val: T) =
  ## Set every logical element of `t` to `val`.
  let n = size(t.layout)
  for i in 0 ..< n:
    t(i) = val

# Constantine
# Copyright (c) 2018-2019    Status Research & Development GmbH
# Copyright (c) 2020-Present Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#            Public pragmas
#
# ############################################################

template global*() {.pragma.}
template workgroup*(size: untyped): untyped {.pragma.}
  ## Workgroup size for a kernel: `{.workgroup: (X, Y, Z).}` sets the threads per threadgroup.

template device*() {.pragma.}
template const_mem*(): untyped {.pragma.}
  ## Allocates a GPU-block var in read-only constant memory (`constant`/`uniform`/`__constant__`).

template smem*(): untyped {.pragma.}
  ## Allocates a GPU-block var in threadgroup-scope memory (`threadgroup`/`workgroup`/`__shared__`).

# ############################################################
#
#            Private pragmas
#
# ############################################################

template builtin*(kind: untyped = nil) {.pragma.}

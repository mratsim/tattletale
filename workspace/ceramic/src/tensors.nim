# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Curated public surface of the tensor tier.
## One import covers the CuTe tensor.hpp vocabulary, the sections mirror
## the Demystifying CuTe walkthrough.
##
## | Symbols                                                               |
## | --------------------------------------------------------------------- |
## | `TensorOwned`, `TensorView`, `AnyTensor`                              |
## | `make_tensor`, `make_tensor_like`, `make_view`, `view`                |
## | `layout`, `shape`, `stride`, `rank`, `size`, `cosize`                 |
## | `()`, `[]`, `[]=`, `slice`, `displace`                                |
## | `inner_partition`, `outer_partition`, `local_tile`, `local_partition` |
## | `$`                                                                   |
##
## Excluded names and their scope:
##
## | Scope                                          |
## | ---------------------------------------------- |
## | private selector helper of layout_indexing.nim |
## | internal tuple-building macro                  |
## | `crd2idx`, `make_layout`, `Int`, `+%`          |
## | consumed by the tensor tier, not re-exported   |

import
  workspace/ceramic/src/tensors/tensor_datatypes,
  workspace/ceramic/src/tensors/tensor_selectors

# ═══════════════════════════════════════════════════════════════
#  Datatypes, construction, accessors
# ═══════════════════════════════════════════════════════════════

# Everything in these modules is public surface:
# - the tensor types (TensorOwned, TensorView, AnyTensor)
# - constructors, accessors, operators and display
export tensor_datatypes
export tensor_selectors

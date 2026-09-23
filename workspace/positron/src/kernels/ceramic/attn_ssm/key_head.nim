## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#     Key-head mapping (attn_ssm local extension)
#
# ############################################################

## Grouped-query key-head mapping, the one key-head derivation the attention
## and state-space kernels share:
## - one query head reads one key head
## - the key head is `queryHead div (numQueryHeads div numKeyHeads)`
##
## Register-tile naming convention, shared by the gdn and kda kernels:
##
## | name    | holds                                                            |
## | ------- | ---------------------------------------------------------------- |
## | `<x>T`  | the element-dtype register tile of operand x, loaded from memory |
## | `<x>32` | the fp32 register tile of the same operand                       |
##
## An fp32-storage operand loads straight into its `32` form, an element-dtype
## operand widens its `T` form into the `32` form.
##
## A template, so the arithmetic inlines into host launcher code and {.device.} kernel bodies alike.

template keyHeadOf*(queryHead, numQueryHeads, numKeyHeads: int32): int32 =
  ## Key head of a query head under grouped-query attention, `queryHead div (numQueryHeads div numKeyHeads)`.
  ##
  ## Contract:
  ## - numKeyHeads > 0, numQueryHeads an exact multiple of numKeyHeads,
  ##   the callers assert at the launcher
  ##
  ## - worked example, 32 query heads over 8 key heads, query head 19 → key head 4
  queryHead div (numQueryHeads div numKeyHeads)

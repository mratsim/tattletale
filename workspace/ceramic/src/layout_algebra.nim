# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Curated public surface of the layout algebra.
## One import covers the whole CuTe-style layout-algebra vocabulary.
## The sections mirror the Demystifying CuTe walkthrough.
##
## | Section               | Symbols                                                                        |
## | --------------------- | ------------------------------------------------------------------------------ |
## | Datatypes             | `Layout`, `Int`, `size`, `cosize`, `rank`                                      |
## | Construction          | `make_layout` and friends                                                      |
## | Views and selectors   | `mode`, `groupModes`, `zipModes`                                               |
## | Indexing              | `crd2idx`, `idx2crd`, `slice`, `dice`, `X`/`Y`/`_`                             |
## | Algebra               | `coalesce`, `filter_zeros`, `filter_inactive`, `complement`, `compose`         |
## | Partitioning          | `logical_divide`, `zipped_divide`, `tiled_divide`, `flat_divide`, `tile_unzip` |
## | Inverses and analysis | `right_inverse`, `left_inverse`, `max_common_layout`, `max_common_vector`      |
## | Products              | `logical_product` through `tile_to_shape`                                      |
## | Pointer arithmetic    | `+%`                                                                           |
##
## Excluded names and their scope:
##
## | Excluded                           | Scope                                   |
## | ---------------------------------- | --------------------------------------- |
## | `getIndicesSortedByStride`         | stride-sorted index permutation         |
## | `complementScalar`/`Multi`/`Gaps`  | complement backends                     |
## | `rightInverseChain`, `composeImpl` | inverse-chain and composition internals |
## | `zipped_divide_builder`            | divide builder                          |
## | `unwrap`, `buildStride`            | macro helpers                           |
## | `divisibilityCheck`, `LayoutCT`    | macro helpers                           |

import
  workspace/ceramic/src/int_tuples,
  workspace/ceramic/src/layout_algebra/layouts_datatypes,
  workspace/ceramic/src/layout_algebra/layout_constructors,
  workspace/ceramic/src/layout_algebra/layouts,
  workspace/ceramic/src/layout_algebra/layout_indexing,
  workspace/ceramic/src/layout_algebra/layout_algebra,
  workspace/ceramic/src/layout_algebra/ptr_arithmetic

# ═══════════════════════════════════════════════════════════════
#  Datatypes
# ═══════════════════════════════════════════════════════════════

# Everything in these modules is public surface:
# the Int static-integer type, IntTuple ops (product_each, flatten, zip),
# the Layout type, its predicates and its selectors.
export int_tuples
export layouts_datatypes

# ═══════════════════════════════════════════════════════════════
#  Construction, the constructors only
#  (the LayoutCT emitter machinery is not part of the surface)
# ═══════════════════════════════════════════════════════════════

export layout_constructors.col_major_strides
export layout_constructors.make_layout
export layout_constructors.compact_order
export layout_constructors.make_layout_like
export layout_constructors.make_fragment_like

# ═══════════════════════════════════════════════════════════════
#  Views, selectors, indexing
# ═══════════════════════════════════════════════════════════════

export layouts.mode
export layouts.isCompact
export layouts.filter_zeros
export layouts.padRight
export layouts.padLeft
export layouts.mapLeavesWith
export layouts.upcast
export layouts.downcast
export layouts.groupModes
export layouts.takeModes
export layouts.selectModes
export layouts.replaceMode
export layouts.zipModes
export layouts.zipModesWith

# crd2idx, idx2crd, slice, dice, X/Y markers, call operator.
# `hasUnderscoreImpl` is a private helper of layout_indexing.nim
# and does not appear on the curated surface.
export layout_indexing except hasUnderscoreImpl

# ═══════════════════════════════════════════════════════════════
#  Algebra: coalesce, filter, complement, compose
# ═══════════════════════════════════════════════════════════════

## Merge contiguous modes whose strides form a compact run.
export layout_algebra.coalesce

## Zero-out stride-0 modes, their shapes become Int[1].
export layouts.filter_zeros

## Drop inactive modes, size-1 shapes with stride 0.
export layout_algebra.filter_inactive

## Complement of a layout, the stride-space layout covering
## every offset `layout` leaves unused.
export layout_algebra.complement

## Compose two layouts, function composition threaded mode by mode.
export layout_algebra.compose

# ═══════════════════════════════════════════════════════════════
#  Partitioning: divide a layout by a tiler
# ═══════════════════════════════════════════════════════════════

## Logical division, the CuTe ⊙ operator,
## with result `(rest, tile) = layout ⋅ tiler`.
export layout_algebra.logical_divide

## logical_divide with tile and rest modes zipped, mode-interleaved
## for hierarchical traversal.
export layout_algebra.zipped_divide

## logical_divide with the tile modes grouped.
export layout_algebra.tiled_divide

## logical_divide with the rest modes flattened per mode.
export layout_algebra.flat_divide

## logical_divide with tile and rest unzipped afterwards.
export layout_algebra.tile_unzip

# ═══════════════════════════════════════════════════════════════
#  Inverses and common-layout analysis
# ═══════════════════════════════════════════════════════════════

## Right inverse `b` such that `layout ∘ b` is compact.
export layout_algebra.right_inverse

## Left inverse `b` such that `b ∘ layout` is compact.
export layout_algebra.left_inverse

## Largest mode-wise common sub-layout of two layouts.
export layout_algebra.max_common_layout

## Largest vector width common to two layouts, their contiguous size.
export layout_algebra.max_common_vector

# ═══════════════════════════════════════════════════════════════
#  Products: extend a block layout by a tiler layout
# ═══════════════════════════════════════════════════════════════

## Logical product, the CuTe ⊗ operator.
## Appends the tiler modes after the block modes.
export layout_algebra.logical_product

## Logical product with the tiler nested inside each block mode.
export layout_algebra.nested_product

## Logical product with block and tiler modes zipped.
export layout_algebra.zipped_product

## zipped_product with the tile modes then grouped.
export layout_algebra.tiled_product

## zipped_product with the tiler modes then flattened per mode.
export layout_algebra.flat_product

## Product with tiler modes interleaved as blocked sub-blocks.
export layout_algebra.blocked_product

## Product with tiler modes interleaved as raked sub-blocks.
export layout_algebra.raked_product

## Rebuild a block layout to tile a target shape, LayoutLeft-ordered
## unless `ord_shape` says otherwise.
export layout_algebra.tile_to_shape

# ═══════════════════════════════════════════════════════════════
#  Pointer arithmetic
# ═══════════════════════════════════════════════════════════════

## Element-count pointer increment (openArray variant slices), the `+%` operator.
export ptr_arithmetic

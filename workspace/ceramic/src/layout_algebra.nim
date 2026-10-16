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
## | Section               | Symbols                                                                                                                     |
## | --------------------- | --------------------------------------------------------------------------------------------------------------------------- |
## | Datatypes             | `Layout`, `Int`, `size`, `cosize`, `rank`                                                                                   |
## | Construction          | `col_major_strides`, `make_layout`, `make_ordered_layout`, `make_layout_like`, `make_identity_layout`, `make_fragment_like` |
## | Indexing              | `crd2idx`, `idx2crd`, `slice`, `dice`, `X`/`Y`/`_`                                                                          |
## | Algebra               | `coalesce`, `complement`, `compose`                                                                                         |
## | Partitioning          | `logical_divide`, `zipped_divide`, `tiled_divide`, `flat_divide`                                                            |
## | Inverses and analysis | `right_inverse`, `left_inverse`                                                                                             |
## | Products              | `logical_product`, `zipped_product`, `tiled_product`, `flat_product`, `blocked_product`, `raked_product`                    |

import
  workspace/ceramic/src/int_tuples,
  workspace/ceramic/src/layout_algebra/layouts_datatypes,
  workspace/ceramic/src/layout_algebra/layout_constructors,
  workspace/ceramic/src/layout_algebra/layout_compiletime,
  workspace/ceramic/src/layout_algebra/layouts,
  workspace/ceramic/src/layout_algebra/layout_indexing,
  workspace/ceramic/src/layout_algebra/ism_coord_strides,
  workspace/ceramic/src/layout_algebra/layout_algebra

# ═══════════════════════════════════════════════════════════════
#  Datatypes
# ═══════════════════════════════════════════════════════════════

export int_tuples
export ism_coord_strides
export layouts_datatypes.Layout
export layouts_datatypes.`===`
export layouts_datatypes.`$`
export layouts_datatypes.can_group_a_into_b
export layouts_datatypes.compatible
export layouts_datatypes.congruent
export layouts_datatypes.cosize
export layouts_datatypes.rank
export layouts_datatypes.size
export layouts_datatypes.StrideOrder
export layouts_datatypes.weakly_congruent

# ═══════════════════════════════════════════════════════════════
#  Construction
# ═══════════════════════════════════════════════════════════════

export layout_constructors.col_major_strides
export layout_constructors.make_layout
export layout_constructors.make_ordered_layout
export layout_constructors.make_layout_like
export layout_constructors.make_identity_layout
export layout_constructors.make_fragment_like

# ═══════════════════════════════════════════════════════════════
#  Indexing
# ═══════════════════════════════════════════════════════════════

export layout_indexing.crd2idx
export layout_indexing.crd2idxImpl
export layout_indexing.idx2crd
export layout_indexing.inBounds
export layout_indexing.hasUnderscore
export layout_indexing.`()`
export layout_indexing_slicedice.X
export layout_indexing_slicedice.Y
export layout_indexing_slicedice._
export layout_indexing_slicedice.`*`
export layout_indexing_slicedice.slice
export layout_indexing_slicedice.dice

# ═══════════════════════════════════════════════════════════════
#  Algebra
# ═══════════════════════════════════════════════════════════════

export layout_algebra.coalesce
export layout_algebra.complement
export layout_algebra.compose

# ═══════════════════════════════════════════════════════════════
#  Partitioning
# ═══════════════════════════════════════════════════════════════

export layout_algebra.logical_divide
export layout_algebra.zipped_divide
export layout_algebra.tiled_divide
export layout_algebra.flat_divide

# ═══════════════════════════════════════════════════════════════
#  Inverses and analysis
# ═══════════════════════════════════════════════════════════════

export layout_algebra.right_inverse
export layout_algebra.left_inverse

# ═══════════════════════════════════════════════════════════════
#  Products
# ═══════════════════════════════════════════════════════════════

export layout_algebra.logical_product
export layout_algebra.zipped_product
export layout_algebra.tiled_product
export layout_algebra.flat_product
export layout_algebra.blocked_product
export layout_algebra.raked_product

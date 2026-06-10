## Experimental layout operations — proposed additions to CuTe's layout algebra.
##
## These operations are not (yet) in the CuTe reference but follow its
## conventions. They may be proposed upstream in the future.

import ./layouts

# ═══════════════════════════════════════════════════════════════════════════
#  nested_product — categorical product preserving sub-layout grouping
# ═══════════════════════════════════════════════════════════════════════════
#
#  CuTe has:
#    logical_product   — flat concatenation of modes
#    blocked_product   — interleaved product (zip modes)
#    raked_product     — swapped interleaved product
#
#  Missing: a product that NESTS each argument's modes as a sub-tuple.
#  nested_product fills this gap.
#
#  Given:
#    A: (a0, a1, ...):(sa0, sa1, ...)
#    B: (b0, b1, ...):(sb0, sb1, ...)
#
#  Returns:
#    ((a0, a1, ...), (b0, b1, ...)) : ((sa0, sa1, ...), (sb0, sb1, ...))
#
#  The resulting layout maps ((c0, c1, ...), (d0, d1, ...)) to
#    c0*sa0 + c1*sa1 + ... + d0*sb0 + d1*sb1 + ...
#
#  This is the categorical product of two layout morphisms (product type
#  of their coordinate spaces, additive sum of their index results).
# ═══════════════════════════════════════════════════════════════════════════

func nested_product*[A, B: Layout](a: A; b: B): auto =
  ## Categorical product of two layouts, preserving each argument's mode grouping.
  make_layout((a.shape, b.shape), (a.stride, b.stride))

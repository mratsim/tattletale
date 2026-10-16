# Test: layout_indexing, inBounds
# Run: nim c -r workspace/ceramic/tests/layout/t_layout_in_bounds.nim
#
# Expected results cross-checked against pycute `in_bounds`
# (ground truth: gen_in_bounds_cases.py in the op scratchspace).

import workspace/ceramic/src/layout_algebra

# ═══════════════════════════════════════════════════════════════
#  inBounds — scalar shape
# ═══════════════════════════════════════════════════════════════

block:
  doAssert     7.inBounds(8)
  doAssert not 8.inBounds(8)
  doAssert not (-1).inBounds(8)
  doAssert     0.inBounds(8)

# ═══════════════════════════════════════════════════════════════
#  inBounds — tuple shape, tuple coord
# ═══════════════════════════════════════════════════════════════

block:
  doAssert     (2, 5).inBounds((4, 8))
  doAssert     (0, 0).inBounds((4, 8))
  doAssert     (3, 7).inBounds((4, 8))
  doAssert not (4, 0).inBounds((4, 8))
  doAssert not (0, 8).inBounds((4, 8))
  doAssert not (-1, 0).inBounds((4, 8))
  doAssert not (2, -1).inBounds((4, 8))
  doAssert not (4, 8).inBounds((4, 8))

block:
  doAssert     (1, 2, 3).inBounds((3, 4, 5))
  doAssert not (3, 4, 5).inBounds((3, 4, 5))
  doAssert not (2, 4, 0).inBounds((3, 4, 5))

# ═══════════════════════════════════════════════════════════════
#  inBounds — nested shapes
# ═══════════════════════════════════════════════════════════════

block:
  doAssert     (1, (0, 7)).inBounds((4, (2, 8)))
  doAssert not (1, (2, 7)).inBounds((4, (2, 8)))
  doAssert not (1, (0, 8)).inBounds((4, (2, 8)))
  doAssert not (0, (1, 8)).inBounds((4, (2, 8)))
  doAssert not (4, (1, 0)).inBounds((4, (2, 8)))
  doAssert not (-1, (0, 0)).inBounds((4, (2, 8)))

# ═══════════════════════════════════════════════════════════════
#  inBounds — scalar coord, decomposed, no wrap on the last leaf
# ═══════════════════════════════════════════════════════════════

block:
  doAssert     0.inBounds((4, 8))
  doAssert     31.inBounds((4, 8))
  doAssert not 32.inBounds((4, 8))
  doAssert not 63.inBounds((4, 8))
  doAssert not (-1).inBounds((4, 8))

block:
  doAssert     7.inBounds((3, 4, 5))
  doAssert     59.inBounds((3, 4, 5))
  doAssert not 60.inBounds((3, 4, 5))

block:
  doAssert     31.inBounds((4, (2, 8)))
  doAssert     32.inBounds((4, (2, 8)))
  doAssert     63.inBounds((4, (2, 8)))
  doAssert not 64.inBounds((4, (2, 8)))

block:
  doAssert     255.inBounds((8, 8, 4))
  doAssert not 256.inBounds((8, 8, 4))

# ═══════════════════════════════════════════════════════════════
#  inBounds — scalar coord element into a nested dimension
# ═══════════════════════════════════════════════════════════════

block:
  doAssert     (1, 5).inBounds((4, (2, 8)))
  doAssert not (1, 17).inBounds((4, (2, 8)))
  doAssert not (1, -1).inBounds((4, (2, 8)))
  doAssert not (1, 16).inBounds((4, (2, 8)))

# ═══════════════════════════════════════════════════════════════
#  inBounds — layout overload, against the layout's shape
# ═══════════════════════════════════════════════════════════════

block:
  let L = make_layout((4, 8), (1, 4))
  doAssert     (2, 5).inBounds(L)
  doAssert not (0, 8).inBounds(L)
  doAssert     31.inBounds(L)
  doAssert not 32.inBounds(L)

# ═══════════════════════════════════════════════════════════════
#  inBounds — runtime coords, the predicate stays a runtime bool
# ═══════════════════════════════════════════════════════════════

block:
  var c = 7
  doAssert     c.inBounds(8)
  doAssert not (c + 1).inBounds(8)

# ═══════════════════════════════════════════════════════════════
#  inBounds — static inputs fold at compile time
# ═══════════════════════════════════════════════════════════════

block:
  const foldedTrue = (2, 5).inBounds((4, 8))
  const foldedFalse = (0, 8).inBounds((4, 8))
  const foldedScalar = 32.inBounds((4, (2, 8)))
  doAssert     foldedTrue
  doAssert not foldedFalse
  doAssert     foldedScalar

echo "t_layout_in_bounds: all green"

# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Curated public surface of tattletile's layout algebra.
## The transformation layer resolves Tile IR layouts in the passes.
##
## One import covers the whole vocabulary:
##
## - hierarchical tuples and stride scalars (`htuple`)
## - layouts, their evaluation, size and cosize (`layout`)
## - the algebra ops: `compose`, `logicalDivide`/`zippedDivide`, `complement`,
##   `rightInverse`, `coalesce`, `logicalProduct`
## - swizzle concretization: F2 arithmetic, the Swizzle functor, swizzled
##   layouts over shared-memory tiles (`swizzle`)
##
## Algorithms are ported from the frozen python reference implementation
## (vendored pycute) so Nim outputs match it exactly.
##
## Run:
##   nim c -r --hints:off --warnings:off --nimcache:nimcache/b4 \
##     --outdir:build/b4 workspace/tattletile/tests/test_layout_algebra_conformance.nim

import ./layout_algebra/htuple
import ./layout_algebra/layout
import ./layout_algebra/algebra
import ./layout_algebra/swizzle

export htuple
export layout
export algebra
export swizzle

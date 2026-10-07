# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Run:
#   nim c -r --hints:off --warnings:off \
#     --outdir:build/wip --nimcache:nimcache/wip \
#     workspace/ceramic/tests/layout_algebra/manual_compose_reject_on_divisibility.nim


import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/layout_constructors

# TODO
#   A statically violating composition is a compile-time error,
static:
  doAssert not compiles(compose(make_layout((2, 3), (2, 1)), make_layout(6, -1)))
  doAssert compiles(compose(make_layout((2, 3), (2, 1)), make_layout(6, 1)))

echo "ALL TESTS PASSED"

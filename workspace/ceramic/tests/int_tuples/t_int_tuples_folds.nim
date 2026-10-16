# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option.
# This file may not be copied, modified, or distributed except according to those terms.

# Run: nim c -r --hints:off --warnings:off --outdir:build/wip workspace/ceramic/tests/int_tuples/t_int_tuples_folds.nim

import workspace/ceramic/src/int_tuples {.all.}

# ═══════════════════════════════════════════════════════════════
#  prefix product scan
# ═══════════════════════════════════════════════════════════════

proc runPrefixProductScanTests =
  block:
    # literal tuple seed
    doAssert prefix_product_scan(((2, 3), 4), (8, 1)) === ((8, 16), 1)
  block:
    # a symbol-bound tuple seed matches the literal seed
    let s = (8, 1)
    doAssert prefix_product_scan(((2, 3), 4), s) == ((8, 16), 1)
  echo "  prefix product scan: 2 cases OK"

# ═══════════════════════════════════════════════════════════════
#  suffix product scan
# ═══════════════════════════════════════════════════════════════

proc runSuffixProductScanTests =
  block:
    # flat container, the leaves after each leaf multiply the seed
    doAssert suffix_product_scan((2, 3, 4), 1) === (12, 4, 1)
  block:
    # nested container, per-mode seeds scan per dimension
    doAssert suffix_product_scan(((2, 3), 4), (1, 1)) === ((3, 1), 1)
  echo "  suffix product scan: 2 cases OK"

runPrefixProductScanTests()
runSuffixProductScanTests()
echo "ALL INT_TUPLES_FOLDS TESTS PASSED"

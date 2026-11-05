# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ./builtins_pragmas
import ./builtins_catalog

# ═══════════════════════════════════════════════════════════════════════
#   Atomics
# ═══════════════════════════════════════════════════════════════════════

proc atomicFetchAdd*[T](p: ptr T, v: T): T {.builtin: cbkAtomicFetchAdd.} = discard

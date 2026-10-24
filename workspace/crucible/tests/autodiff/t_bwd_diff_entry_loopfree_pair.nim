# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Loop-free kernel pair: pair derivation without an index loop
##
## Test that `bkMetal.bwd_diff` expands a kernel with no index loop and no float scalar parameters.
##
## Such a kernel has no scalar cotangent, hence no reduce-partials
## workspace and no reduce kernel:
##
## - the pair derives with zero scalar workspaces and zero reduce kernels
## - the backward signature is the d_-block plus the forward signature verbatim
## - the backward scatters the output cotangent into the read buffers' shadows
##
## Run:
##   nim c -r --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests/autodiff --nimcache:nimcache/tests/autodiff \
##     workspace/crucible/tests/autodiff/t_bwd_diff_entry_loopfree_pair.nim

import std/strutils
import workspace/crucible

# ── Proof kernel ──────────────────────────────────────────────────────────

proc pairAddPlain(y, a, b: ptr UncheckedArray[float32]) {.global, workgroup: (1, 1).} =
  ## One thread adds two buffered slots: `y[0] = a[0] + b[1]`.
  ## No loop, no scalar parameter.
  y[0] = a[0] + b[1]

const pairAddPair = bkMetal.bwd_diff(pairAddPlain)

# ── Pair metadata and backward structure ──────────────────────────────────

static:
  doAssert pairAddPair.kernelName == "pairAddPlain"
  doAssert pairAddPair.backend == bkMetal

  # Backward signature: the d_-block in forward order, block 2 verbatim.
  doAssert pairAddPair.bwdSignature == @["d_y", "d_a", "d_b", "y", "a", "b"]

  # No float scalar parameter, so no scalar cotangent: no reduce-partials workspace, no reduce kernel.
  doAssert pairAddPair.scalarWorkspaces.len == 0
  doAssert pairAddPair.reduceSources.len == 0

  # The backward reads only the gradient shadows, so the printer prunes
  # every unread block-2 primal from the binding list.
  doAssert pairAddPair.bwdKernelParams == @["d_y", "d_a", "d_b"]

  # The output cotangent scatters into the read buffers' shadows.
  doAssert "d_a[0] = (d_a[0] + d_y[0]);" in pairAddPair.bwdSource
  doAssert "d_b[1] = (d_b[1] + d_y[0]);" in pairAddPair.bwdSource

echo "  OK — loop-free pair: zero scalar workspaces, no reduce kernel"

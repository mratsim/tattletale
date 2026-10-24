# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Mixed-float widths: the scalar's type is the partials width authority
##
## A kernel mixing widths (float32 buffers, float64 scalar) types the reduce-partials consistently on all three sites:
##
## - the backward kernel's partials parameter (was: the seed expression's width)
## - the generated reduce kernel's partials parameter
## - the runtime scratch allocation (the scalar's Nim type, by construction)
##
## Metal can never hit this, MSL refuses fp64 at print. The exercising backend
## is Vulkan GLSL (double prints there) and MoltenVK pipeline creation refuses
## fp64, so the assertions are compile-time on the printed sources.
##
## Run:
##   nim c -r --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/autodiff/t_ad_mixed_widths.nim

import std/strutils

import workspace/crucible

proc scale32(dst, a: ptr UncheckedArray[float32], w: float64, n: int32) {.global, workgroup: (1, 1).} =
  for i in 0 ..< n:
    dst[i] = float32(w) * a[i]

const adPair = bkGlsl.bwd_diff(scale32)

static:
  doAssert adPair.scalarWorkspaces.len == 1,
    "one scalar cotangent expected"
  doAssert adPair.scalarWorkspaces[0].gradName == "d_w",
    "the scalar cotangent reduces into d_w"
  doAssert "Ref_double d_w_partials" in adPair.bwdSource,
    "the backward kernel's partials are typed by the scalar's float64 width"
  doAssert "Ref_double d_w_partials" in adPair.reduceSources[0].source,
    "the reduce kernel's partials are typed by the scalar's float64 width"
  doAssert "d_w_partials.x[i] = (d_w_partials.x[i] + double(" in adPair.bwdSource,
    "the cotangent converts to the scalar's width before the accumulation"

echo "[OK] mixed-width partials typed by the scalar's width on all sites"

# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/webgpu/types/t_webgpu_reserved_rename.nim

import std/strutils
import workspace/crucible

# ── Types ──────────────────────────────────────────────────────────
type
  KwPair = object
    layout: int32
    count: int32

# ── Kernel: `layout` as param, field and access site ───────────────
const kernel = webgpu:
  proc kwKernel(output: ptr UncheckedArray[int32], layout: int32) {.global.} =
    var pair: KwPair
    pair.layout = layout
    pair.count = layout
    output[0] = pair.layout + pair.count

doAssert "layout_nim" in kernel,
  "reserved-word identifiers must rename, got:\n" & kernel
doAssert "  layout_nim: i32," in kernel,
  "struct field decl must rename, got:\n" & kernel
doAssert not kernel.contains("layout: i32"),
  "a bare reserved word must not survive as an identifier, got:\n" & kernel
doAssert "count: i32" in kernel,
  "non-colliding field must stay untouched, got:\n" & kernel

echo "  OK - t_webgpu_reserved_rename"

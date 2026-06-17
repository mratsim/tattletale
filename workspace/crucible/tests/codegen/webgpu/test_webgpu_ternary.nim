## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## WebGPU: ternary expression lowering via select().
## WGSL has no ternary ?: operator, but supports select(f, t, cond).

import std/strformat
import std/strutils
import workspace/crucible/src/codegen/wgpu

const code = webgpu:
  proc ternaryKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = if true: 1'u32 else: 0'u32

# Should not crash — check for select() lowering
doAssert code.contains("select"), &"Expected select() in:\n{code}"
echo code
echo "  OK — WGSL ternary lowering via select()"

## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## Vulkan: non-pointer (by-value) kernel parameters as push constants.
##
## In GLSL compute shaders, kernel entry points cannot have parameters.
## Pointer params → SSBO (already works).
## By-value scalar params → should be emitted as push constants.
## Currently they are silently dropped.

import std/strformat
import std/strformat
import std/strutils
import workspace/crucible
const code = vulkan:
  proc kernelWithVal(val: uint32;
                     output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = val

# The kernel should only have SSBO + push constants, no function params
doAssert code.contains("push_constant"), &"Expected push_constant block in:\n{code}"

# The kernel should have no function parameters
doAssert code.contains("void kernelWithVal()"), &"Expected void kernelWithVal() in:\n{code}"

# The by-value param `val` should appear in push constants, not function signature
doAssert code.contains("val"), &"Expected `val` in push constants or SSBO in:\n{code}"
echo code
echo "  OK — non-pointer kernel param as push constant"

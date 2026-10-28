# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Test-tree macros over the CIRv2 frontend:
## - lower a kernel body
## - surface the Leng render (lowerTrace, cirv2Metal) or the refusal message
##   (lowerErr) as a compile-time string
## - no runtime component

import
  workspace/crucible/src/codegen/ir/cirv2_pretty,
  workspace/crucible/src/codegen/frontend/nim_to_crucible,
  std/macros

macro lowerTraceImpl(body: typed): string =
  ## Lowers `body`, verifies it, and returns the one-node-per-line Leng render
  ## as a compile-time string. A refusal or a verifier violation surfaces
  ## as a compile-time error.
  let (m, err) = tryLowerCirv2(body)
  if err.len > 0:
    error(err)
  result = newLit(renderModule(m))

macro lowerTrace*(body: untyped): string =
  ## Lowers the kernel body and expands to its Leng render string:
  ## - returns the render as a compile-time string
  ## - a refusal or verifier violation is a compile-time error
  ##
  ## Example:
  ##   const trace = lowerTrace:
  ##     proc k(o: ptr UncheckedArray[uint32]) {.global.} =
  ##       o[0] = 1'u32
  result = newCall(bindSym"lowerTraceImpl", body)

macro lowerErrImpl(body: typed): string =
  ## Returns the frontend's refusal message for `body`:
  ## - "" when the body lowers and verifies clean
  ## - otherwise the first refusal message, the zero-tolerance channel
  ##
  ## Example:
  ##   const err = lowerErrImpl:
  ##     proc k(o: ptr UncheckedArray[uint32]) {.global.} =
  ##       case o[0]
  ##       of 1'u32: o[0] = 0'u32
  ##       else: discard
  ##   doAssert "case statement" in err
  result = newLit(tryLowerCirv2(body).err)

macro lowerErr*(body: untyped): string =
  ## Lowers the kernel body and expands to its refusal message:
  ## - returns "" when the body lowers and verifies clean
  ## - otherwise the first refusal message
  ##
  ## Example:
  ##   const err = lowerErr:
  ##     proc k(o: ptr UncheckedArray[uint32]) {.global.} =
  ##       case o[0]
  ##       of 1'u32: o[0] = 0'u32
  ##       else: discard
  ##   doAssert "case statement" in err
  result = newCall(bindSym"lowerErrImpl", body)

macro cirv2Metal*(body: untyped): string =
  ## Paired with the public `metal:` entry: lowers an untyped kernel body
  ## and expands to the Leng render. Returns the render as a compile-time string.
  ##
  ## Example:
  ##   const trace = cirv2Metal:
  ##     proc k(o: ptr UncheckedArray[uint32]) {.global.} =
  ##       o[0] = 1'u32
  result = newCall(bindSym"lowerTraceImpl", body)

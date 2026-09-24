# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option, this file may not be copied, modified, or distributed except according to those terms.

# ─── Element-dtype surface, mirrored from transformers, shared by the ceramic suites ───

## Element-dtype surface of the ceramic kernel suites, mirrored from the transformers tier:
##
## - `ScalarKind` with `narrowTo`/`widenTo` mirrors the libtorch `ScalarKind`
##   dtype enum and the transformers layers' `.to(dtype)` conversions
## - the ulp tolerance block (`UlpDatatype`, `binade`, `binadeStep`, `ulpStepAt`, `ulpDatatypeName`)
##   mirrors the transformers harness's allowance vocabulary (workspace/transformers/tests/harness/harness.nim)
##
## - the scalar bit surgery itself (`f32ToBf16`, `fp16ToFp32`, ...) and the seeded
##   rng live in the properties support module, this surface re-exports that whole
##   module so a suite imports one dtype surface
## - suites' band constants stay suite-local, only this surface is shared

import std/math
import ../properties/properties
export properties
proc readRecord*[T](src: ptr UncheckedArray[T]; count: int): seq[T] =
  ## Returns the host-side record of `count` elements read out of one
  ## kernel-written buffer, the bars' and bit-identity checks' raw material.
  ##
  ## Contract:
  ## - `src` stays valid over the whole copy, the seq owns its own storage
  result = newSeq[T](count)
  for i in 0 ..< count:
    result[i] = src[i]

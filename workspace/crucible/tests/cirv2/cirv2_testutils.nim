# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Cirv2 test utilities: compile-time verifier rejection assertions.
##
## Tests run inside `static:` blocks.
## `verifyMessage` walks a proc purely and returns the rejection message, so a negative test asserts on that string.
## A failing `doAssert` surfaces as a compile-time error naming the check.
## A clean compile of the test file is the pass signal.

import std/strutils
import
  workspace/crucible/src/codegen/ir/cirv2_lang,
  workspace/crucible/src/codegen/ir/cirv2_verify

proc expectVerifyError*(fn: CirProc, substring: string) =
  ## Returns: nothing, rejects through `doAssert`.
  ##
  ## Contract:
  ## - asserts `verifyMessage(fn)` names `substring` in the rejection message
  ## - a clean verification is a failure, the caller expects a rejection
  ##
  ## Example:
  ##   badProc().expectVerifyError("ite at f/body[0]: condition is not bool")
  let msg = verifyMessage(fn)
  doAssert msg.len > 0,
    "expected a verifier rejection containing \"" & substring & "\""
  doAssert substring in msg,
    "verifier message must contain \"" & substring & "\" but was: " & msg

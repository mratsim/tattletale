# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Instrumentation: the raise helpers shared by user-input validation.
##
## `checkValue` guards caller-facing input: a failed check raises catchable
## `ValueError` naming the failed check, so a bad config, checkpoint or request
## surfaces as a recoverable error. Internal invariants stay on `doAssert`,
## a defect signal rather than a recoverable path.

template checkValue*(cond: bool; msg: string) =
  ## Guard for user-input validation: raises catchable `ValueError` carrying
  ## `msg` when `cond` is false. Internal invariants stay on `doAssert`.
  if not cond:
    raise newException(ValueError, msg)

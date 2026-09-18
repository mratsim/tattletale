# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# chattyninja v4 POC. Template-level failure surface.

type
  TemplateError* = ref object of CatchableError
    ## A template-level failure. Causes are a raise_exception call, an undefined operation,
    ## a breached recursion cap. The class name matches the `expected_error.exception` field
    ## recorded on the corpus `err_*` rows.

  NotImplementedError* = ref object of CatchableError
    ## A construct that is declared and dispatched but deliberately left out of this stage, so a gap
    ## surfaces as a gap and never as a wrong answer.

proc err*(msg: string): TemplateError =
  ## Returns an unraised template error.
  new(result)
  result.msg = msg

proc newImplementError*(msg: string): NotImplementedError =
  ## Returns an unraised gap error.
  new(result)
  result.msg = msg

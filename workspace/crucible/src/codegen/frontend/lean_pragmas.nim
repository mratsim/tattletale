# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## lean_pragmas: verification pragmas on kernel procs and their payload language
##
## Lean properties cannot be written in Nim, so each pragma payload is a string holding
## a Lean property over the kernel's parameter names plus the binders `tid` and `n`.
## The payload is parsed and rendered by `backends/lean_lang.nim`, whose
## `specPragmaNames` constant is the pragma vocabulary the macro capture below filters.
##
## - `{.precondition: "<payload>".}` - assumed at launch, never proved
## - `{.invariant: "<payload>".}` - holds whenever the guarded branch is entered, implied by the guard `tid < n`
## - `{.postcondition: "<payload>".}` - holds for the written buffer after the thread ran
##
## - `prop := cmp ('∧' cmp)*`
## - `cmp  := sum ('=' | '≠' | '<' | '≤') sum`
## - `sum  := term ('+' term)*`
## - `term := ident | ident 'tid' | uint literal`
##
## Example: `{.postcondition: "output tid = a tid + b tid".}` on the add kernel.
## A well-formed payload stating a false property fails its Lean proof attempt and is downgraded to a sorry-backed scaffold.
## Anything the grammar cannot parse is refused loudly.

template precondition*(expr: static string) {.pragma.}
  ## Payload: a Lean property assumed true at kernel launch, like `"tid < n"`.

template invariant*(expr: static string) {.pragma.}
  ## Payload: a Lean property implied by the guard of the guarded branch, `tid < n` in the tracer model, like `"tid < n"`.

template postcondition*(expr: static string) {.pragma.}
  ## Payload: a Lean property over the kernel parameters, `tid` and `n`,
  ## stating the written buffer's state after the thread ran, like `"output tid = a tid + b tid"`.

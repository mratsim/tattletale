## Entry-point machinery of the shader targets: the entry-input row grammar
## and the module's one-kernel selection.
##
## Shader entry model:
##
## - a shader module compiles one entry point per engine invocation
## - the entry takes only the coordinate-builtin inputs its body reads
## - kernel parameters bind as resources: WGSL `@group`/`@binding` slots and HLSL register or push-constant blocks
##
## The module assembly runs through the shared `printModule` skeleton in `print_common`:
##
## | engine item       | the printer supplies                                    |
## | ----------------- | ------------------------------------------------------- |
## | `EntryInputRow`   | the entry-input text of one coordinate builtin kind     |
## | `emitEntryInputs` | the entry head's input list over the body-walk set      |
## | `getEntryKernel`  | the module's one kernel, refusals recorded on multiples |
import std/strutils

import ../ir/cirv2_lang
import ./print_common
import ./print_compute

type EntryInputRow* = array[CoordBuiltinKind, string]
  ## Entry-input text of one coordinate builtin kind, in the form the target's entry signature declares:
  ## a WGSL `@builtin(...)` parameter, an HLSL semantic-carrying input.
  ##
  ## "" carries no entry input: the kind the target reads another way.
  ## Cases: the body's workgroup literal, a lowering-pass materialized parameter, no read at all.

template emitEntryInputs*(kinds: untyped, row: EntryInputRow): string =
  ## Returns the entry-input texts of one shader entry point: one row text per coordinate builtin kind the entry takes.
  ##
  ## `kinds` is the set the entry's body walk collected (`ctx.builtins` on WGSL, `ctx.coordReads` on HLSL).
  ## The entry takes exactly the inputs its body reads, in the set's own iteration order: first use on WGSL, enum order on HLSL.
  ##
  ##   "fn k(" & emitEntryInputs(ctx.builtins, wgslEntryInputs) & ") {"
  var texts: seq[string]
  for k in kinds:
    if row[k].len > 0:
      texts.add row[k]
  texts.join(", ")

proc getEntryKernel*(ctx: var auto, m: CirModule): CirProc =
  ## Returns the module's one kernel: a shader engine compiles one entry point per module invocation.
  ## Returns nil with a refusal recorded when the module has none or several kernels.
  ##
  ##   if ctx.failing():
  ##     return ("", ctx.err)
  let kernels = moduleKernels(m)
  if kernels.len != 1:
    ctx.fail("exactly one kernel per module, found " & $kernels.len)
    return
  result = kernels[0]

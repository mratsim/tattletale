## Shader-tier emission shared by the WGSL and HLSL printers: the entry-point machinery and the module assembly both targets emit.
##
## The two shader targets split from the compute class on the entry and runtime model:
##
## - a shader module compiles one entry point per engine invocation
## - the entry takes only the coordinate-builtin inputs its body reads
## - kernel parameters bind as resources: WGSL `@group`/`@binding` slots, HLSL register or push-constant blocks
##
## Section bodies stay in the printers: WGSL grammar and bindings in `wgsl_lang`, HLSL register and semantics glue in `hlsl_lang`.
## Each printer owns a `PrintCtx` (`ctx: var auto`) and supplies the target's vocabulary as `mixin` procs beside it, per `print_common`:
##
## | mixin proc                   | the printer supplies                          |
## | ---------------------------- | --------------------------------------------- |
## | `prepShaderProc(fn)`         | the per-proc walk state of the target         |
## | `emitShaderSig(fn)`          | the signature head, entry attributes included |
## | `emitShaderStructs(m)`       | the referenced struct declarations            |
## | `emitShaderBindings(kernel)` | the kernel's resource-binding section         |
## | `emitShaderFns(m)`           | the device-function section                   |
## | `emitComputeGlobal(s)`       | one hoisted program-scope declaration         |
import std/strutils

import ../ir/cirv2_lang
import ./print_common
import ./print_compute

type EntryInputRow* = array[CoordBuiltinKind, string]
  ## Entry-input text of one coordinate builtin kind, in the form the target's entry signature declares:
  ## a WGSL `@builtin(...)` parameter, an HLSL semantic-carrying input.
  ##
  ## "" names no entry input: the kind the target reads another way.
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

proc selectShaderKernel*(ctx: var auto, m: CirModule): CirProc =
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

proc printShaderProc*(ctx: var auto, fn: CirProc): string =
  ## Returns the text of one device function or the entry kernel: the target's per-proc state, then the body, then the head.
  ## Returns "" once the walk recorded a refusal.
  ##
  ## The body prints before the signature assembles: the entry head names the coordinate-builtin inputs the body walk collected.
  ## The head's inputs are WGSL's `@builtin(...)` parameters or HLSL's semantics, both collected during the walk.
  ##
  ## Contract:
  ## - a device function's head is its bare signature line, and the caller opens the body on that line
  ## - an entry kernel's head is the full entry head, target attribute lines included, and its open brace placement is the target's
  ##   same line on WGSL, own line under the `[numthreads]` head on HLSL
  ##
  ##   let txt = printShaderProc(ctx, fn)  # "fn square(x: i32) -> i32 {" and the body and "}"
  mixin prepShaderProc, emitShaderSig
  ctx.fnName = fn.name
  ctx.indent = 0
  ctx.prepShaderProc(fn)
  let bodyTxt = ctx.printStmts(fn.body)
  if ctx.failing():
    return ""
  let head = ctx.emitShaderSig(fn)
  if ctx.failing():
    return ""
  if fn.isKernel:
    return head & bodyTxt & '\n' & '}'
  return head & " {\n" & bodyTxt & '\n' & '}'

proc printShaderModule*(ctx: var auto, m: CirModule, kernel: CirProc,
    prelude: string, hoistedSpaces: set[AddressSpace]): tuple[src: string, err: string] =
  ## Returns the full shader module assembly: the section order both targets emit.
  ## Returns "" as the error when clean, else the first refusal.
  ##
  ## Sections:
  ## - `prelude`: the target's leading lines, WGSL's `enable f16` and the `NIM_BOOL` alias, "" on HLSL, printed first
  ## - the referenced struct declarations, then the kernel's resource-binding section
  ## - the hoisted program-scope declarations of `hoistedSpaces`, through `printComputeGlobals` and `emitComputeGlobal`
  ## - the device functions, then the entry kernel last: its body walk fills the input set its signature reads
  ##
  ## Every section checks the walk's first refusal, so a refusal anywhere records once and returns.
  mixin emitShaderStructs, emitShaderBindings, emitShaderFns
  ctx.fnName = kernel.name
  result.src = prelude
  result.src &= ctx.emitShaderStructs(m)
  if ctx.failing():
    return ("", ctx.err)
  result.src &= ctx.emitShaderBindings(kernel)
  if ctx.failing():
    return ("", ctx.err)
  result.src &= printComputeGlobals(ctx, m, hoistedSpaces)
  if ctx.failing():
    return ("", ctx.err)
  result.src &= ctx.emitShaderFns(m)
  if ctx.failing():
    return ("", ctx.err)
  result.src &= ctx.printShaderProc(kernel) & "\n"
  if ctx.failing():
    return ("", ctx.err)

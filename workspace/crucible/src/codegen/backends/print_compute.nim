# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Compute-tier emission shared by the CUDA C++, OpenCL C, GLSL and MSL printers:
## proc signatures and the module's struct declarations.
##
## | proc                | contract                                                     |
## | ------------------- | ------------------------------------------------------------ |
## | printComputeSig     | one signature line: prefix, return, name, parameters         |
## | printComputeProc    | one proc or kernel: signature, body, closing brace           |
## | printComputeStructs | the module's struct declarations, C-family and typedef forms |
##
## Each printer owns a `PrintCtx` (`ctx: var auto`) and supplies the target
## vocabulary as `mixin` procs beside it, matching the `print_common` pattern.
##
## - `emitComputeParams(fn)`: printed parameter texts, in declaration order
## - `emitComputePrefix(fn)`: linkage prefix (`__global__`, `__kernel`, ...)
## - `emitComputeRet(t)`: return type name or declaration text
## - `emitComputeStructField(struct, field)`: one struct member's declaration text
##
## The body prints before the signature assembles: the MSL kernel signature
## appends the coordinate-builtin parameters that the body walk collects.

import ../ir/cirv2_lang
import ./print_common

proc printComputeSig*(ctx: var auto, fn: CirProc): string =
  ## Signature text of one proc or kernel, without a body:
  ## `<prefix><ret> <name>(<params>)`.
  ##
  ## A kernel's name is its exportc override when one is set, else the proc name.
  ## The host runtime looks a kernel up by its emitted name.
  ## Returns "" once the walk recorded a refusal.
  ##
  ##   printComputeSig(ctx, fn)  # "static inline float f(float x)" on OpenCL C
  mixin emitComputeParams, emitComputePrefix, emitComputeRet
  ctx.fnName = fn.name
  let name = if fn.isKernel and fn.exportName.len > 0: fn.exportName else: fn.name
  let params = ctx.emitComputeParams(fn)
  if ctx.failing():
    return ""
  result = ctx.emitComputePrefix(fn) & ctx.emitComputeRet(fn.ret) & ' ' & name &
    '(' & params.join(", ") & ")"

proc printComputeProc*(ctx: var auto, fn: CirProc): string =
  ## Text of one proc or kernel: signature, body, closing brace.
  ##
  ## The body prints before the signature assembles: the MSL kernel signature
  ## appends the coordinate-builtin parameters that the body walk collects.
  ## Returns "" once the walk recorded a refusal.
  ##
  ##   printComputeProc(ctx, fn)  # "extern \"C\" __global__ void k(float* o) { ... }"
  ctx.fnName = fn.name
  ctx.indent = 0
  let bodyTxt = ctx.printStmts(fn.body)
  if ctx.failing():
    return ""
  let sig = printComputeSig(ctx, fn)
  if ctx.failing():
    return ""
  return sig & " {\n" & bodyTxt & '\n' & '}'

proc printComputeStructs*(ctx: var auto, m: CirModule, typedef: bool, padding: string): string =
  ## Declaration text of the module's struct types, one per `CirModule.types`
  ## entry in encounter order.
  ##
  ## Expected input:
  ## - `typedef`: OpenCL C names a struct through `typedef struct { ... } Name`,
  ##   the C-family form is `struct Name { ... }`
  ## - `padding`: member text of the zero-field struct declaration, without
  ##   indentation and without the member terminator. Every target compiler
  ##   requires at least one member
  ##
  ## Output: one declaration per struct, members indented two spaces,
  ## declarations separated by a blank line. A non-struct type in `m.types`
  ## is a refusal.
  ##
  ##   printComputeStructs(ctx, m, false, "char _")  # "struct Pixel { ... }"
  mixin emitComputeStructField
  for t in m.types:
    if t.kind != ctStruct:
      ctx.fail("module type " & t.name & " is not a struct")
      break
    if typedef:
      result &= "typedef struct {\n"
    else:
      result &= "struct " & t.name & " {\n"
    if t.fields.len == 0:
      result &= "  " & padding & ";\n"
    for f in t.fields:
      result &= "  " & ctx.emitComputeStructField(t, f) & ";\n"
    if typedef:
      result &= "} " & t.name & ";\n\n"
    else:
      result &= "};\n\n"

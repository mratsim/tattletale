# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Shared helpers for the Lean backend test binaries
## (`t_lean_emit`, `t_lean_loop`, `t_lean_execute`, `t_lean_pragmas`)

import std/[macros, os, osproc, strutils]
import workspace/crucible/src/builtins/builtins
import workspace/crucible/src/builtins/builtins_compilermagic
import workspace/crucible/src/codegen/frontend/nim_to_crucible
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/passes_normalizations
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/backends/lang_lean

export cirv2_lang, lang_lean, builtins

const leanProjectDir* = currentSourcePath().parentDir()

type LeanIr* = tuple[text: string, refusal: string]
  ## Compile result of `leanIr`: the emitted module text, or the caught
  ## `lean_lang: ...` refusal message with empty text

macro leanIr*(body: typed): LeanIr =
  ## Lowers the body through the Lean pass pipeline and emits its module text
  ## at compile time. A shape refusal surfaces as the `refusal` field instead
  ## of a compile error, so the test battery can assert on refusal messages.
  ##
  ## Returns `(text, refusal)`: exactly one field is non-empty
  crucibleCompileTarget = ctLean
  var reg = PassRegistry.new()
  registerNormalizationPasses(reg)
  let (m, lerr) = lowerCirv2(body)
  if lerr.len > 0:
    error(lerr)
  let perr = runPasses(m, reg)
  if perr.len > 0:
    error(perr)
  var kernels: seq[CirProc]
  for p in m.procs:
    if p.isKernel:
      kernels.add p
  try:
    result = newLit((text: codegen(m), refusal: ""))
  except ValueError as e:
    result = newLit((text: "", refusal: e.msg))

proc expectRefusal*(prefix: string, body: proc) =
  ## Runs `body`, expects the `lean_lang: ...` ValueError whose message
  ## starts with `prefix`.
  ##
  ## Contract:
  ## - `prefix` is the refusal's documented reason, so a refusal raised for
  ##   the wrong reason fails with expected and got in the message
  ## - no ValueError at all fails with the expected prefix in the message
  try:
    body()
  except ValueError as e:
    doAssert e.msg.startsWith(prefix),
      "wrong refusal reason, expected '" & prefix & "', got: " & e.msg
    return
  doAssert false, "expected a refusal: " & prefix

proc expectLeanRefusal*(prefix: string, ir: LeanIr) =
  ## Expects the compile-time `lean_lang: ...` refusal captured in `ir`,
  ## asserting the message starts with `prefix`
  doAssert ir.refusal.startsWith(prefix),
    "wrong refusal reason, expected '" & prefix & "', got: " & ir.refusal
  doAssert ir.text.len == 0, "a refused kernel emits no module text"

proc elaborate*(relPath: string): tuple[output: string, exitCode: int] =
  ## Elaborates one emitted Lean module through the lake environment,
  ## warnings escalated to errors so a sorry fails
  execCmdEx("lake env lean -DwarningAsError=true " & relPath,
    workingDir = leanProjectDir)

proc execute*(relPath: string): tuple[output: string, exitCode: int] =
  ## Runs one emitted Lean module's `main` through the lake environment
  execCmdEx("lake env lean --run " & relPath,
    workingDir = leanProjectDir)

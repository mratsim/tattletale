# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Shared helpers for the Lean frontend test binaries (`t_lean_emit`, `t_lean_pragmas`, `t_lean_loop`, `t_lean_execute`)

import std/[os, osproc]

const leanProjectDir* = currentSourcePath().parentDir()

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

proc elaborate*(relPath: string): tuple[output: string, exitCode: int] =
  ## Elaborates one emitted Lean module through the lake environment,
  ## warnings escalated to errors so a sorry fails
  execCmdEx("lake env lean -DwarningAsError=true " & relPath,
    workingDir = leanProjectDir)

proc execute*(relPath: string): tuple[output: string, exitCode: int] =
  ## Runs one emitted Lean module's `main` through the lake environment
  execCmdEx("lake env lean --run " & relPath,
    workingDir = leanProjectDir)

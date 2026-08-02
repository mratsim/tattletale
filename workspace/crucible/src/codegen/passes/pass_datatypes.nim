## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import std / [macros, sequtils, sets, tables]
import ../ir/gpu_types

export gpu_types

type
  PassKind* = enum
    pkValidation     ## Check-only: error/warn if invariant violated
    pkTransform      ## Mutates IR
    pkAnalysis       ## Computes metadata, no mutation

  PassPhase* = enum
    phaseEarly       ## Right after IR construction (normalization)
    phasePreprocessing  ## Preprocessing stage (after legalization, before lowering)
    phaseMain        ## Before lowering (optimizations, analysis, validation)

  GpuPass* = ref object of RootObj
    name*: string
    phase*: PassPhase
    kind*: PassKind
    description*: string
    dependsOn*: seq[string]
    run*: proc(ctx: var GpuContext): void {.nimcall.}

  PassRegistry* = ref object
    passes*: seq[GpuPass]
    donePasses*: HashSet[string]

# ─── Walk ──

proc walk*(body: var GpuAst; pre: proc(n: var GpuAst): void): void =
  ## Depth-first pre-order traversal. Calls `pre` at every node,
  ## then recurses into children via `mitems`.
  pre(body)
  for child in body.mitems:
    child.walk(pre)

# ─── Registration ──

proc register*(reg: var PassRegistry; name: string; kind: PassKind;
               phase: PassPhase; description: string;
               run: proc(ctx: var GpuContext): void {.nimcall.};
               dependsOn: seq[string] = @[]) =
  reg.passes.add GpuPass(name: name, kind: kind, phase: phase,
                         description: description,
                         dependsOn: dependsOn, run: run)

# ─── Execution ──

proc runPasses*(ctx: var GpuContext; reg: var PassRegistry) =
  ## Run all registered passes in order, verifying dependencies.
  ## TODO: pass the AST explicitly (coupled to ctx.allFnTab)
  for p in reg.passes:
    for dep in p.dependsOn:
      if dep notin reg.donePasses:
        error "\"" & p.name & "\" requires \"" & dep & "\" to run first"
    when defined(debugPasses):
      echo "[pass] ", p.name, " (", p.phase, ")"
    p.run(ctx)
    reg.donePasses.incl p.name

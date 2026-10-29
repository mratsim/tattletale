# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Pass machinery over CIRv2: the pass record, the registry, and the runner.
##
## A pass run takes the module in place and returns its refusal:
## "" when clean, else the first refusal.
## The CIRv2 rejection channel is pure strings, no exceptions.

import std/sets
import ../ir/cirv2_lang

export cirv2_lang

type
  PassKind* = enum
    pkValidation     ## Check-only: refuses on invariant violation
    pkTransform      ## Changes the IR
    pkAnalysis       ## Computes metadata, changes nothing

  PassPhase* = enum
    phaseEarly         ## Right after IR construction (legalization, normalization)
    phasePreprocessing ## Between normalization and lowering
    phaseMain          ## Before printing (optimization, lowering, validation)

  CirPass* = ref object
    name*: string
    kind*: PassKind
    phase*: PassPhase
    description*: string
    dependsOn*: seq[string]
    run*: proc(m: CirModule): string {.nimcall.}
      ## One pass over the module, "" when clean, else the first refusal

  PassRegistry* = ref object
    passes*: seq[CirPass]
    donePasses*: HashSet[string]

func register*(reg: var PassRegistry, name: string, kind: PassKind,
               phase: PassPhase, description: string,
               run: proc(m: CirModule): string {.nimcall.},
               dependsOn: seq[string] = @[]) =
  reg.passes.add CirPass(name: name, kind: kind, phase: phase,
                         description: description,
                         dependsOn: dependsOn, run: run)

proc runPasses*(m: CirModule, reg: var PassRegistry): string =
  ## Runs the registered passes in registration order.
  ## Every dependency must already have run.
  ## Returns "" when the whole pipeline is clean, else the first refusal.
  for p in reg.passes:
    for dep in p.dependsOn:
      if dep notin reg.donePasses:
        return "pass \"" & p.name & "\" requires \"" & dep & "\" to run first"
    let err = p.run(m)
    if err.len > 0:
      return "pass " & p.name & ": " & err
    reg.donePasses.incl p.name

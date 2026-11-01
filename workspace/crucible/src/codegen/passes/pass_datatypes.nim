# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Pass machinery over CIRv2: pass record, registry, and runner.
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

  CirPass* = ref object
    name*: string
    kind*: PassKind
    description*: string
    run*: proc(m: CirModule): string {.nimcall.}
      ## One pass over the module, "" when clean, else the first refusal

  PassRegistry* = ref object
    passes*: seq[CirPass]
    donePasses*: HashSet[string]

func register*(reg: var PassRegistry, name: string, kind: PassKind, description: string, run: proc(m: CirModule): string {.nimcall.}) =
  reg.passes.add CirPass(name: name, kind: kind, description: description, run: run)

proc runPasses*(m: CirModule, reg: var PassRegistry): string =
  ## Runs the registered passes in registration order.
  ##
  ## Contract:
  ## - a transform registered after an analysis refuses, as the analysis records
  ##   on the module (features, call graph, ref blocks) would go stale
  ## - the printers read the analysis records, they never rescan
  ## - returns "" when the whole pipeline is clean, else the first refusal
  var sawAnalysis = false
  for p in reg.passes:
    if p.kind == pkAnalysis:
      sawAnalysis = true
    elif p.kind == pkTransform and sawAnalysis:
      return "pass \"" & p.name & "\" transforms after an analysis pass," &
        " the analysis records would go stale"
    let err = p.run(m)
    if err.len > 0:
      return "pass " & p.name & ": " & err
    reg.donePasses.incl p.name

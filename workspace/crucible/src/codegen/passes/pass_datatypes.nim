# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/sets
import ../ir/cirv2_lang
import ../ir/cirv2_verify

export cirv2_lang

# ############################################################
#
#            Pass datatypes
#
# ############################################################

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
  ## - on modules with verifyIR set, verifyModule runs once, after the last
  ##   transform pass runs clean: intermediate transforms legitimately emit
  ##   callees that later transforms resolve (span-type callees until lowerSpans rewrites them),
  ##   and consumers read only the final boundary
  ## - returns "" when the whole pipeline is clean, else the first refusal
  var lastTransform = -1
  for i, p in reg.passes:
    if p.kind == pkTransform:
      lastTransform = i
  var sawAnalysis = false
  for i, p in reg.passes:
    if p.kind == pkAnalysis:
      sawAnalysis = true
    elif p.kind == pkTransform and sawAnalysis:
      return "pass \"" & p.name & "\" transforms after an analysis pass," &
        " the analysis records would go stale"
    let err = p.run(m)
    if err.len > 0:
      return "pass " & p.name & ": " & err
    reg.donePasses.incl p.name
    if i == lastTransform and m.verifyIR:
      let verr = verifyModule(m)
      if verr.len > 0:
        return "verify: " & verr

# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## Normalization: explicit unroll marking over CIRv2.
##
## The frontend lowers a `{.unroll.}` statement pragma as the raw-text marker statement `emitS(@["{.unroll.}"], @[])`.
## This pass consumes every marker:
##
## - the loop whose body directly states one gets `unroll = true`
## - a marker outside a loop body drops, no printer ever sees either
## - a marker nested inside an inner loop's body marks that inner loop, never the enclosing one:
##   a loop's body is scanned for its own marker before the walk descends into the remaining statements
## - every target registers this pass in the normalization stage, before the feature scan reads the unroll flags

import ../ir/cirv2_lang
import ./pass_datatypes

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerMarkExplicitUnrolls*(reg: var PassRegistry) =
  ## Registers the explicit-unroll marking pass in the normalization stage.
  reg.register("markExplicitUnrolls", pkTransform, phaseEarly,
    "Sets the unroll flag on the loops whose body states the `{.unroll.}` marker statement and drops every marker",
    run = markExplicitUnrolls)

# ═══════════════════════════════════════════════════════════════════════
# Unroll marking
# ═══════════════════════════════════════════════════════════════════════

const unrollMarker = "{.unroll.}"

func isUnrollMarker(s: CirStmt): bool =
  s.kind == sEmit and s.emitOperands.len == 0 and s.emitParts == @[unrollMarker]

proc markExplicitUnrolls(ss: var seq[CirStmt]) =
  var kept: seq[CirStmt]
  for s in mitems(ss):
    case s.kind
    of sLoop:
      var body: seq[CirStmt]
      var marked = false
      for b in s.loopBody:
        if b.isUnrollMarker:
          marked = true
        else:
          body.add b
      if marked:
        s.unroll = true
      markExplicitUnrolls(body)
      s.loopBody = body
      kept.add s
    of sIte:
      markExplicitUnrolls(s.thenBody)
      markExplicitUnrolls(s.elseBody)
      kept.add s
    else:
      if not s.isUnrollMarker:
        kept.add s
  ss = kept

proc markExplicitUnrolls(m: CirModule): string =
  for fn in m.procs:
    markExplicitUnrolls(fn.body)
  markExplicitUnrolls(m.globals)
  return ""

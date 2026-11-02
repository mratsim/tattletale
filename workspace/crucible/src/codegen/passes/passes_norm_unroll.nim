# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## Normalization: explicit unroll marking over CIRv2.
##
## The frontend lowers a `{.unroll.}` statement pragma into the raw-text marker statement `emitS(@["{.unroll.}"], @[])`.
## This pass consumes every marker.
##
## - the loop whose body directly states one gets `unroll = true`
## - a marker outside a loop body drops
## - a marker nested inside an inner loop's body marks that inner loop, never the enclosing one (the scan checks a body before descending)
import ../ir/cirv2_lang
import ./pass_datatypes

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerMarkExplicitUnrolls*(reg: var PassRegistry) =
  reg.register("markExplicitUnrolls", pkTransform,
    "Sets the unroll flag on the loops whose body states the `{.unroll.}`" &
    " marker statement and drops every marker",
    run = markExplicitUnrollsPass)

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
    of sWhile:
      var body: seq[CirStmt]
      var marked = false
      for b in s.whileBody:
        if b.isUnrollMarker:
          marked = true
        else:
          body.add b
      if marked:
        s.whileUnroll = true
      markExplicitUnrolls(body)
      s.whileBody = body
      kept.add s
    of sIte:
      markExplicitUnrolls(s.thenBody)
      markExplicitUnrolls(s.elseBody)
      kept.add s
    else:
      if not s.isUnrollMarker:
        kept.add s
  ss = kept

proc markExplicitUnrollsPass(m: CirModule): string =
  for fn in m.procs:
    markExplicitUnrolls(fn.body)
  return ""

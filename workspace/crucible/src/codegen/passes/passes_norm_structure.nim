# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## Normalization: control-flow structuring, the frontend's lab/jmp condition
## code compiled back into structured statements.
##
## The frontend lowers `and`/`or` conditions through the two-target compiler
## (Appel Cx) over forward-only scoped `lab`/`jmp` and if/elif chains over one
## shared merge label.
##
##   flat lab/jmp list ──structure──▶ nested ite/loop + break/continue jmps
##
## Contract:
##
## - input: a verified CIRv2 module, the verifier guarantees forward-only scoped `jmp` and targeted `lab`
## - output: an equivalent module whose only remaining `lab`/`jmp` are a loop's continue and exit labels, either jmp names continue/break
## - any other `lab`/`jmp` refuses loudly, naming the proc and the label
##
## Shapes consumed, all frontend-generated:
##
## - a Cx test `ite(c, [jmp A], [jmp B])`: one block continues the condition chain, the other reaches the shared merge
## - an elif chain `ite(c, body, [])` ending in `jmp(merge)`: the rest of the list is the else, the merge label closes it
## - a plain `ite` with real bodies: if/else, or a loop guard with one empty side jumping to the loop exit: the break form
## - a block's trailing `jmp` to a label outside the current list: the block's exit, dropped,
##   the caller's continuation takes over
##
## Any other `jmp` to a label outside the current list refuses loudly.

import ../ir/cirv2_lang
import ../frontend/nim_to_crucible
import ./pass_datatypes
import std/[macros, tables, strutils]

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerStructuring*(reg: var PassRegistry) =
  ## Registers the control-flow structuring pass, the normalization stage.
  reg.register("structureControlFlow", pkTransform, phasePreprocessing,
    "Compiles the lab/jmp condition code into structured statements",
    run = structurePass)

proc structurePass(m: CirModule): string =
  let (structured, err) = structureModule(m)
  if err.len == 0:
    m[] = structured[]
  return err


type
  StructCtx = object
    fnName: string
    ## First refusal of the walk, "" while clean.
    err: string
    labPos: Table[string, int]
    ## Current loop's exit and continue labels, both "" when no loop is open.
    exitL, contL: string

proc failing(ctx: StructCtx): bool =
  ctx.err.len > 0

proc fail(ctx: var StructCtx, msg: string) =
  if ctx.err.len == 0:
    ctx.err = "cirv2 structure: at " & ctx.fnName & ": " & msg

# ═══════════════════════════════════════════════════════════════════════
# Block reconstruction
# ═══════════════════════════════════════════════════════════════════════

proc structurizeList(ctx: var StructCtx, stmts: seq[CirStmt]): seq[CirStmt]
  ## Forward declaration: every statement list defines its own label table.

proc structurizeBlock(ctx: var StructCtx, stmts: seq[CirStmt], lo, hi: int): seq[CirStmt] =
  ## Structurizes the slice `stmts[lo ..< hi]`.
  ##
  ## - the slice's own labels form its label table
  ## - only the slice's trailing `jmp` may target a label outside the slice,
  ##   that jump is the block's exit, dropping it lets the caller's
  ##   continuation take over
  ## - any other jump refuses loudly
  if ctx.failing():
    return
  var sub = ctx
  var endIx = hi
  if hi > lo and stmts[hi - 1].kind == sJmp:
    var targetInside = false
    for j in lo ..< hi - 1:
      if stmts[j].kind == sLab and stmts[j].labName == stmts[hi - 1].jmpTarget:
        targetInside = true
        break
    if not targetInside:
      # block-exit jump: the target names the caller's continuation
      # hiding it from the walk, so an earlier leaving jump cannot pass for it
      endIx = hi - 1
  result = sub.structurizeList(stmts[lo ..< endIx])
  ctx.err = sub.err

proc isTrivialJmp(s: seq[CirStmt]): bool =
  ## Returns true for a branch body of exactly one `jmp`, the Cx test shape.
  s.len == 1 and s[0].kind == sJmp

proc findBlockEnd(ctx: StructCtx, stmts: seq[CirStmt], start: int): int =
  ## Returns one after the last statement of the block beginning at `start`:
  ## the first jump leaving the list (unknown target) or a merge jump.
  ## A block with no exit jump runs to the list end.
  var j = start
  while j < stmts.len:
    let t = stmts[j]
    if t.kind == sJmp and (not ctx.labPos.hasKey(t.jmpTarget) or
        ctx.labPos[t.jmpTarget] > j):
      return j + 1
    j.inc
  return stmts.len

proc structurize(ctx: var StructCtx, stmts: seq[CirStmt]): seq[CirStmt] =
  ## One pass over a flat statement list, consuming its merge labels.
  ## `ctx.labPos` must be the table of `stmts`.
  if ctx.failing():
    return
  var i = 0
  while i < stmts.len and not ctx.failing():
    let s = stmts[i]
    case s.kind
    of sLab:
      # loop exit/continue labels survive, every other lab is a merge
      # marker consumed by the branch that jumps to it
      if s.labName != ctx.exitL and s.labName != ctx.contL:
        ctx.fail("unexpected label " & s.labName & " with no structuring branch")
        continue
      result.add s
      i.inc
    of sJmp:
      if s.jmpTarget == ctx.exitL or s.jmpTarget == ctx.contL:
        result.add s  # break/continue, named at emission
      elif not ctx.labPos.hasKey(s.jmpTarget):
        ctx.fail("jump to " & s.jmpTarget & " leaves the statement list")
        continue
      else:
        ctx.fail("unexpected jump to " & s.jmpTarget)
        continue
      i.inc
    of sIte:
      let thenTgt = if isTrivialJmp(s.thenBody): s.thenBody[0].jmpTarget else: ""
      let elseTgt = if isTrivialJmp(s.elseBody): s.elseBody[0].jmpTarget else: ""
      let thenKnown = thenTgt.len > 0 and ctx.labPos.hasKey(thenTgt)
      let elseKnown = elseTgt.len > 0 and ctx.labPos.hasKey(elseTgt)

      if thenTgt.len == 0 and elseTgt.len == 0:
        var ite = s
        if s.elseBody.len == 0 and s.thenBody.len > 0 and
            s.thenBody[^1].kind == sJmp and
            s.thenBody[^1].jmpTarget != ctx.exitL and
            s.thenBody[^1].jmpTarget != ctx.contL:
          # elif chain shape: a real then side ending in the shared merge jump.
          # Everything after it is the else side.
          # Its merge label sits later in this list (its lab closes the chain),
          # or outside it for a nested chain slice closed by the enclosing list.
          let m = s.thenBody[^1].jmpTarget
          let eElse = if ctx.labPos.hasKey(m): ctx.labPos[m] else: stmts.len
          var sub = ctx
          ite.thenBody = sub.structurizeList(s.thenBody[0 ..< s.thenBody.len - 1])
          ctx.err = sub.err
          ite.elseBody = ctx.structurizeBlock(stmts, i + 1, eElse)
          result.add ite
          i = if eElse < stmts.len: eElse + 1 else: eElse
        else:
          # both sides real (or empty): plain if/else, each body restructures
          if s.thenBody.len > 0:
            var sub = ctx
            ite.thenBody = sub.structurizeList(s.thenBody)
            ctx.err = sub.err
          if s.elseBody.len > 0:
            var sub = ctx
            ite.elseBody = sub.structurizeList(s.elseBody)
            ctx.err = sub.err
          result.add ite
          i.inc
      elif thenKnown and elseKnown:
        # Cx test: two jump targets, both labels in this list. One block is
        # exactly the rest of the condition chain ending at the other branch's
        # label and the other block runs to the shared merge.
        let a = ctx.labPos[thenTgt]
        let b = ctx.labPos[elseTgt]
        if a == b:
          ctx.fail("condition jumps both targets to " & thenTgt)
          continue
        let first = min(a, b)
        let last = max(a, b)
        # chain block ends at the other branch's label, its exit jump
        # names the shared merge when one exists
        var merge = ""
        for j in first + 1 ..< last:
          let t = stmts[j]
          if t.kind == sJmp and (not ctx.labPos.hasKey(t.jmpTarget) or
              ctx.labPos[t.jmpTarget] > last):
            merge = t.jmpTarget
            break
        var eSecond: int
        if merge.len > 0 and ctx.labPos.hasKey(merge):
          # merge label lives in this list, the branch block runs to it
          eSecond = ctx.labPos[merge]
          # a trailing jump to the merge belongs to the structure, not the block
          if eSecond > last + 1 and stmts[eSecond - 1].kind == sJmp and
              stmts[eSecond - 1].jmpTarget == merge:
            eSecond.dec
        else:
          eSecond = ctx.findBlockEnd(stmts, last + 1)
        var ite = CirStmt(kind: sIte, cond: s.cond)
        let chainBody = ctx.structurizeBlock(stmts, first + 1, last)
        let branchBody = ctx.structurizeBlock(stmts, last + 1, eSecond)
        if a < b:
          # chain on the then side (`and`), the branch on the else side
          ite.thenBody = chainBody
          ite.elseBody = branchBody
        else:
          # chain on the else side (`or`), the branch on the then side
          ite.thenBody = branchBody
          ite.elseBody = chainBody
        result.add ite
        if merge.len > 0 and ctx.labPos.hasKey(merge):
          i = ctx.labPos[merge] + 1
        else:
          i = eSecond
      elif thenTgt.len > 0 and elseTgt.len > 0 and (thenKnown or elseKnown):
        # both sides jump but one label lives outside this list: the unknown side
        # is empty and the known side's block is the branch
        let p = if thenKnown: ctx.labPos[thenTgt] else: ctx.labPos[elseTgt]
        let e = ctx.findBlockEnd(stmts, p + 1)
        var ite = CirStmt(kind: sIte, cond: s.cond)
        let body = ctx.structurizeBlock(stmts, p + 1, e)
        if thenKnown:
          ite.thenBody = body
        else:
          ite.elseBody = body
        result.add ite
        i = e
      elif thenTgt.len > 0 and elseTgt.len > 0:
        ctx.fail("condition jumps to " & thenTgt & " and " & elseTgt &
          ", both labels outside the statement list")
      elif thenTgt.len > 0 or elseTgt.len > 0:
        # one side jumps, the other is empty
        let tgt = if thenTgt.len > 0: thenTgt else: elseTgt
        let isGuardBreak = ctx.exitL.len > 0 and tgt == ctx.exitL
        let isContinue = ctx.contL.len > 0 and tgt == ctx.contL
        if not (isGuardBreak or isContinue):
          ctx.fail("one-sided jump to " & tgt & " outside a loop guard")
          continue
        result.add s  # guard's break or a continue, named at emission
        i.inc
    of sLoop:
      # exit label sits right after the loop, the continue label inside
      if i + 1 >= stmts.len or stmts[i + 1].kind != sLab:
        ctx.fail("loop without an exit label after it")
        continue
      let exitL = stmts[i + 1].labName
      var contL = ""
      for t in s.loopBody:
        if t.kind == sLab and t.labName.startsWith(ContinueLabelPrefix):
          if contL.len > 0:
            ctx.fail("loop with two continue labels inside")
            break
          contL = t.labName
      var sub = ctx
      sub.exitL = exitL
      sub.contL = contL
      let body = sub.structurizeList(s.loopBody)
      ctx.err = sub.err
      result.add CirStmt(kind: sLoop, loopBody: body, unroll: s.unroll)
      result.add stmts[i + 1]  # exit label, the printer's break target
      i += 2
    else:
      result.add s
      i.inc

proc structurizeList(ctx: var StructCtx, stmts: seq[CirStmt]): seq[CirStmt] =
  ## Structurizes one statement list.
  ## The list's own labels form its label table: a jump to a label outside
  ## this list refuses. `structurizeBlock` strips a block-exit jump before the walk,
  ## the caller's continuation then takes over.
  if ctx.failing():
    return
  ctx.labPos = initTable[string, int]()
  for i, t in stmts:
    if t.kind == sLab:
      ctx.labPos[t.labName] = i
  return ctx.structurize(stmts)

proc structureProc(fn: CirProc): tuple[fn: CirProc, err: string] =
  ## Returns `fn` with its body's lab/jmp condition code compiled into
  ## structured statements. The error is "" when clean, else the first
  ## refusal naming a shape the frontend does not generate.
  var ctx = StructCtx(fnName: fn.name)
  let outFn = CirProc(name: fn.name, params: fn.params, ret: fn.ret,
    isKernel: fn.isKernel, forceInline: fn.forceInline,
    exportName: fn.exportName, workgroup: fn.workgroup)
  outFn.body = ctx.structurizeList(fn.body)
  result = (outFn, ctx.err)

proc structureModule(m: CirModule): tuple[m: CirModule, err: string] =
  ## Returns `m` with every proc body structured:
  ## - the error is "" for a clean module
  ## - otherwise the first refusal of the walk over the procs
  result = (CirModule(name: m.name, types: m.types), "")
  # program-scope declarations pass through, the printers emit them in encounter order
  result.m.globals = m.globals
  for fn in m.procs:
    let (sfn, err) = structureProc(fn)
    if err.len > 0:
      result.err = err
      return
    result.m.procs.add sfn

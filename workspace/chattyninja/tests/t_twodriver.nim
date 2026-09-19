# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Read-only artifact proof for the chattyninja engine.
##
## `Machine` is two fields and no mutable state, so one compiled artifact serves any number
## of independent renders.
##
## - two `Driver`s over the SAME `Machine`, stepped in lockstep and interleaved, agree byte
##   for byte with a sequential render
## - a second `Machine` rendered alternately with the first leaves both renders undisturbed,
##   and the artifact's bytes stay identical across rendering
## - chunking composes with `cur`, small chunks yield the same bytes as one pull
##
## `-d:ChunkSize=7` forces chunk boundaries below the render length, which the default
## 4096 never reaches on corpus renders.
##
## Run:
##   $ ./workspace/chattyninja/run_tests.sh t_twodriver

import std/[os, strutils]
import cnj_types, cnj_values, cnj_parse, cnj_engine
import workspace/data_structures/src/small_seqs
import rows

type ArtifactDefect = ref object of CatchableError

proc fail(msg: string): void {.noreturn.} =
  var e = ArtifactDefect()
  e.msg = msg
  raise e

func bytesOf(m: Machine): string =
  ## Copies the borrowed template text, so a write through `m: Machine` is observable.
  result = newString(m.jinja.len)
  for i in 0 ..< m.jinja.len:
    result[i] = m.jinja[i]

func payloadOf(m: Machine): seq[tuple[kind: NodeKind, slots: seq[int32]]] =
  ## Copies every node's kind and payload slots, so a render-time write into the arena is
  ## observable. Slots are compared by value, since a spilled payload's heap block address
  ## is an allocation detail, not part of the artifact's meaning.
  result = newSeq[tuple[kind: NodeKind, slots: seq[int32]]](m.nodes.len)
  for i, nd in m.nodes:
    result[i].kind = nd.kind
    for j, s in nd.slots:
      result[i].slots.add s

proc renderAll(m: Machine, t: Tables, ctx: Value, clock: float64): string =
  ## Renders once, whole, through the pull interface.
  var d = newDriver(ctx, clock)
  pullAll(m, t, d)

proc renderStepwise(m: Machine, t: Tables, ctx: Value, clock: float64): string =
  ## Renders once by stepping the driver manually:
  ##   one chunk per pull, exactly as `pullAll` would
  ## consume it. Exposed so two renders can be interleaved at chunk granularity.
  var d = newDriver(ctx, clock)
  while true:
    let before = d.cur
    var chunk = ""
    for c in items(m, t, d):
      for ch in c:
        chunk.add ch
      break
    if d.cur == before and chunk.len == 0:
      break
    result.add chunk
  doAssert d.curNode == noLink, "the walk did not reach the end of the arena"

let srcA = templateSource("deepseekv2lite")
let (nodesA, tablesA) = parseTemplate(srcA)
let machineA = Machine(jinja: srcA, nodes: nodesA)

let srcB = templateSource("moonlight")
let (nodesB, tablesB) = parseTemplate(srcB)
let machineB = Machine(jinja: srcB, nodes: nodesB)

let rowA = loadRow("deepseekv2lite", "assistant_history")
let rowA2 = loadRow("deepseekv2lite", "cjk_content")
let rowB = rows("moonlight")[0]

# Two drivers over one artifact, byte-identical to each other and to a sequential render.
block twoDriversOneMachine:
  let one = renderAll(machineA, tablesA, rowA.context, rowA.clock)
  let other = renderAll(machineA, tablesA, rowA.context, rowA.clock)
  doAssert one == rowA.rendered, "a single render is not the recorded bytes"
  doAssert one == other, "two drivers over one Machine disagreed"
  doAssert one.len > 0

  # Lockstep pulls one chunk from each driver in turn. Interleaving is the stronger claim,
  # because a render that leaked through shared state would diverge only when the two
  # drivers run interleaved.
  var d1 = newDriver(rowA.context, rowA.clock)
  var d2 = newDriver(rowA.context, rowA.clock)
  var a, b: string
  while true:
    let done1 = d1.curNode == noLink and d1.pend.kind == pkNone
    let done2 = d2.curNode == noLink and d2.pend.kind == pkNone
    if done1 and done2:
      break
    if not done1:
      for c in items(machineA, tablesA, d1):
        for ch in c:
          a.add ch
        break
    if not done2:
      for c in items(machineA, tablesA, d2):
        for ch in c:
          b.add ch
        break
  doAssert a == one, "interleaved driver 1 differs from the sequential render"
  doAssert b == one, "interleaved driver 2 differs from the sequential render"
  doAssert a.len == rowA.rendered.len

# Different contexts over one artifact at the same time:
#   drivers must not share scopes or sinks.
block sameMachineDifferentContexts:
  var d1 = newDriver(rowA.context, rowA.clock)
  var d2 = newDriver(rowA2.context, rowA2.clock)
  var a, b: string
  while true:
    let done1 = d1.curNode == noLink and d1.pend.kind == pkNone
    let done2 = d2.curNode == noLink and d2.pend.kind == pkNone
    if done1 and done2:
      break
    if not done1:
      for c in items(machineA, tablesA, d1):
        for ch in c:
          a.add ch
        break
    if not done2:
      for c in items(machineA, tablesA, d2):
        for ch in c:
          b.add ch
        break
  doAssert a == rowA.rendered, "context bleed between drivers over one Machine"
  doAssert b == rowA2.rendered, "context bleed between drivers over one Machine"
  doAssert a != b, "the two rows rendered identically, so this block proves nothing"

# Two artifacts rendered alternately:
#   neither is disturbed by the other.
block twoMachinesInterleaved:
  let wantA = renderAll(machineA, tablesA, rowA.context, rowA.clock)
  let wantB = renderAll(machineB, tablesB, rowB.context, rowB.clock)
  doAssert wantB == rowB.rendered, "moonlight alone is not the recorded bytes"

  var dA = newDriver(rowA.context, rowA.clock)
  var dB = newDriver(rowB.context, rowB.clock)
  var a, b: string
  while true:
    let doneA = dA.curNode == noLink and dA.pend.kind == pkNone
    let doneB = dB.curNode == noLink and dB.pend.kind == pkNone
    if doneA and doneB:
      break
    if not doneA:
      for c in items(machineA, tablesA, dA):
        for ch in c:
          a.add ch
        break
    if not doneB:
      for c in items(machineB, tablesB, dB):
        for ch in c:
          b.add ch
        break
  doAssert a == wantA, "interleaving a second artifact changed the first"
  doAssert b == wantB, "interleaving a second artifact changed the second"

# The artifact itself is unchanged by rendering. Same node payloads, same arena length, same text.
block artifactBytesUnchangedAfterRender:
  let before = payloadOf(machineA)
  let beforeText = bytesOf(machineA)
  let ctxBefore = machineA.nodes.len
  discard renderAll(machineA, tablesA, rowA.context, rowA.clock)
  discard renderAll(machineA, tablesA, rowA2.context, rowA2.clock)
  discard renderAll(machineA, tablesA, rowA.context, rowA.clock)
  doAssert machineA.nodes.len == ctxBefore
  var differing = -1
  for i, nd in machineA.nodes:
    if nd.kind != before[i].kind or nd.slots != before[i].slots:
      differing = i
      break
  doAssert differing < 0, "node " & $differing & " was mutated by a render"
  doAssert bytesOf(machineA) == beforeText, "the borrowed template text was written through"

# Chunking composes with `cur`:
#   chunk size must not change a byte.
block chunkSizeDoesNotChangeBytes:
  let whole = renderAll(machineA, tablesA, rowA.context, rowA.clock)
  let stepwise = renderStepwise(machineA, tablesA, rowA.context, rowA.clock)
  doAssert stepwise == whole, "chunk-by-chunk delivery differs from a single pull"
  doAssert whole == rowA.rendered
  when ChunkSize > 1:
    # With the default chunk size every corpus render is one piece, so record that the small-chunk
    # path is exercised by the `-d:ChunkSize=7` build in `run_tests.sh` rather than here.
    echo "t_twodriver: ChunkSize=", ChunkSize

echo "t_twodriver: one artifact, several drivers, identical bytes; artifact unchanged by rendering"

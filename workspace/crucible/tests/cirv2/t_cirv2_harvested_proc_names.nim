# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Harvested proc naming: one CIRv2 proc per distinct callee instantiation.
##
## Test that callees whose instantiated signatures erase identically keep
## distinct CIRv2 procs, each harvested under its own name with its own body
## (a collision runs the first callee's body, dedup keyed on the CIRv2 name).
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/cirv2 --nimcache:nimcache/tests/cirv2 \
##     workspace/crucible/tests/cirv2/t_cirv2_harvested_proc_names.nim

import std/strutils
import std/macros
import
  workspace/crucible/src/builtins/builtins_pragmas,
  workspace/crucible/src/codegen/frontend/nim_to_crucible,
  workspace/crucible/tests/cirv2/cirv2_lower_macros

proc accumulateSum[T](x: var T, y: T) = x = x + y
proc accumulateProduct[T](x: var T, y: T) = x = x * y

proc collisionKernel(y: ptr UncheckedArray[float32]) {.global.} =
  var a = 1.0'f32
  accumulateSum(a, 2.0'f32)
  accumulateProduct(a, 3.0'f32)
  y[0] = a

const collisionTrace = lowerTrace:
  collisionKernel

proc `++`(x: var float32, y: float32) = x = x + y
proc `--`(x: var float32, y: float32) = x = x - y

proc opCollisionKernel(y: ptr UncheckedArray[float32]) {.global.} =
  var a = 1.0'f32
  a ++ 2.0'f32
  a -- 3.0'f32
  y[0] = a

const opTrace = lowerTrace:
  opCollisionKernel

type ProcSection = tuple[name, body: string]

proc identAfter(trace: string, start: int, keyword: string): string =
  ## Identifier after `keyword` at `start`: characters up to the first
  ## delimiter (space, paren, or newline). Proc headers write the space
  ## between keyword and name, so the scan starts after it.
  let nameStart = start + keyword.len
  var i = nameStart
  while i < trace.len and trace[i] notin {' ', ')', '\n'}:
    i.inc
  return trace[nameStart ..< i]

proc procSections(trace: string, headerMarker: string): seq[ProcSection] =
  ## Returns sections as (name, body) pairs for every proc section whose
  ## header starts with `headerMarker`, spanning from its header line
  ## through the next proc header or the end of the trace.
  var i = 0
  while true:
    i = trace.find(headerMarker, i)
    if i < 0:
      return
    let name = trace.identAfter(i, "(proc ")
    let next = trace.find("(proc ", i + 1)
    let body = if next < 0: trace[i .. ^1] else: trace[i ..< next]
    result.add((name, body))
    i = i + 1

proc sectionNamed(secs: seq[ProcSection], prefix: string): string =
  ## First section name starting with `prefix`, "" when absent.
  for s in secs:
    if s.name.startsWith(prefix):
      return s.name

proc sectionBody(secs: seq[ProcSection], name: string): string =
  ## Body of the section named `name`, "" when absent.
  for s in secs:
    if s.name == name:
      return s.body

static:
  # each callee harvests under its own name: the name hashes callee
  # identity (definition site included), not only the erased signature
  let procSecs = collisionTrace.procSections("(proc ")
  let sumName = procSecs.sectionNamed("accumulateSum_")
  let productName = procSecs.sectionNamed("accumulateProduct_")
  # two distinct names: one harvested body cannot serve both callees
  doAssert sumName.len > 0 and productName.len > 0 and sumName != productName,
    "distinct callees with identical erased signatures share one name: " &
    sumName & " vs " & productName

  # each harvested body computes its own operation
  let sumBody = procSecs.sectionBody(sumName)
  let productBody = procSecs.sectionBody(productName)
  doAssert "(add (f +32)" in sumBody, "accumulateSum body must add: " & sumBody
  doAssert "(mul (f +32)" in productBody,
    "accumulateProduct body must multiply: " & productBody
  doAssert "(mul (f +32)" notin sumBody
  doAssert "(add (f +32)" notin productBody

static:
  # both operator callees harvest their own proc: `++` and `--` erase to one signature.
  # Callee names plus_plus_<hash> and minus_minus_<hash> keep the harvests distinct.
  var opSecs: seq[ProcSection] = @[]
  for s in opTrace.procSections("(proc "):
    if s.name.startsWith("plus_plus_") or s.name.startsWith("minus_minus_"):
      opSecs.add(s)
  doAssert opSecs.len == 2,
    "expected two harvested operator procs, got: " & $opSecs
  doAssert opSecs[0].name != opSecs[1].name,
    "distinct operator callees share one harvested proc name"

  # each call site binds its own callee's proc: a second call cannot
  # silently reuse the first callee's body
  var targets: seq[string]
  var i = 0
  while true:
    i = opTrace.find("(call ", i)
    if i < 0:
      break
    let target = opTrace.identAfter(i, "(call ")
    if target.startsWith("plus_plus_") or target.startsWith("minus_minus_"):
      targets.add(target)
    i = i + 1
  doAssert targets.len == 2,
    "expected two operator calls, got: " & $targets
  doAssert targets[0] != targets[1],
    "both operator calls bind one harvested proc: " & $targets
  doAssert targets[0] == opSecs[0].name or targets[0] == opSecs[1].name
  doAssert targets[1] == opSecs[0].name or targets[1] == opSecs[1].name

  # each harvested body computes its own operation: the second call
  # target computes a sub, not a copy of the first callee's add
  var addName, subName = ""
  for s in opSecs:
    if "(add (f +32)" in s.body:
      addName = s.name
    if "(sub (f +32)" in s.body:
      subName = s.name
  doAssert addName.len > 0, "the ++ body must lower to add"
  doAssert subName.len > 0, "the -- body must lower to sub"
  doAssert addName != subName,
    "one harvested body serves both operator callees: " & addName

# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

import std/macros
import launch_common {.all.}
export launch_common

# ▸ Types

type
  LaunchSugar*[E] = object
    ## Transient chevron sugar, created per `engine.launchKernel` access.
    engine: E

  LaunchProxy[E] = object
    engine: E
    cfg: LaunchConfig

proc makeProxy[E](engine: E, cfg: LaunchConfig): LaunchProxy[E] {.inline.} =
  LaunchProxy[E](engine: engine, cfg: cfg)

# ▸ Public API

proc launchKernel*[E](engine: E): LaunchSugar[E] {.inline.} =
  ## Chevron accessor: bare access returns chevron sugar.
  ## - Chevrons:
  ##   `engine.launchKernel<<(16, 16), (32, 8)>>("k", out, (a, b))`
  ## - Do form:
  ##     `engine.launchKernel("k", out, (a, b)) do:`
  ##       `grid = (ceil_div(m, 32), ceil_div(n, 32))`
  ##       `blk = (32, 1)`
  LaunchSugar[E](engine: engine)

macro launchKernel*[E](engine: E, kernel, output: untyped, rest: varargs[untyped]) =
  ## Plain launch or `do:`-scheduled launch, dispatches to the backend's `launchKernelImpl`.
  ## - Chevrons:
  ##   `engine.launchKernel<<(16, 16), (32, 8)>>("k", out, (a, b))`
  ## - Do form:
  ##     `engine.launchKernel("k", out, (a, b)) do:`
  ##       `grid = (ceil_div(m, 32), ceil_div(n, 32))`
  ##       `blk = (32, 1)`
  var argsNode: NimNode = nil
  var cfgNode: NimNode = nil
  var schedule: NimNode = nil
  case rest.len
  of 1:
    argsNode = rest[0]
  of 2:
    if rest[1].kind == nnkStmtList:
      argsNode = rest[0]
      schedule = rest[1]
    else:
      argsNode = rest[0]
      cfgNode = rest[1]
  of 3:
    argsNode = rest[0]
    cfgNode = rest[1]
    schedule = rest[2]
  else:
    error "launchKernel takes (kernel, output, args[, cfg][, do-schedule]), got " & $(rest.len + 3) & " args"
  doAssert cfgNode.isNil or schedule.isNil,
    "launchKernel: pass the config or a `do:` schedule, not both"
  if not schedule.isNil:
    var fields: seq[NimNode] = @[]
    for a in schedule:
      doAssert a.kind == nnkAsgn and $a[0] in ["grid", "blk", "sharedMem", "stream"],
        "launchKernel do-schedule expects grid/blk/sharedMem/stream = ..., got " & a.repr
      fields.add nnkExprColonExpr.newTree(a[0], a[1])
    doAssert fields.len >= 2,
      "launchKernel do-schedule needs at least grid and blk, got " & $fields.len & " field(s)"
    let tup = nnkPar.newTree(fields)
    cfgNode = quote do: chevronConfig(`tup`)
  if cfgNode.isNil:
    cfgNode = quote do: LaunchConfig()
  result = quote do:
    `engine`.launchKernelImpl(`kernel`, `output`, `argsNode`, `cfgNode`)

macro `<<`*[E](r: LaunchSugar[E], cfg: untyped): untyped =

  result = quote do:
    makeProxy(`r`.engine, chevronConfig(`cfg`))

macro `>>`*(proxy: typed, call: untyped): untyped =
  ## Builds the launch from the chevron proxy.
  ## - Chevrons:
  ##   `engine.launchKernel<<(16, 16), (32, 8)>>("k", out, (a, b))`
  ## - Do form:
  ##     `engine.launchKernel("k", out, (a, b)) do:`
  ##       `grid = (ceil_div(m, 32), ceil_div(n, 32))`
  ##       `blk = (32, 1)`
  doAssert call.kind in {nnkPar, nnkTupleConstr, nnkCall, nnkBracket},
    "chevron RHS must be (\"kernel\", output, (args...)), got: " & call.repr
  doAssert call.len == 3,
    "chevron RHS must be (\"kernel\", output, (args...)), got: " & call.repr
  let kernelN = call[0]
  let outputN = call[1]
  let argsN = call[2]
  result = quote do:
    `proxy`.engine.launchKernel(`kernelN`, `outputN`, `argsN`, `proxy`.cfg)

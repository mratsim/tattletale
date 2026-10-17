# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

import std/macros
from ../engines_common import HwEngine, LaunchConfig, Dim3, dim3
import launch_common {.all.}
export launch_common

# ▸ Types

type
  LaunchSugar*[E] = object
    ## Transient chevron sugar, created per `engine.launchKernel` access.
    engine: E

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
  var gridN: NimNode = nil
  var blkN: NimNode = nil
  case rest.len
  of 1:
    argsNode = rest[0]
  of 2:
    if rest[1].kind == nnkStmtList:
      argsNode = rest[0]
      for a in rest[1]:
        doAssert a.kind == nnkAsgn and $a[0] in ["grid", "blk"],
          "launchKernel do-schedule expects grid = ... / blk = ..., got " & a.repr
        if $a[0] == "grid": gridN = a[1] else: blkN = a[1]
      doAssert gridN != nil and blkN != nil,
        "launchKernel do-schedule needs grid and blk, got " & rest[1].repr
    else:
      argsNode = rest[0]
      let cfgN = rest[1]
      gridN = quote do: `cfgN`.grid
      blkN = quote do: `cfgN`.blk
  else:
    error "launchKernel takes (kernel, output, args[, cfg][, do-schedule]), got " & $(rest.len + 3) & " args"
  if gridN.isNil: gridN = quote do: dim3(1)
  if blkN.isNil:  blkN  = quote do: dim3(1)
  result = quote do:
    `engine`.launchKernelImpl(`kernel`, `output`, `argsNode`, `gridN`, `blkN`)

proc `<<`*[E, G, B: int|tuple](r: LaunchSugar[E], cfg: tuple[grid: G, blk: B]): LaunchConfig[E] =
  ## Builds the launch config from the chevron left half.
  LaunchConfig[E](engine: r.engine, grid: dim3(cfg.grid), blk: dim3(cfg.blk))

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
    `proxy`.engine.launchKernelImpl(`kernelN`, `outputN`, `argsN`, `proxy`.grid, `proxy`.blk)

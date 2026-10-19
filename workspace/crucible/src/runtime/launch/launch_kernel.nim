# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import workspace/crucible/src/runtime/engines/engines_api
import workspace/crucible/src/runtime/engines/nvrtc
import workspace/crucible/src/runtime/engines/cl
import workspace/crucible/src/runtime/engines/vk
import workspace/crucible/src/runtime/engines/wgpu
import workspace/crucible/src/runtime/engines/metal
import workspace/crucible/src/runtime/launch/launch_serialization {.all.}

# ═════════════════════════════════════════════════════════════════════════
# ▸ Types
# ═════════════════════════════════════════════════════════════════════════
type
  LaunchSugar*[E] = object
    engine: E
    cfg: LaunchConfig

# ═════════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═════════════════════════════════════════════════════════════════════════
  ## Chevron accessor: bare access returns chevron sugar.
  ## - Chevrons:
  ##   `engine.launchKernel<<(16, 16), (32, 8)>>("k", out, (a, b))`
  ## - Do form:
  ##     `engine.launchKernel("k", out, (a, b)) do:`
  ##       `grid = (ceil_div(m, 32), ceil_div(n, 32))`
  ##       `blk = (32, 1)`

proc launchKernel*[E](engine: E): LaunchSugar[E] {.inline.} =
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
  var cfgNode = nnkObjConstr.newTree(bindSym"LaunchConfig")
  case rest.len
  of 1:
    argsNode = rest[0]
  of 2:
    if rest[1].kind == nnkStmtList:
      argsNode = rest[0]
      var fields: seq[NimNode]
      for a in rest[1]:
        doAssert a.kind == nnkAsgn,
          "launchKernel do-schedule expects grid = ... / blk = ... / sharedMem = ... / stream = ..., got " & a.repr
        if $a[0] in ["grid", "blk"]:
          fields.add(newColonExpr(a[0], bindSym"dim3".newCall(a[1])))
        else:
          fields.add(newColonExpr(a[0], a[1]))
      cfgNode = nnkObjConstr.newTree(bindSym"LaunchConfig")
      for f in fields:
        cfgNode.add(f)
    else:
      argsNode = rest[0]
      cfgNode = rest[1]
  else:
    error "launchKernel takes (kernel, output, args[, cfg][, do-schedule]), got " & $(rest.len + 3) & " args"
  let impl = bindSym"launchKernelImpl"
  return quote do:
    var argsVal = `argsNode`
    `impl`(`engine`, `kernel`, `output`, argsVal, `cfgNode`)

func `<<`*[E: HwEngine, G, B: int|tuple](r: LaunchSugar[E], cfg: tuple[grid: G, blk: B]): LaunchSugar[E] {.inline.} =
  result = r
  result.cfg.grid = dim3(cfg.grid)
  result.cfg.blk = dim3(cfg.blk)

func `<<`*[E: HwEngine, G, B: int|tuple, S: int](r: LaunchSugar[E], cfg: tuple[grid: G, blk: B, sharedMem: S]): LaunchSugar[E] {.inline.} =
  result = r
  result.cfg.grid = dim3(cfg.grid)
  result.cfg.blk = dim3(cfg.blk)
  result.cfg.sharedMem = cfg.sharedMem

func `<<`*[E: HwEngine](r: LaunchSugar[E], cfg: LaunchConfig): LaunchSugar[E] {.inline.} =
  LaunchSugar[E](engine: r.engine, cfg: cfg)

macro `>>`*(proxy: typed, call: untyped): untyped =
  doAssert call.kind in {nnkPar, nnkTupleConstr, nnkCall, nnkBracket},
    "chevron RHS must be (\"kernel\", output, (args...)), got: " & call.repr
  doAssert call.len == 3,
    "chevron RHS must be (\"kernel\", output, (args...)), got: " & call.repr
  let kernelN = call[0]
  let outputN = call[1]
  let argsN = call[2]
  return quote do:
    var argsVal = `argsN`
    `proxy`.engine.launchKernelImpl(`kernelN`, `outputN`, argsVal, `proxy`.cfg)

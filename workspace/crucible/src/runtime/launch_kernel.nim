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

# ═════════════════════════════════════════════════════════════════════════
# ▸ Types
# ═════════════════════════════════════════════════════════════════════════

type
  LaunchSugar*[E] = object
    ## Transient chevron sugar, created per `engine.launchKernel` access.
    engine: E

  LaunchProxy*[E] = object
    engine: E
    cfg: LaunchConfig

proc makeProxy[E](engine: E, cfg: LaunchConfig): LaunchProxy[E] {.inline.} =
  LaunchProxy[E](engine: engine, cfg: cfg)

# ═════════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═════════════════════════════════════════════════════════════════════════

proc launchKernel*[E](engine: E): LaunchSugar[E] {.inline.} =
  ## Launch accessor, a proc never a template, templates would hijack the plain call.
  ## A proc resolves by signature, 4 args hit the engine's plain `launchKernel`,
  ## a bare `engine.launchKernel` hits this accessor.
  LaunchSugar[E](engine: engine)

macro `<<`*[E](r: LaunchSugar[E], cfg: untyped): untyped =
  ## Chevron launch config, parsed by `chevronConfig`, named or positional:
  ##   engine.launchKernel<<(grid, blk), sharedMem: n>>

  result = quote do:
    makeProxy(`r`.engine, chevronConfig(`cfg`))

macro `>>`*(proxy: typed, call: untyped): untyped =
  ## Builds the run call: `engine.launchKernel<<cfg>>("kernel", output, (args...))`
  doAssert call.kind in {nnkPar, nnkTupleConstr, nnkCall, nnkBracket},
    "chevron RHS must be (\"kernel\", output, (args...)), got: " & call.repr
  doAssert call.len == 3,
    "chevron RHS must be (\"kernel\", output, (args...)), got: " & call.repr
  let kernelN = call[0]
  let outputN = call[1]
  let argsN = call[2]
  result = quote do:
    `proxy`.engine.launchKernel(`kernelN`, `outputN`, `argsN`, `proxy`.cfg)

# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

import std/macros

# ═════════════════════════════════════════════════════════════════════════
# ▸ Types
# ═════════════════════════════════════════════════════════════════════════

type
  Dim3* = object
    ## A 3D launch dims, `Dim3(x: n)` defaults y/z to 1.
    x*, y*, z* = 1

  LaunchConfig* = object
    ## Launch geometry, grid/blk per-backend, sharedMem CUDA/OpenCL only, stream CUDA only.
    grid*, blk* = default(Dim3)
    sharedMem*, stream* = 0

  LaunchSugar*[E] = object
    ## Transient chevron sugar, created per `engine.launchKernel` access.
    engine: E

  LaunchProxy*[E] = object
    engine: E
    cfg: LaunchConfig

# ═════════════════════════════════════════════════════════════════════════
# ▸ Constructors/destructors
# ═════════════════════════════════════════════════════════════════════════
# Private: the Dim3 dim-conversion overloads and the LaunchProxy factory.
# Both must precede the `<<` macro, bindSym and quote resolve symbols at macro definition.
proc dim3[T: SomeInteger](x: T): Dim3 {.inline.} = Dim3(x: int(x))
proc dim3[T: SomeInteger](t: tuple[a: T]): Dim3 {.inline.} = Dim3(x: int(t.a))
proc dim3[T: SomeInteger](t: tuple[a, b: T]): Dim3 {.inline.} = Dim3(x: int(t.a), y: int(t.b))
proc dim3[T: SomeInteger](t: tuple[a, b, c: T]): Dim3 {.inline.} = Dim3(x: int(t.a), y: int(t.b), z: int(t.c))


proc makeProxy[E](engine: E, cfg: LaunchConfig): LaunchProxy[E] {.inline.} =
  LaunchProxy[E](engine: engine, cfg: cfg)

# ═════════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═════════════════════════════════════════════════════════════════════════

proc launchKernel*[E](engine: E): LaunchSugar[E] =
  ## `run` accessor, a proc never a template, templates would hijack the plain call.
  ## A proc resolves by signature, 4 args hit the engine's plain `run`,
  ## a bare `engine.launchKernel` hits this accessor.
  LaunchSugar[E](engine: engine)

macro `<<`*[E](r: LaunchSugar[E], cfg: untyped): untyped =
  ## Chevron launch config, named or positional:
  ##   engine.launchKernel<<(grid, blk), sharedMem: n>> / engine.launchKernel<<grid, blk, sharedMem, stream>>

  let cfgAst = cfg
  var gridN, blkN, smN, stN: NimNode
  let named = cfgAst.kind in {nnkPar, nnkTupleConstr} and
              cfgAst.len > 0 and cfgAst[0].kind == nnkExprColonExpr
  if named:
    for ch in cfgAst:
      doAssert ch.kind == nnkExprColonExpr,
        "chevron: mixing named and positional fields is not allowed: " & cfgAst.repr
      let key = ch[0].strVal
      case key
      of "grid": gridN = ch[1]
      of "blk":  blkN  = ch[1]
      of "sharedMem": smN = ch[1]
      of "stream":    stN = ch[1]
      else: doAssert false, "chevron: unknown field '" & ch[0].repr & "'"
    if gridN.isNil: gridN = newLit(1)
    if blkN.isNil:  blkN  = newLit(1)
    if smN.isNil:   smN   = newLit(0)
    if stN.isNil:   stN   = newLit(0)
  else:
    doAssert cfgAst.len in 2..4,
      "chevron positional form needs 2..4 args, got: " & cfgAst.repr
    gridN = cfgAst[0]
    blkN  = cfgAst[1]
    smN   = if cfgAst.len >= 3: cfgAst[2] else: newLit(0)
    stN   = if cfgAst.len >= 4: cfgAst[3] else: newLit(0)
  let dim3Sym = bindSym"dim3"
  result = quote do:
    makeProxy(`r`.engine, LaunchConfig(
      grid: `dim3Sym`(`gridN`), blk: `dim3Sym`(`blkN`),
      sharedMem: `smN`, stream: `stN`))

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

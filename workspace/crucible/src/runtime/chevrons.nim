# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## Chevron launch-config surface — the shared runtime types and the `<<`/`>>`
## sugar. Lives outside `runtime/engines` so the per-backend engine modules
## (runtime/engines/*) can use it WITHOUT importing back into the public
## module (breaks the circular dependency); `runtime/engines` re-exports this
## module as part of the public HwEngine API.
##
## `Dim3`/`LaunchConfig` are the launch-geometry contract; `run` is the sugar
## accessor (`engine.run`); `<<` maps a config AST to a `LaunchConfig` and
## `>>` completes the call. `makeProxy` + the `dim3` overloads are private
## helpers the `<<` macro reaches via quote/bindSym.

{.experimental: "codeReordering".}

import std/macros

# ═════════════════════════════════════════════════════════════════════════
# ▸ Types
# ═════════════════════════════════════════════════════════════════════════

type
  Dim3* = object
    ## A 3D launch extent — used for both the grid and the block axes of a
    ## launch. Nominal object (no structural collision with user tuples);
    ## field defaults make `Dim3(x: n)` → (n, 1, 1).
    x*, y*, z* = 1

  LaunchConfig* = object
    ## Launch geometry — per-backend interpretation:
    ##   grid      → CUDA gridDim / OpenCL global-per-axis (= grid·blk) /
    ##               Vulkan vkCmdDispatch group count / WebGPU
    ##               dispatchWorkgroups
    ##   blk       → CUDA blockDim / OpenCL local_work_size / shader-baked
    ##               (validated loudly) on Vulkan/WebGPU
    ##   sharedMem → CUDA dynamic smem / OpenCL __local (ignored elsewhere)
    ##   stream    → CUDA-only for now
    ## y/z are CUDA-only for now; OpenCL/Vulkan/WebGPU consume the x axis
    ## (multi-axis work sizes land in a follow-up).
    grid*, blk* = default(Dim3)
    sharedMem*, stream* = 0

  RunSugar*[E] = object
    ## Transient chevron sugar: created per `engine.run` access, owns the ref
    ## only for the chevron expression. No field on the engine, no
    ## back-pointer, no self-cycle — `=destroy` is deterministic at scope exit.
    engine: E

  LaunchProxy*[E] = object
    engine: E
    cfg: LaunchConfig

# ═════════════════════════════════════════════════════════════════════════
# ▸ Constructors/destructors
# ═════════════════════════════════════════════════════════════════════════
# Private: the Dim3 extent-conversion overloads + the LaunchProxy factory.
# Both must precede the `<<` macro — bindSym and quote resolve symbols at
# macro definition, so they cannot follow it.
proc dim3(x: int): Dim3 {.inline.} = Dim3(x: x)
proc dim3(t: tuple[a: int]): Dim3 {.inline.} = Dim3(x: t.a)
proc dim3(t: tuple[a, b: int]): Dim3 {.inline.} = Dim3(x: t.a, y: t.b)
proc dim3(t: tuple[a, b, c: int]): Dim3 {.inline.} = Dim3(x: t.a, y: t.b, z: t.c)


proc makeProxy[E](engine: E, cfg: LaunchConfig): LaunchProxy[E] {.inline.} =
  ## E is inferred from `engine` — the `<<` macro emits a clean call without
  ## spelling the generic explicitly (Nim object constructors cannot infer
  ## generic params, so the inference lives in this helper instead).
  LaunchProxy[E](engine: engine, cfg: cfg)

# ═════════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═════════════════════════════════════════════════════════════════════════

proc run*[E](engine: E): RunSugar[E] =
  ## `run` accessor — a proc, never a template: templates replace identifiers
  ## before overload resolution and would hijack `engine.run("kernel", ...)`.
  ## A proc resolves by signature: 4 args → the engine's plain `run`; a bare
  ## `engine.run` → this accessor. No clash.
  RunSugar[E](engine: engine)

macro `<<`*[E](r: RunSugar[E], cfg: untyped): untyped =
  ## Chevron launch-config sugar — field mapping only: named fields are read
  ## by name (any order; defaults grid=blk=1, sharedMem=stream=0); positional
  ## forms take 2..4 args in (grid, blk, sharedMem, stream) order. Each extent
  ## is emitted as `dim3(<raw expr>)` — the overloads convert int / 1-tuple /
  ## 2-tuple / 3-tuple (extents are positional tuples).
  ## Mixed named/positional, unknown named fields and positional counts outside
  ## 2..4 are rejected loudly at compile time.
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
  ## Builds the actual run call: `engine.run(kernel, output, args, cfg)`.
  ## `call` is the full `("kernel", output, (alpha, A, beta, B))` AST.
  doAssert call.kind in {nnkPar, nnkTupleConstr, nnkCall, nnkBracket},
    "chevron RHS must be (\"kernel\", output, (args...)), got: " & call.repr
  doAssert call.len == 3,
    "chevron RHS must be (\"kernel\", output, (args...)), got: " & call.repr
  let kernelN = call[0]
  let outputN = call[1]
  let argsN = call[2]
  result = quote do:
    `proxy`.engine.run(`kernelN`, `outputN`, `argsN`, `proxy`.cfg)




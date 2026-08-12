# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## HwEngine — the sole public runtime API.
##
##   var engine = bkCuda.init()                          # live context, no kernel yet
##   engine.ingest(source)                               # compile; drops previous artifact
##   let artifact = engine.getArtifact()                 # PTX / OpenCL src / SPIR-V / WGSL
##   engine.run("kernel", output, (alpha, A, beta, B))   # plain: engine-default geometry (grid/blk fields)
##   engine.run<<(2, 128)>>("kernel", output, args)  # chevrons: explicit 2-slot
##   engine.run<<(2, 128, 4096)>>(...)               # + sharedMem
##   engine.run<<(grid: 2, blk: 128, sharedMem: 4096, stream: 0)>>(...)  # named (colon syntax)
##   engine.ingest(otherSource)                          # reuse: invalidate + recompile
##   # =destroy via RAII resource fields at ref death
##
## BackendKind is imported from `codegen/ir/gpu_types` (the single leaf enum
## module this runtime is allowed to depend on) and re-exported: users
## importing both `codegen/gpu_compiler` and `runtime/engines` get one enum
## (no ambiguous `bkCuda`).
##
## Error policy: `check(status, quitOnFailure = true)` — stacktrace + stderr +
## quit(1). No exceptions as the public contract.
##
## Chevron gotchas (verified Nim 2.2.10):
##   - `run<<` as a single identifier is a parser minefield — always write
##     `engine.run<<(cfg)>>(...)`.
##   - The `run` accessor is a proc (never a template — templates hijack
##     `engine.run("kernel", ...)` before overload resolution).
##   - RunSugar is transient: created per `engine.run` access, holds the ref
##     only for the chevron expression, no back-pointer, deterministic `=destroy`.
##   - `block` is a Nim keyword → the launch-config field is `blk`.
##   - Named tuples use colon syntax: `(grid: 2, blk: 128)`.

import std/macros
import std/typetraits

import ../codegen/ir/gpu_types
export gpu_types

# ═════════════════════════════════════════════════════════════════════════
# Public types — the external (typed) contract
# ═════════════════════════════════════════════════════════════════════════

type
  LaunchConfig* = tuple[grid, blk, sharedMem, stream: int]
    ## Launch geometry — per-backend interpretation:
    ##   grid      → CUDA gridDim / OpenCL global (= grid·blk) / Vulkan
    ##               vkCmdDispatch group count / WebGPU dispatchWorkgroups
    ##   blk       → CUDA blockDim / OpenCL local_work_size / shader-baked
    ##               (validated loudly) on Vulkan/WebGPU
    ##   sharedMem → CUDA dynamic smem / OpenCL __local (ignored elsewhere)
    ##   stream    → CUDA-only for now

  ArgBlob* = tuple[data: pointer, size: int]
    ## Type-erased internal layer:
    ##   size >= 0 → device buffer: memcpy `size` bytes host→device, bind as
    ##               buffer/SSBO/storage
    ##   size <  0 → trivial by-value scalar of `-size` bytes (no device alloc)
    ## The output of `run` is always treated as a device buffer (uploaded
    ## before launch, read back after) regardless of the sign of its size.

  RunSugar*[E] = object
    ## Transient chevron sugar: created per `engine.run` access, owns the ref
    ## only for the chevron expression. No field on the engine, no
    ## back-pointer, no self-cycle — `=destroy` is deterministic at scope exit.
    engine: E

  LaunchProxy*[E] = object
    engine: E
    cfg: LaunchConfig

# ═════════════════════════════════════════════════════════════════════════
# Arg flattening — external macro/template layer
# ═════════════════════════════════════════════════════════════════════════
# Containers (seq/array/string) → `(addr, len·sizeof(T))` device buffers;
# scalars (incl. raw pointer values) → `(addr, -sizeof(T))` by-value.
# by the engine `run` template at the call site) so their addresses stay valid
# for the whole launch — a template-local temp would die before `runImpl`.

template blobOf*[T](x: seq[T], storage: var seq[byte]): ArgBlob =
  (data: (if x.len > 0: cast[pointer](addr x[0]) else: nil),
          size: x.len * sizeof(T))

template blobOf*[N, T](x: array[N, T], storage: var seq[byte]): ArgBlob =
  (data: cast[pointer](addr x[0]), size: sizeof(x))

template blobOf*(x: string, storage: var seq[byte]): ArgBlob =
  (data: (if x.len > 0: cast[pointer](addr x[0]) else: nil), size: x.len)

template blobOf*[T](x: T, storage: var seq[byte]): ArgBlob =
  let off = storage.len
  storage.setLen(off + sizeof(T))
  var tmp = x   # make literals/consts addressable
  copyMem(addr storage[off], addr tmp, sizeof(T))
  (data: cast[pointer](addr storage[off]), size: -sizeof(T))

template outBlob*[T](x: var seq[T]): ArgBlob =
  (data: (if x.len > 0: cast[pointer](addr x[0]) else: nil),
          size: x.len * sizeof(T))

template outBlob*[N, T](x: var array[N, T]): ArgBlob =
  (data: cast[pointer](addr x[0]), size: sizeof(x))

template outBlob*(x: var string): ArgBlob =
  (data: (if x.len > 0: cast[pointer](addr x[0]) else: nil), size: x.len)

template outBlob*[T](x: var T): ArgBlob =
  (data: cast[pointer](addr x), size: sizeof(T))

macro flattenBlobs*(args: untyped, storage: var seq[byte]): untyped =
  ## Flatten a tuple of typed kernel args into ArgBlobs, in tuple order.
  ## By-value scalars are memcpy'd into `storage` (pre-sized — no realloc, so
  ## the blobs' data pointers stay stable). A bare scalar (e.g. `(42'u32)` — a
  ## parenthesized expr, not a 1-tuple) is accepted as a single by-value blob.
  ##
  ## Implemented as a macro that emits `blobOf(el, storage)` for each original
  ## argument expression. A template that copies the tuple first (`var t =
  ## args`) is broken on this Nim 2.2.10 build: seq fields in a copied tuple
  ## go through `eqcopy` and lose their buffer identity (the blob data pointer
  ## no longer points at the caller's seq), silently corrupting the upload.
  let els =
    if args.kind in {nnkPar, nnkTupleConstr, nnkBracket}: args
    else: newTree(nnkPar, args)   # bare scalar/parenthesized expr
  let sizeVar = genSym(nskVar, "scalarBytes")
  let blobsVar = genSym(nskVar, "blobs")
  let prep = newStmtList()   # per-element scalar-size `when` statements
  let append = newStmtList() # per-element blob construction
  for el in els:
    prep.add quote do:
      when (`el` is seq) or (`el` is array) or (`el` is string):
        discard
      else:
        `sizeVar` += sizeof(`el`)
    append.add quote do:
      `blobsVar`.add blobOf(`el`, `storage`)
  result = quote do:
    block:
      `storage`.setLen(0)
      var `sizeVar` = 0
      `prep`
      `storage`.setLen(`sizeVar`)
      var `blobsVar`: seq[ArgBlob]
      `append`
      `blobsVar`

# ═════════════════════════════════════════════════════════════════════════
# Chevron machinery (generic over the engine type, verified Nim 2.2.10)
# ═════════════════════════════════════════════════════════════════════════

proc run*[E](engine: E): RunSugar[E] =
  ## `run` accessor — a proc, never a template: templates replace identifiers
  ## before overload resolution and would hijack `engine.run("kernel", ...)`.
  ## A proc resolves by signature: 4 args → the engine's plain `run`; a bare
  ## `engine.run` → this accessor. No clash.
  RunSugar[E](engine: engine)

proc `<<`*[E](r: RunSugar[E], cfg: tuple[grid, blk: int]): LaunchProxy[E] =
  LaunchProxy[E](engine: r.engine,
                 cfg: (grid: cfg[0], blk: cfg[1], sharedMem: 0, stream: 0))

proc `<<`*[E](r: RunSugar[E], cfg: tuple[grid, blk, sharedMem: int]): LaunchProxy[E] =
  LaunchProxy[E](engine: r.engine,
                 cfg: (grid: cfg[0], blk: cfg[1], sharedMem: cfg[2], stream: 0))

proc `<<`*[E](r: RunSugar[E], cfg: tuple[grid, blk, sharedMem, stream: int]): LaunchProxy[E] =
  LaunchProxy[E](engine: r.engine,
                 cfg: (grid: cfg[0], blk: cfg[1], sharedMem: cfg[2], stream: cfg[3]))

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

# ═════════════════════════════════════════════════════════════════════════
# The concept — structural contract over the four concrete engines
# ═════════════════════════════════════════════════════════════════════════

type
  HwEngine* = concept engine, type E, type T
    ## Any engine satisfying ingest/getArtifact/run and the `run` accessor.
    ## Concept procs cannot use `auto` returns — bare signatures only.
    proc ingest(engine: E, source: string)
    proc getArtifact(engine: E)
    proc run(engine: E, kernel: string, output: var T, args: tuple)
    proc run(engine: E): RunSugar[E]

# ═════════════════════════════════════════════════════════════════════════
# Per-backend engines (imported after the machinery — circular dependency)
# ═════════════════════════════════════════════════════════════════════════

import ./engines/nvrtc {.all.}
import ./engines/cl {.all.}
import ./engines/vk {.all.}
import ./engines/wgpu {.all.}

# Selective export of the public engine surface only — the {.all.} imports
# grant access to the engine modules' private factories (newCudaEngine,
# newOpenCLEngine, newVulkanEngine, newWgpuEngine) without leaking them:
# `export module` after `import {.all.}` would re-export the privates too.
export CudaEngine, OpenCLEngine, VulkanEngine, WgpuEngine
export ingest, getArtifact, run, check, deviceVendor

# ═════════════════════════════════════════════════════════════════════════
# init — static BackendKind dispatch → the concrete engine
# ═════════════════════════════════════════════════════════════════════════

proc init*(backend: static BackendKind): auto =
  ## Live context, no kernel yet. `ingest` compiles the source.
  ## Defaults: CUDA 32×128 (the historical NVRTC launch default),
  ## OpenCL 1×1 (single work-item, the historical execOpenCL default),
  ## Vulkan/WebGPU blk = the shader-baked workgroup size (validated at run).
  when backend == bkCuda:
    result = newCudaEngine(32, 128)
  elif backend == bkOpenCL:
    result = newOpenCLEngine(1, 1)
  elif backend == bkVulkan:
    result = newVulkanEngine(1, 256)
  elif backend == bkWGSL:
    result = newWgpuEngine(1, 64)
  else:
    {.error: "init: unknown BackendKind".}

proc init*(backend: static BackendKind, source: string): auto =
  ## One-shot convenience: init + ingest.
  result = init(backend)
  result.ingest(source)

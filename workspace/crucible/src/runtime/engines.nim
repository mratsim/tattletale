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
##   let artifact = engine.getArtifact()                 # PTX / OpenCL src / SPIR-V / WGSL / MSL
##   engine.run("kernel", output, (alpha, A, beta, B))   # plain: defaults — grid=blk=1 (Vulkan/WebGPU blk = shader-baked)
##   engine.run<<(2, 128)>>("kernel", output, args)              # 1D unchanged: (grid, blk)
##   engine.run<<((2,3), (128,2))>>(...)                         # 2D: tuple extents, padded to 3D
##   engine.run<<(grid: (cta_m, cta_n), blk: 256)>>(...)         # named, mixed tuple/int
##   engine.run<<(blk: 128)>>(...)                               # named subset (grid=1, sharedMem=0, stream=0 defaults)
##   engine.ingest(otherSource)                          # reuse: invalidate + recompile
##   # =destroy via RAII resource fields at ref death
##
## BackendKind is imported from `codegen/ir/gpu_types` (the single leaf enum
## module this runtime is allowed to depend on) and re-exported: users
## importing both `codegen/gpu_compiler` and `runtime/engines` get one enum
## (no ambiguous `bkCuda`).
##
## Error policy: `check(status)` — stacktrace + stderr +
## quit(1). No exceptions as the public contract.
##
## Chevron gotchas (verified Nim 2.2.10):
##   - `run<<` as a single identifier is a parser minefield — always write
##     `engine.run<<(cfg)>>(...)`.
##   - `<<` is a macro now (was 3 proc overloads): it normalizes the untyped
##     config AST — int or tuple per axis, positional 2..4 or named subset.
##   - The `run` accessor is a proc (never a template — templates hijack
##     `engine.run("kernel", ...)` before overload resolution).
##   - RunSugar is transient: created per `engine.run` access, holds the ref
##     only for the chevron expression, no back-pointer, deterministic `=destroy`.
##   - `block` is a Nim keyword → the launch-config field is `blk`.
##   - Named tuples use colon syntax: `(grid: 2, blk: 128)`.
##
## Structure: imports/exports at the top, then types, then functions
## (constructors, destructors, public, private). The chevron surface
## (Dim3/LaunchConfig/`<<`/`>>`) lives in `runtime/chevrons`, imported and
## re-exported here; the per-backend engines (runtime/engines/*) import
## chevrons directly, so this module has no circular dependency and all
## imports sit at the top.

import ../codegen/ir/gpu_types
import ./chevrons
import ./engines/arg_blobs
import ./engines/nvrtc {.all.}
import ./engines/cl {.all.}
import ./engines/vk {.all.}
import ./engines/wgpu {.all.}
import ./engines/metal {.all.}

export gpu_types
export chevrons

# Selective export of the public engine surface only — the {.all.} imports
# grant access to the engine modules' private factories (newCudaEngine,
# newOpenCLEngine, newVulkanEngine, newWgpuEngine) without leaking them:
# `export module` after `import {.all.}` would re-export the privates too.
export ingest, getArtifact, run, check, deviceName, PtrArg

# ═════════════════════════════════════════════════════════════════════════
# ▸ Types
# ═════════════════════════════════════════════════════════════════════════

type
  HwEngine* = concept engine, type E
    ## Any engine satisfying the per-engine primitives `run` dispatches to:
    ## ingest/getArtifact/runImpl/deviceName plus the `run` accessor. The
    ## launch proc `run` itself is generic here and only requires these.
    ## Concept procs cannot use `auto` returns — bare signatures only.
    proc ingest(engine: E, source: string)
    proc getArtifact(engine: E)
    proc runImpl(engine: E, kernel: string, output: ArgBlob,
                 blobs: seq[ArgBlob], cfg: LaunchConfig)
    proc deviceName(engine: E): string
    proc run(engine: E): RunSugar[E]

# ═════════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═════════════════════════════════════════════════════════════════════════

proc init*(backend: static BackendKind): auto =
  ## Live context, no kernel yet. `ingest` compiles the source.
  ## Launch geometry is chevron-only — the engine carries no grid/blk:
  ## plain `run` uses grid=blk=1 (CUDA/OpenCL/Metal) or blk = the shader-baked
  ## workgroup size (Vulkan/WebGPU).
  when backend == bkCuda:
    result = newCudaEngine()
  elif backend == bkOpenCL:
    result = newOpenCLEngine()
  elif backend == bkVulkan:
    result = newVulkanEngine()
  elif backend == bkWGSL:
    result = newWgpuEngine()
  elif backend == bkMetal:
    result = newMetalEngine()
  else:
    {.error: "init: unknown BackendKind".}

proc init*(backend: static BackendKind, source: string): auto =
  ## One-shot convenience: init + ingest.
  result = init(backend)
  result.ingest(source)

proc run*[T, A](engine: HwEngine, kernel: string, output: var T, args: A,
                cfg = LaunchConfig()) =
  ## Launch `kernel` with the output as binding 0 and the args in order
  ## (output first, per CONVENTIONS.md). The output's current bytes are
  ## uploaded before launch and read back after (in-place β·C works).
  ## Scalars bind by value (ArgBlob negative size). `cfg` carries the launch
  ## geometry — normally built by the chevrons (`run<<(grid, blk)>>`); the
  ## default is grid=blk=1 (Vulkan/WebGPU dispatch with the shader-baked
  ## workgroup size instead, see runImpl).
  ##
  ## Args are flattened here, not in a helper: the tuple is value-copied into
  ## THIS frame, so array fields live for the whole launch and the blob
  ## pointers stay valid until `runImpl` consumes them.
  var blobStorage: seq[byte]   # backing store for by-value scalars
  var blobs: seq[ArgBlob]
  when A is tuple:
    var size = 0
    for f in fields(args):
      when (typeof(f) is seq) or (typeof(f) is array) or (typeof(f) is string) or
           (typeof(f) is PtrArg):
        discard
      else:
        size += sizeof(f)
    blobStorage.setLen(0)
    blobStorage.setLen(size)
    for f in fields(args):
      when (typeof(f) is seq) or (typeof(f) is array):
        {.warning: "[crucible-perf] seq/array arg copied in full host->device at every launch; a persistent buffer avoids the copy for large tensors".}
      blobs.add blobOf(f, blobStorage)
  else:
    blobStorage.setLen(0)
    blobStorage.setLen(sizeof(A))
    blobs.add blobOf(args, blobStorage)
  # Zero-size blobs are rejected here, at the shared layer, so the four
  # engines cannot diverge on empty seq/string args. Empty arrays are
  # caught at compile time in `blobOf` (static doAssert).
  let outArg = outBlob(output)
  doAssert outArg.size != 0, "run: output must not be empty (size 0)"
  for i, b in blobs:
    doAssert b.size != 0,
      "run: arg " & $i & " of " & kernel & " is empty (size 0), " &
      "empty seq/string args are not supported"
  runImpl(engine, kernel, outArg, blobs, cfg)

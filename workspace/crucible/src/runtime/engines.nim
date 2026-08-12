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
import ./engines/nvrtc {.all.}
import ./engines/cl {.all.}
import ./engines/vk {.all.}
import ./engines/wgpu {.all.}

export gpu_types
export chevrons

# Selective export of the public engine surface only — the {.all.} imports
# grant access to the engine modules' private factories (newCudaEngine,
# newOpenCLEngine, newVulkanEngine, newWgpuEngine) without leaking them:
# `export module` after `import {.all.}` would re-export the privates too.
export ingest, getArtifact, run, check, deviceName

# ═════════════════════════════════════════════════════════════════════════
# ▸ Types
# ═════════════════════════════════════════════════════════════════════════

type
  HwEngine* = concept engine, type E, type T
    ## Any engine satisfying ingest/getArtifact/run and the `run` accessor.
    ## Concept procs cannot use `auto` returns — bare signatures only.
    proc ingest(engine: E, source: string)
    proc getArtifact(engine: E)
    proc run(engine: E, kernel: string, output: var T, args: tuple)
    proc run(engine: E): RunSugar[E]
    proc deviceName(engine: E): string

# ═════════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═════════════════════════════════════════════════════════════════════════

proc init*(backend: static BackendKind): auto =
  ## Live context, no kernel yet. `ingest` compiles the source.
  ## Launch geometry is chevron-only — the engine carries no grid/blk:
  ## plain `run` uses grid=blk=1 (CUDA/OpenCL) or blk = the shader-baked
  ## workgroup size (Vulkan/WebGPU).
  when backend == bkCuda:
    result = newCudaEngine()
  elif backend == bkOpenCL:
    result = newOpenCLEngine()
  elif backend == bkVulkan:
    result = newVulkanEngine()
  elif backend == bkWGSL:
    result = newWgpuEngine()
  else:
    {.error: "init: unknown BackendKind".}

proc init*(backend: static BackendKind, source: string): auto =
  ## One-shot convenience: init + ingest.
  result = init(backend)
  result.ingest(source)

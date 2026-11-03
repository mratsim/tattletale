# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Embedding: the product compile-time entry, Tile IR text in, a const kernel object out.
##
## `embedKernel` runs the full pipeline in the Nim VM:
## - `staticRead` of the Tile IR file
## - parse, tile passes, CIRv2 bridge, crucible's shipped pipeline
## - the emitted artifact returns as a `const`: Metal Shading Language source
##   or binary SPIR-V words from one shared CIRv2 module
##
## The host binary ships only the emitted artifact, no parser, no runtime compiler.
## Runtime parsing (`parseTileIr` on a string) stays the dev and inspection path.
##
## Example:
##   const axpy = embedKernel("fixtures/vec_add_axpy.tileir", "axpy", threads = 4)
##   const axpySpv = embedKernel("fixtures/vec_add_axpy.tileir", "axpy", threads = 4, backendKind = gbSpirv)
##
## Host side:
## - the Metal route: `engine.ingest(axpy.msl)`
## - the SPIR-V route: `engine.ingest(axpySpv.words)`
## - the launch names the kernel, see `tests/manual_embed_axpy_gemm.nim`
##
## Run:
##   nim c -r --hints:off --warnings:off --nimcache:nimcache/tattletile \
##     --outdir:build/tattletile workspace/tattletile/tests/test_embed_spirv_words.nim

import std/os

import ./tile_ir_parser
import ./low_tile

type GpuBackend* = enum
  ## Crucible backend selection for an embedded kernel.
  ## A compile-entry parameter: one shared bridge CIRv2 module feeds every backend,
  ## no tattletile lowering runs per backend.
  gbMetal
    ## Metal Shading Language source, through crucible's Metal pass composition
  gbSpirv
    ## Binary SPIR-V words (Vulkan environment) through crucible's SPIR-V passes

type EmbeddedKernel* = object
  ## A kernel embedded at compile time, ready for a crucible engine.
  ## - `name`: the entry name, passed to `engine.launchKernel`
  ## - `threadCount`: the compiled thread count, `checkLaunchThreads`
  ##   compares the launch block against it before dispatch
  ## - `msl`: the Metal artifact, populated for gbMetal
  ## - `words`: the SPIR-V artifact, populated for gbSpirv
  name*: string
  threadCount*: int
  case backend*: GpuBackend
  of gbMetal:
    msl*: string
      ## Metal Shading Language source, passed to `engine.ingest`
  of gbSpirv:
    words*: seq[uint32]
      ## Binary SPIR-V words, passed to `engine.ingest`

proc checkLaunchThreads*(k: EmbeddedKernel, blkX: SomeInteger) =
  ## Refuses a launch block whose x size differs from the compiled thread count.
  ## `embedKernel` bakes `threads` into the coordinate formula and the SPIR-V words:
  ## a smaller block leaves the upper tile coordinates unwritten, a larger one
  ## reads elements that no thread owns.
  ##
  ## Raises:
  ## - SubsetError naming the kernel and both thread counts
  if int(blkX) != k.threadCount:
    raise refuseBridge(k.name, "launch block", "x size " & $blkX & " does not match the compiled thread count " & $k.threadCount)

template embedKernel*(path: static string, entry: static string, threads: static int,
                      backendKind: static GpuBackend = gbMetal): EmbeddedKernel =
  ## Compiles one Tile IR file entry in the Nim VM, `backendKind` picks the artifact: Metal Shading Language source or binary SPIR-V words.
  ## Const-evaluable, assign the result to a `const` in the host binary.
  ##
  ## Contract:
  ## - a template, `path` resolves relative to the instantiating module, resolved at the caller's site (the `parseTileIrFile` contract)
  ## - evaluate in a `const` or `static` context, the crucible pipeline is compile-time only
  ## - one bridge CIRv2 module, `backendKind` selects crucible's pass composition and printer, no tattletile lowering per backend
  ##
  ## Example:
  ##   const axpy = embedKernel("fixtures/vec_add_axpy.tileir", "axpy", threads = 4)
  ##   const axpySpv = embedKernel("fixtures/vec_add_axpy.tileir", "axpy", threads = 4, backendKind = gbSpirv)
  ##
  ## Raises:
  ## - SubsetError naming the construct and the kernel, per parser pass
  ## - SubsetError naming the construct and the kernel, per lowering step
  when backendKind == gbMetal:
    EmbeddedKernel(
      name: entry,
      threadCount: threads,
      backend: gbMetal,
      msl: compileTileIrToMsl(parseTileIrFile(
        if path.isAbsolute: path
        else: instantiationInfo(-1, true).filename.parentDir / path), entry, threads))
  else:
    EmbeddedKernel(
      name: entry,
      threadCount: threads,
      backend: gbSpirv,
      words: compileTileIrToSpirv(parseTileIrFile(
        if path.isAbsolute: path
        else: instantiationInfo(-1, true).filename.parentDir / path), entry, threads))

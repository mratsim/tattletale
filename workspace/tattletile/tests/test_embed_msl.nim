# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Embedded fixtures on the Metal route: the product compile entry
##
## Test that `embedKernel` const-evaluates both fixtures through crucible's
## Metal pass composition to MSL source. Kernel name, backend tag and buffer slot bindings survive the pipeline.
##
## Run:
##   nim c -r --hints:off --warnings:off --nimcache:nimcache/tattletile \
##     --outdir:build/tattletile workspace/tattletile/tests/test_embed_msl.nim

import std/strutils

import workspace/tattletile/tattletile

const Axpy = embedKernel("../fixtures/vec_add_axpy.tileir", "axpy", threads = 4)
  ## Axpy kernel, compiled from Tile IR text to MSL source in the Nim VM.

const Gemm = embedKernel("../fixtures/gemm.tileir", "gemm", threads = 4)
  ## Gemm kernel, compiled from Tile IR text to MSL source in the Nim VM.

static:
  doAssert Axpy.backend == gbMetal, "the Metal route populates the msl field"
  doAssert Axpy.name == "axpy", "the kernel name reaches the embedded kernel"
  doAssert "kernel void axpy(" in Axpy.msl, "the axpy entry signature"
  # launch ABI: the output buffer binds to slot 0
  doAssert "device float* ptr_c [[buffer(0)]]" in Axpy.msl, "axpy output slot"
  doAssert "device float* ptr_a [[buffer(2)]]" in Axpy.msl, "axpy input slot"
  doAssert "device float* ptr_b [[buffer(4)]]" in Axpy.msl, "axpy input slot"

  doAssert Gemm.backend == gbMetal, "the Metal route populates the msl field"
  doAssert Gemm.name == "gemm", "the kernel name reaches the embedded kernel"
  doAssert "kernel void gemm(" in Gemm.msl, "the gemm entry signature"
  doAssert "device float* ptr_c [[buffer(0)]]" in Gemm.msl, "gemm output slot"
  doAssert "device float* ptr_a [[buffer(1)]]" in Gemm.msl, "gemm input slot"
  doAssert "device float* ptr_b [[buffer(2)]]" in Gemm.msl, "gemm input slot"

when isMainModule:
  echo "[OK] embedKernel gbMetal: both fixtures const-evaluate to MSL with the launch ABI"

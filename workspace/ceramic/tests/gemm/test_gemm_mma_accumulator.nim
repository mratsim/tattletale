# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Host checks for the NVIDIA MMA asm's accumulator register class.
## The atom's instruction signature carries the accumulator type in its D-type dot token.
## - Dispatch derives the accumulator element from the signature
## - the builder turns the element into the asm constraint
## Constraint per element:
## - float32 → "f", the float registers
## - int32 (the s32 atoms) → "r", the integer registers
## Device-side asm emission is checked on the CUDA box (pending).
## Runs on the CPU, no GPU needed:
##   nim cpp -r --outdir:build/tests/test_gemm_mma_accumulator workspace/ceramic/tests/gemm/test_gemm_mma_accumulator.nim

import std/strutils
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/src/hardware/h_mma_dispatch

proc testAccumulatorElement() =
  ## The accumulator element derives from the instruction's D-type token,
  ## the instruction string is the ground truth.
  doAssert accumulatorElementName(
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32") == "float32"
  doAssert accumulatorElementName(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32") == "float32"
  doAssert accumulatorElementName(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32") == "float32"
  doAssert accumulatorElementName(
    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32") == "int32"
  doAssert accumulatorElementName(
    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32") == "float32"

proc testAccumulatorConstraints() =
  ## The constraint clause carries the accumulator element's register class:
  ## - int32 → integer registers ("r"), never the float registers ("f")
  ## - float32 → the float registers ("f")
  let s32Asm = buildNvidiaMmaAsm(
    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32", 16, 8, 4,
    "d", "a", "b", "d", "int32", "uint32", "uint32", "int32")
  doAssert s32Asm.contains("\"+r\"(`d0`)"), s32Asm
  doAssert not s32Asm.contains("\"f\""), s32Asm
  let f32Asm = buildNvidiaMmaAsm(
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32", 4, 2, 4,
    "d", "a", "b", "d", "float32", "uint32", "uint32", "float32")
  doAssert f32Asm.contains("\"+f\"(`d0`)"), f32Asm

testAccumulatorElement()
testAccumulatorConstraints()
echo "  [OK] mma accumulator: the s32 atoms' accumulator takes integer registers (\"r\"), the f32 atoms' float registers (\"f\"), both derived from the instruction signature"

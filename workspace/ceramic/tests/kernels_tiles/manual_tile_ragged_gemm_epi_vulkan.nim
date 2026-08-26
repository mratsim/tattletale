## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run: nim cpp -r --hints:off --warnings:off \
##   --outdir:build/tests/manual_tile_ragged_gemm_epi_vulkan \
##   --nimcache:nimcache/tests/manual_tile_ragged_gemm_epi_vulkan \
##   workspace/ceramic/tests/kernels_tiles/manual_tile_ragged_gemm_epi_vulkan.nim

## Vulkan GEMM epilogue probe (kEnds-free, ragged-edge shapes).
##
## Compiles `gemm_with_epilogue` with the REAL `vulkan:` macro (the
## Vulkan IR legalization passes run inside the macro), glslang-checks the
## emitted GLSL, and value-runs it on MoltenVK vs an fp32-exact host
## reference. Ragged = the tile does not divide M/N/K (operator definition):
## the shapes below are deliberately not multiples of 32/16. The host
## zero-fills the K-padding (columns/rows K..Kp-1) so the kernel's K loop
## over Kp/16 blocks stays exact; the M/N padding is garbage and must never
## leak into the real M×N region. No kEnds, no Lengths, no caller padding
## in the kernel API.
##
## Two kernels: the plain strided-C epilogue (EpiAXPBYStrided, α=2/β=4) and
## a user-defined scale epilogue (EpiScale, a plain value struct). Each has
## a Metal twin run for cross-backend parity (the kernels are
## backend-agnostic; all adaptation lives in the Vulkan legalization passes).

import std/[os, osproc, strformat, strutils, tempfiles]
import workspace/crucible
import ../tile_test_utils
import ../libtest_epilogues
import ../../src/atoms_mma_partitioning
import ../../src/kernels/k_tile_gemm

{.experimental: "callOperator".}

# ═════════════════════════════════════════════════════════════════════════
#  User epilogue: a plain value struct plus an `apply` proc (no kEnds)
# ═════════════════════════════════════════════════════════════════════════

type EpiScale[T] = object
  ## Scale epilogue: D = s·AB. Plain scalar field — the Vulkan passes must
  ## carry it as a value struct (unlike EpiAXPBYStrided, whose ptr field is
  ## flattened to leaves).
  s*: T

func apply[T, Sh, StAB, StR](
    op: EpiScale[T];
    tmp: var (TensorView[T, Sh, StR] or Tensor[T, Sh, StR]);
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  ## Per-thread epilogue math: tmp = s·AB.
  const S = toIntVal(size(tmp))
  for i in 0 ..< S:
    tmp(i) = op.s * AB(i)

func apply[T; R, C: static int; AT, ABT: static MmaAtom](
    op: EpiScale[T];
    tmp: var RtLeft[T, R, C, AT];
    AB: RtLeft[T, R, C, ABT]) {.inline.} =
  ## Per-slot epilogue math (fragment-resident accumulator form): tmp = s·AB.
  static:
    doAssert AT.getM() == ABT.getM() and AT.getN() == ABT.getN() and
      toIntVal(AT.valuesPerThread(opC)) == toIntVal(ABT.valuesPerThread(opC)),
      "apply: the accumulator and operand tiles must share the atom's subtile grid and per-lane count"
  const rowTiles = R div AT.getM()
  const colTiles = C div AT.getN()
  const vpt = toIntVal(AT.valuesPerThread(opC))
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        tmp.frags[n][m].frag[v] = op.s * AB.frags[n][m].frag[v]

static:
  doAssert EpiScale[float32] is Epilogue, "EpiScale must satisfy the Epilogue concept"

# ═════════════════════════════════════════════════════════════════════════
#  Vulkan kernels (REAL `vulkan:` macro — the pass pipeline runs in-macro)
#  One kernel per source: the Vulkan engine only ingests single-kernel
#  sources with scalar params (see engines/vk.nim ingest contract).
# ═════════════════════════════════════════════════════════════════════════

const plainEpiVk = vulkan:
  proc plainEpi(D: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16],
                C: ptr UncheckedArray[float32], Alpha, Beta: float32,
                M, N, K: int32, rsc, csc: int32) {.global, workgroup: (32, 1, 1).} =
    gemm_with_epilogue(D, N, 1, A, K, 1, B, N, 1, M, K, N,
      initEpiAXPBY(Alpha, Beta, C, rsc, csc), C)

const scaleUserVk = vulkan:
  proc fusedScaleUser(D: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16],
                      Scale: float32, M, N, K: int32) {.global, workgroup: (32, 1, 1).} =
    gemm_with_epilogue(D, N, 1, A, K, 1, B, N, 1, M, K, N,
      EpiScale[float32](s: Scale))

# ═════════════════════════════════════════════════════════════════════════
#  Metal twins — the SAME kernels on Metal: the kernels + host refs are
#  correct, so any Vulkan value mismatch is squarely in the Vulkan lowering.
# ═════════════════════════════════════════════════════════════════════════

const gemmVkMsl = metal:
  proc plainEpi(D: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16],
                C: ptr UncheckedArray[float32], Alpha, Beta: float32,
                M, N, K: int32, rsc, csc: int32) {.global.} =
    gemm_with_epilogue(D, N, 1, A, K, 1, B, N, 1, M, K, N,
      initEpiAXPBY(Alpha, Beta, C, rsc, csc), C)

  proc fusedScaleUser(D: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16],
                      Scale: float32, M, N, K: int32) {.global.} =
    gemm_with_epilogue(D, N, 1, A, K, 1, B, N, 1, M, K, N,
      EpiScale[float32](s: Scale))

# ═════════════════════════════════════════════════════════════════════════
#  Host inputs + fp32-exact reference (S7 harness, source of truth)
# ═════════════════════════════════════════════════════════════════════════

proc buildRaggedEdgeA(M, Mp, K, Kp: int): seq[uint16] =
  ## A over the padded Mp×Kp buffer: the real M×K region carries the exact
  ## fp16 pattern 1 + 2m + 7k; the K-padding columns (k in K..Kp-1) are
  ## ZERO (the kernel reads them inside the K loop, so garbage would leak
  ## into the accumulator); the padded M rows stay 0xDEAD garbage — their
  ## D outputs are outside the checked M×N region.
  result = newSeq[uint16](Mp * Kp)
  for i in 0 ..< Mp * Kp:
    result[i] = 0xDEAD'u16
  for m in 0 ..< M:
    for k in 0 ..< K:
      result[m * Kp + k] = fp32ToFp16(float32(1 + 2 * m + 7 * k))
    for k in K ..< Kp:
      result[m * Kp + k] = 0'u16

proc buildRaggedEdgeB(K, Kp, N, Np: int): seq[uint16] =
  ## B over the padded Kp×Np buffer: the real K×N region carries the exact
  ## fp16 pattern 1 + 3k + 11n; the K-padding rows (k in K..Kp-1) are ZERO;
  ## the padded N columns stay 0xDEAD garbage.
  result = newSeq[uint16](Kp * Np)
  for i in 0 ..< Kp * Np:
    result[i] = 0xDEAD'u16
  for k in 0 ..< K:
    for n in 0 ..< N:
      result[k * Np + n] = fp32ToFp16(float32(1 + 3 * k + 11 * n))
    for n in N ..< Np:
      result[k * Np + n] = 0'u16

proc checkClose(name: string; actual, expected: seq[float32];
                M, N, Np: int; tol = 1e-2'f32) =
  ## Element-wise |a − b| ≤ tol over the real M×N region (actual is
  ## Mp×Np-strided, expected M×N), worst |Δ| reported.
  var worstD = 0.0'f32
  var nBad = 0
  var firstBad = ""
  for m in 0 ..< M:
    for n in 0 ..< N:
      let d = abs(actual[m * Np + n] - expected[m * N + n])
      if d > worstD: worstD = d
      if actual[m * Np + n] != actual[m * Np + n] or d > tol:
        inc nBad
        if firstBad.len == 0:
          firstBad = &"(m={m}, n={n}): got {actual[m * Np + n]}, want {expected[m * N + n]}"
  echo "  ", name, ": ",
       (if nBad == 0: &"PASS — worst |Δ| = {worstD} (tolerance {tol})"
        else: &"FAIL — {nBad} bad, e.g. {firstBad}, worst |Δ| = {worstD}")
  if nBad != 0:
    quit 1

proc checkPlain(engine: var auto; M, N, K: int) =
  ## D = 2·AB + 4·C, C at runtime strides (2·Np, 1), on a ragged-edge shape:
  ## M/N are not multiples of 32 and K is not a multiple of 16, so the grid
  ## covers padding tiles and the K loop reads a zero-filled partial block.
  ## The host zero-fills only the K-padding; the M/N padding is garbage and
  ## must never leak into the checked M×N region.
  let Mp = ((M + 31) div 32) * 32
  let Np = ((N + 31) div 32) * 32
  let Kp = ((K + 15) div 16) * 16
  let Ah = buildRaggedEdgeA(M, Mp, K, Kp)
  let Bh = buildRaggedEdgeB(K, Kp, N, Np)
  let rsc = 2 * Np
  let csc = 1
  var Cp = newSeq[float32](Mp * rsc)
  for i in 0 ..< Mp * rsc:
    Cp[i] = 0xDEAD'f32      # garbage padding
  for m in 0 ..< M:
    for n in 0 ..< N:
      Cp[m * rsc + n * csc] = float32(1 + 3 * m + 5 * n)
  var D = newSeq[float32](Mp * Np)
  for i in 0 ..< Mp * Np:
    D[i] = 0xDEAD'f32
  echo &"  plainEpi {M}×{N}×{K} (α=2, β=4, strided C, garbage M/N padding):"
  engine.run << (grid: (Np div 32, Mp div 32), blk: (32, 1)) >> (
    "plainEpi", D, (Ah, Bh, Cp, 2.0'f32, 4.0'f32,
                    int32(Mp), int32(Np), int32(Kp), int32(rsc), int32(csc)))
  var Cref = newSeq[float32](M * N)
  for m in 0 ..< M:
    for n in 0 ..< N:
      Cref[m * N + n] = Cp[m * rsc + n * csc]
  let refC = sum(scale(gemmRef(M, N, K, Mp, Np, Kp, Ah, Bh), 2.0'f32),
                 scale(Cref, 4.0'f32))
  checkClose("plainEpi vs fp32-exact reference", D, refC, M, N, Np)

proc checkScaleUser(engine: var auto; M, N, K: int) =
  ## The user epilogue (EpiScale, plain value struct): D = 2·AB, on a
  ## ragged-edge shape with the same zero-filled K-padding.
  let Mp = ((M + 31) div 32) * 32
  let Np = ((N + 31) div 32) * 32
  let Kp = ((K + 15) div 16) * 16
  let Ah = buildRaggedEdgeA(M, Mp, K, Kp)
  let Bh = buildRaggedEdgeB(K, Kp, N, Np)
  var D = newSeq[float32](Mp * Np)
  echo &"  fusedScaleUser {M}×{N}×{K} (s=2, garbage M/N padding):"
  engine.run << (grid: (Np div 32, Mp div 32), blk: (32, 1)) >> (
    "fusedScaleUser", D, (Ah, Bh, 2.0'f32, int32(Mp), int32(Np), int32(Kp)))
  checkClose("fusedScaleUser vs fp32-exact reference", D,
             scale(gemmRef(M, N, K, Mp, Np, Kp, Ah, Bh), 2.0'f32), M, N, Np)

# ═════════════════════════════════════════════════════════════════════════
#  glslangValidator check (fail-hard, house pattern from test_tile_ops_vulkan)
# ═════════════════════════════════════════════════════════════════════════

proc glslangCheck(src: string; renameFrom: string; label: string) =
  ## Compile the shader with glslangValidator (-V, vulkan1.1 target) after
  ## renaming the target kernel to `main`, the same per-kernel recompilation
  ## the Vulkan engine does. A missing tool or any rejection fails the test.
  var s = src.replace("void " & renameFrom & "()", "void main()")
  let (tmpFile, tmpPath) = createTempFile("vk_gemm_epi", ".comp")
  defer: tmpFile.close()
  tmpFile.write(s)
  tmpFile.flushFile()
  let (outp, exitCode) = execCmdEx(
    "glslangValidator -V --target-env vulkan1.1 " & quoteShell(tmpPath) & " -o /dev/null")
  doAssert exitCode == 0,
    "glslangValidator rejected " & label & ":\n" & outp & "\n--- shader ---\n" & s

# ═════════════════════════════════════════════════════════════════════════
#  Runner
# ═════════════════════════════════════════════════════════════════════════

proc runTest() =   # engines are RAII, so keep them function-local
  checkDrawPins()          # the hash draws must not drift

  # ── Emission part (Vulkan): the legalization passes must have lowered
  #    every ptr/var-param construct — these asserts are the mutation bite
  #    (reverting a pass changes the shapes below and fails the probe).
  echo "── plainEpiVk GLSL (" & $plainEpiVk.len & " chars) ──"
  doAssert "void plainEpi()" in plainEpiVk, "missing kernel entry point:\n" & plainEpiVk
  doAssert "float16_t A[]" in plainEpiVk, "missing A fp16 SSBO:\n" & plainEpiVk
  doAssert "layout(push_constant)" in plainEpiVk, "missing push-constant block:\n" & plainEpiVk
  doAssert "d_rtl = zero" in plainEpiVk,
    "pass 1b value-return shape missing (var-param fn must return by value):\n" & plainEpiVk
  doAssert "a_rtl = loadTile" in plainEpiVk,
    "pass 1b value-return shape missing (loadTile must return by value):\n" & plainEpiVk
  doAssert "d_rtl = mma_AB" in plainEpiVk,
    "pass 1b value-return shape missing (mma_AB must return by value):\n" & plainEpiVk
  doAssert "float16_t*" notin plainEpiVk,
    "device fn still carries a float16_t* param (pass 3 binding missed):\n" & plainEpiVk
  doAssert "float*" notin plainEpiVk,
    "device fn still carries a float* param (pass 3 binding missed):\n" & plainEpiVk
  doAssert "EpiAXPBYStrided" notin plainEpiVk and "StridedOperand" notin plainEpiVk,
    "tainted epilogue struct leaked into GLSL (pass 2 flatten missed):\n" & plainEpiVk

  echo "── scaleUserVk GLSL (" & $scaleUserVk.len & " chars) ──"
  doAssert "void fusedScaleUser()" in scaleUserVk, "missing kernel entry point:\n" & scaleUserVk
  doAssert "struct EpiScalef32" in scaleUserVk,
    "user epilogue must survive as a plain value struct:\n" & scaleUserVk
  doAssert "EpiScalef32(Scale)" in scaleUserVk,
    "user epilogue construction missing:\n" & scaleUserVk
  doAssert "float16_t*" notin scaleUserVk and "float*" notin scaleUserVk,
    "device fn still carries a ptr param:\n" & scaleUserVk

  # ── Metal twins (blast radius): the same kernels still compile on Metal ──
  doAssert "plainEpi" in gemmVkMsl and "fusedScaleUser" in gemmVkMsl,
    "Metal twin drift:\n" & gemmVkMsl

  # ── glslangValidator (per-kernel, like the engine) ──────────────────────
  glslangCheck(plainEpiVk, "plainEpi", "plainEpi")
  glslangCheck(scaleUserVk, "fusedScaleUser", "fusedScaleUser")
  echo "  glslangValidator: plainEpi + fusedScaleUser → SPIR-V OK"

  # ── MoltenVK value runs vs the fp32-exact host reference ────────────────
  var vkEngine = bkVulkan.init()
  vkEngine.ingest(plainEpiVk)
  checkPlain(vkEngine, 37, 55, 48)    # ragged M/N; K divides 16 exactly
  checkPlain(vkEngine, 65, 89, 71)    # ragged M/N/K (Kp = 80, partial block)
  vkEngine.ingest(scaleUserVk)
  checkScaleUser(vkEngine, 65, 89, 71)

  # ── Metal twin runs: the same kernels, same reference (cross-backend) ───
  var mslEngine = bkMetal.init()
  mslEngine.ingest(gemmVkMsl)
  checkPlain(mslEngine, 37, 55, 48)
  checkPlain(mslEngine, 65, 89, 71)
  checkScaleUser(mslEngine, 65, 89, 71)

  echo "manual_tile_ragged_gemm_epi_vulkan: Vulkan gemm_with_epilogue (plain strided + user epilogue, ragged-edge shapes) PASS on MoltenVK + Metal twins"

when isMainModule:
  runTest()

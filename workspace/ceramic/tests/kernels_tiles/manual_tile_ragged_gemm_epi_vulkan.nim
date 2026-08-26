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

## Vulkan GEMM epilogue test (ragged-edge shapes).
##
## Compiles `gemm_with_epilogue` with the `vulkan:` macro (the
## Vulkan IR legalization passes run inside the macro) and value-runs it on
## MoltenVK vs an fp32-exact host reference. Ragged = the tile does not
## exactly divide M/N/K; every GEMM handles it natively in-kernel — no
## Lengths params, no padding descriptors. The shapes below are
## deliberately not multiples of 32/16. B's K-padding rows (k in K..Kp-1)
## are ZERO — the kernel's K loop reads them, and A's K-padding columns
## stay 0xDEAD garbage, so exactness relies on garbage × 0 = 0 (this
## pairing makes the B zero-fill essential in the value runs). The M/N
## padding is garbage and must never leak into the real M×N region.
##
## Two kernels: the plain strided-C epilogue (EpiAXPBYStrided, α=2/β=4)
## and a user-defined scale epilogue (EpiScale, a plain value struct).
## Each has a Metal twin run for cross-backend parity (the kernels are
## backend-agnostic: all adaptation lives in the Vulkan legalization
## passes).

import std/[strformat, strutils]
import workspace/crucible
import ../tile_test_utils
import ../libtest_epilogues
import ../../src/atoms_mma_partitioning
import ../../src/kernels/k_tile_gemm

{.experimental: "callOperator".}

# ═════════════════════════════════════════════════════════════════════════
#  User epilogue: a plain value struct plus an `apply` proc
# ═════════════════════════════════════════════════════════════════════════

type EpiScale = object
  ## Scale epilogue: D = s·AB. Plain scalar field — the Vulkan passes must
  ## carry it as a value struct (unlike EpiAXPBYStrided, whose ptr field is
  ## flattened to leaves).
  s*: float32

func apply(
    op: EpiScale,
    tmp: var TensorView[float32, (Int[32], Int[32]), (Int[32], Int[1])],
    AB: TensorView[float32, (Int[32], Int[32]), (Int[32], Int[1])]) {.inline.} =
  ## Per-thread epilogue math: tmp = s·AB.
  const S = toIntVal(size(tmp))
  for i in 0 ..< S:
    tmp(i) = op.s * AB(i)

func apply(
    op: EpiScale,
    tmp: var RtLeft[float32, 32, 32, UNIVERSAL_8x8x8_F32F16F16F32],
    AB: RtLeft[float32, 32, 32, UNIVERSAL_8x8x8_F32F16F16F32]) {.inline.} =
  ## Per-slot epilogue math (fragment-resident accumulator form): tmp = s·AB.
  static:
    doAssert 32 mod UNIVERSAL_8x8x8_F32F16F16F32.getM() == 0 and
      32 mod UNIVERSAL_8x8x8_F32F16F16F32.getN() == 0,
      "apply: the accumulator tile dims must be exact multiples of the atom dims"
  const rowTiles = 32 div UNIVERSAL_8x8x8_F32F16F16F32.getM()
  const colTiles = 32 div UNIVERSAL_8x8x8_F32F16F16F32.getN()
  const vpt = toIntVal(UNIVERSAL_8x8x8_F32F16F16F32.valuesPerThread(opC))
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        tmp.frags[n][m].frag[v] = op.s * AB.frags[n][m].frag[v]

func apply(
    op: EpiScale,
    tmp: var RtLeft[float32, 32, 32, APPLE_8x8x8_F16],
    AB: RtLeft[float32, 32, 32, APPLE_8x8x8_F16]) {.inline.} =
  ## Per-slot epilogue math for the Metal twin (Apple atom): tmp = s·AB.
  static:
    doAssert 32 mod APPLE_8x8x8_F16.getM() == 0 and
      32 mod APPLE_8x8x8_F16.getN() == 0,
      "apply: the accumulator tile dims must be exact multiples of the atom dims"
  const rowTiles = 32 div APPLE_8x8x8_F16.getM()
  const colTiles = 32 div APPLE_8x8x8_F16.getN()
  const vpt = toIntVal(APPLE_8x8x8_F16.valuesPerThread(opC))
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        tmp.frags[n][m].frag[v] = op.s * AB.frags[n][m].frag[v]

static:
  doAssert EpiScale is Epilogue, "EpiScale must satisfy the Epilogue concept"

# ═════════════════════════════════════════════════════════════════════════
#  Vulkan kernels (`vulkan:` macro — the pass pipeline runs in-macro)
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
      EpiScale(s: Scale))

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
      EpiScale(s: Scale))

# ═════════════════════════════════════════════════════════════════════════
#  Host inputs + fp32-exact host reference (source of truth for both backends)
# ═════════════════════════════════════════════════════════════════════════

proc buildRaggedEdgeA(M, Mp, K, Kp: int): seq[uint16] =
  ## A over the padded Mp×Kp buffer: the real M×K region carries the exact
  ## fp16 pattern 1 + 2m + 7k; the K-padding columns (k in K..Kp-1) stay
  ## 0xDEAD garbage — exactness relies on B's K-padding rows being ZERO
  ## (garbage × 0 = 0; garbage × garbage would leak finite wrong values
  ## into the accumulator — which is how the B zero-fill is essential in
  ## the value runs); the padded M rows stay 0xDEAD garbage — their
  ## D outputs are outside the checked M×N region.
  result = newSeq[uint16](Mp * Kp)
  for i in 0 ..< Mp * Kp:
    result[i] = 0xDEAD'u16
  for m in 0 ..< M:
    for k in 0 ..< K:
      result[m * Kp + k] = fp32ToFp16(float32(1 + 2 * m + 7 * k))

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
  # the K-padding rows (k in K..Kp-1) must be ZERO to match the comment
  # (the kernel reads them inside the K loop — 0xDEAD fp16 is a finite
  # number ≈ −341.3, and a garbage change to Inf/NaN would poison the
  # real M×N region through 0×garbage)
  for k in K ..< Kp:
    for n in 0 ..< Np:
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
  ## covers padding tiles and the K loop reads a partial block: B's
  ## K-padding rows are zero while A's K-padding columns stay garbage
  ## (garbage × 0 = 0 keeps the result exact vs the reference's real-K sum).
  ## The M/N padding is garbage and must never leak into the checked M×N region.
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
  ## ragged-edge shape with the same K-padding pairing (A garbage, B zero).
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

proc checkKGuard(engine: var auto; M, N, K: int) =
  ## HPC-A-002: an UNPADDED K (K mod 16 != 0) must fail loudly — the kernel
  ## returns before writing D, so the untouched 0xDEAD marker survives.
  ## Without the guard the kernel would silently GEMM over K div 16 full
  ## blocks and write a finite truncated result (not NaN) into D.
  let Mp = ((M + 31) div 32) * 32
  let Np = ((N + 31) div 32) * 32
  # Kp == K here: deliberately NO K-padding — the guard must fire before
  # the K loop reads anything.
  let Ah = buildRaggedEdgeA(M, Mp, K, K)
  let Bh = buildRaggedEdgeB(K, K, N, Np)
  let rsc = 2 * Np
  let csc = 1
  var Cp = newSeq[float32](Mp * rsc)
  for i in 0 ..< Mp * rsc:
    Cp[i] = 0xDEAD'f32
  var D = newSeq[float32](Mp * Np)
  for i in 0 ..< Mp * Np:
    D[i] = 0xDEAD'f32
  echo &"  plainEpi {M}×{N}×{K} (UNPADDED K: guard must fire, D untouched):"
  engine.run << (grid: (Np div 32, Mp div 32), blk: (32, 1)) >> (
    "plainEpi", D, (Ah, Bh, Cp, 2.0'f32, 4.0'f32,
                    int32(Mp), int32(Np), int32(K), int32(rsc), int32(csc)))
  if D[0] != 0xDEAD'f32:
    echo &"  FAIL — D[0] = {D[0]} (guard did not fire: silent truncated GEMM)"
    quit 1
  echo "  PASS — D[0] untouched (guard fired before any write)"

# ═════════════════════════════════════════════════════════════════════════
#  Runner
# ═════════════════════════════════════════════════════════════════════════

proc runTest() =   # engines are RAII, so keep them function-local
  checkDrawPins()          # shared runner: seed-derived shapes must stay stable

  # ── Emission part (Vulkan): the legalization passes must have lowered
  #    every ptr/var-param construct — reverting a pass changes the shapes
  #    below and fails this test.
  echo "── plainEpiVk GLSL (" & $plainEpiVk.len & " chars) ──"
  doAssert "void plainEpi()" in plainEpiVk, "missing kernel entry point:\n" & plainEpiVk
  doAssert "float16_t A[]" in plainEpiVk, "missing A fp16 SSBO:\n" & plainEpiVk
  doAssert "layout(push_constant)" in plainEpiVk, "missing push-constant block:\n" & plainEpiVk
  doAssert "d_rtl = zero" in plainEpiVk,
    "value-return shape missing (var-param fn must return by value):\n" & plainEpiVk
  doAssert "a_rtl = loadTile" in plainEpiVk,
    "value-return shape missing (loadTile must return by value):\n" & plainEpiVk
  doAssert "d_rtl = mma_AB" in plainEpiVk,
    "value-return shape missing (mma_AB must return by value):\n" & plainEpiVk
  doAssert "float16_t*" notin plainEpiVk,
    "device fn still carries a float16_t* param (pass 3 binding missed):\n" & plainEpiVk
  doAssert "float*" notin plainEpiVk,
    "device fn still carries a float* param (pass 3 binding missed):\n" & plainEpiVk
  doAssert "EpiAXPBYStrided" notin plainEpiVk and "StridedOperand" notin plainEpiVk,
    "tainted epilogue struct leaked into GLSL (pass 2 flatten missed):\n" & plainEpiVk
  # GPU-B-001: the fp16-subgroup shuffle path (tileKMax reduction trees,
  # universalMma8x8x8) assumes 32-lane subgroups. Pass 4 injects a
  # fail-loudly guard as the kernel's FIRST statement (a <32-lane device
  # returns without writing its outputs, so host value checks fail) and
  # reads the lane id from gl_SubgroupInvocationID (the true subgroup
  # lane) inside the shuffle fns — while non-shuffle fns keep
  # gl_LocalInvocationIndex (the workgroup lane; equal only because the
  # guard fixes subgroup 32 == the kernels' baked 32-wide workgroups).
  doAssert "void plainEpi() {\nif (gl_SubgroupSize < 32u) { return; }" in plainEpiVk,
    "GPU-B-001: missing gl_SubgroupSize<32 fail-loudly guard as first stmt:\n" & plainEpiVk
  doAssert "gl_SubgroupInvocationID" in plainEpiVk,
    "GPU-B-001: shuffle lane id not rewritten to gl_SubgroupInvocationID:\n" & plainEpiVk
  doAssert "int lane = int(gl_LocalInvocationIndex)" in plainEpiVk,
    "GPU-B-001: lane-id rewrite leaked into non-shuffle fns:\n" & plainEpiVk
  # HPC-A-002: ragged-K (K mod 16 != 0) fails loudly — the guard returns
  # before any write, so a host-side value check sees the untouched output
  # instead of a silently truncated result (the K loop iterates K div 16
  # full blocks and would drop the tail).
  doAssert "if (!((K % 16) == 0)) {" in plainEpiVk,
    "HPC-A-002: missing ragged-K fail-loudly guard:\n" & plainEpiVk

  echo "── scaleUserVk GLSL (" & $scaleUserVk.len & " chars) ──"
  doAssert "void fusedScaleUser()" in scaleUserVk, "missing kernel entry point:\n" & scaleUserVk
  doAssert "struct EpiScale" in scaleUserVk,
    "user epilogue must survive as a plain value struct:\n" & scaleUserVk
  doAssert "EpiScale(Scale)" in scaleUserVk,
    "user epilogue construction missing:\n" & scaleUserVk
  doAssert "float16_t*" notin scaleUserVk and "float*" notin scaleUserVk,
    "device fn still carries a ptr param:\n" & scaleUserVk
  doAssert "void fusedScaleUser() {\nif (gl_SubgroupSize < 32u) { return; }" in scaleUserVk,
    "GPU-B-001: missing gl_SubgroupSize<32 fail-loudly guard as first stmt:\n" & scaleUserVk
  doAssert "gl_SubgroupInvocationID" in scaleUserVk,
    "GPU-B-001: shuffle lane id not rewritten to gl_SubgroupInvocationID:\n" & scaleUserVk
  doAssert "int lane = int(gl_LocalInvocationIndex)" in scaleUserVk,
    "GPU-B-001: lane-id rewrite leaked into non-shuffle fns:\n" & scaleUserVk

  # ── Metal twins: the same kernels still compile on Metal ──
  doAssert "plainEpi" in gemmVkMsl and "fusedScaleUser" in gemmVkMsl,
    "Metal twin drift:\n" & gemmVkMsl

  # ── MoltenVK value runs vs the fp32-exact host reference ────────────────
  #    (engine.ingest is the fail-hard glslangValidator → SPIR-V compile path)
  var vkEngine = bkVulkan.init()
  vkEngine.ingest(plainEpiVk)
  checkPlain(vkEngine, 37, 55, 48)    # ragged M/N; K divides 16 exactly
  checkPlain(vkEngine, 65, 89, 71)    # ragged M/N/K (Kp = 80, partial block)
  checkKGuard(vkEngine, 65, 89, 71)   # UNPADDED K: the guard must fire
  vkEngine.ingest(scaleUserVk)
  checkScaleUser(vkEngine, 65, 89, 71)

  # ── Metal twin runs: the same kernels, same reference (cross-backend) ───
  var mslEngine = bkMetal.init()
  mslEngine.ingest(gemmVkMsl)
  checkPlain(mslEngine, 37, 55, 48)
  checkPlain(mslEngine, 65, 89, 71)
  checkKGuard(mslEngine, 65, 89, 71)
  checkScaleUser(mslEngine, 65, 89, 71)

  echo "manual_tile_ragged_gemm_epi_vulkan: Vulkan gemm_with_epilogue (plain strided + user epilogue, ragged-edge shapes) PASS on MoltenVK + Metal twins"

when isMainModule:
  runTest()

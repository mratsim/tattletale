## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run: nim cpp -r --hints:off --warnings:off \
##   --outdir:build/tests/test_tile_ops_vulkan \
##   --nimcache:nimcache/tests/test_tile_ops_vulkan \
##   workspace/ceramic/tests/test_tile_ops_vulkan.nim

import std/[os, osproc, strformat, strutils, tempfiles, math]
import workspace/crucible
import tile_test_utils
import ../src/kernels/k_tile_attn

# ═════════════════════════════════════════════════════════════════════════
#  Device fns
# ═════════════════════════════════════════════════════════════════════════

proc attnRaw(o: ptr UncheckedArray[float32],
             q, k, v: ptr UncheckedArray[float16],
             H, N: int32, D: static int) {.device.} =
  ## Raw-pointer fp16-in/f32-out attention, one q-row per lane (no GlView tile layer).
  ## Kernel params are named out/Q/K/V and device params o/q/k/v on purpose:
  ## the Vulkan rename-based SSBO binding must map o→out, q→Q, k→K, v→V.
  let b = int32(threadgroup_position_in_grid.z)
  let h = int32(threadgroup_position_in_grid.y)
  let n = int32(thread_index_in_threadgroup)
  if n < N:
    let rowBase = ((b * H + h) * N + n) * int32(D)
    let qMul = rsqrt(float32(D)) * 1.44269504089'f32
    var maxS = -3.402823466e38'f32
    for m in 0'i32 ..< N:
      let kRow = ((b * H + h) * N + m) * int32(D)
      var acc = 0.0'f32
      for d in 0'i32 ..< int32(D):
        acc += float32(q[rowBase + d]) * float32(k[kRow + d])
      let s = acc * qMul
      if s > maxS: maxS = s
    var sumE = 0.0'f32
    for m in 0'i32 ..< N:
      let kRow = ((b * H + h) * N + m) * int32(D)
      var acc = 0.0'f32
      for d in 0'i32 ..< int32(D):
        acc += float32(q[rowBase + d]) * float32(k[kRow + d])
      let p = exp2(acc * qMul - maxS)
      sumE += p
      let vRow = ((b * H + h) * N + m) * int32(D)
      for d in 0'i32 ..< int32(D):
        o[rowBase + d] += p * float32(v[vRow + d])
    for d in 0'i32 ..< int32(D):
      o[rowBase + d] = o[rowBase + d] / sumE

# ═════════════════════════════════════════════════════════════════════════
#  The thin {.global.} launchers, one per head dim (D = 64, 128).
#  First param = the output buffer o: engine.run binds the separate outF
#  argument to it and the args tuple to the rest.
#  The Vulkan workgroup is baked into the GLSL (32 lanes, the universal
#  atom's thread count) and must match engine.run's blk exactly.
# ═════════════════════════════════════════════════════════════════════════

const attnVk64 = vulkan:
  proc attnD64(`out`: ptr UncheckedArray[float32],
               Q, K, V: ptr UncheckedArray[float16], H, N: int32) {.global, workgroup: (32, 1, 1).} =
    attnRaw(`out`, Q, K, V, H, N, 64)

const attnVk128 = vulkan:
  proc attnD128(`out`: ptr UncheckedArray[float32],
                Q, K, V: ptr UncheckedArray[float16], H, N: int32) {.global, workgroup: (32, 1, 1).} =
    attnRaw(`out`, Q, K, V, H, N, 128)

# Two kernels sharing the SAME attnRaw-64 instantiation with identical buffer names:
# the device fn must be emitted once and both kernels must be valid GLSL. Scalar
# params + 2 kernels means engine.ingest would quit, so this module is
# glslangCheck-only (never ingested).
const attnVkShared = vulkan:
  proc attnD64a(`out`: ptr UncheckedArray[float32],
                Q, K, V: ptr UncheckedArray[float16], H, N: int32) {.global, workgroup: (32, 1, 1).} =
    attnRaw(`out`, Q, K, V, H, N, 64)
  proc attnD64b(`out`: ptr UncheckedArray[float32],
                Q, K, V: ptr UncheckedArray[float16], H, N: int32) {.global, workgroup: (32, 1, 1).} =
    attnRaw(`out`, Q, K, V, H, N, 64)

# Separate module: the uint16 SSBO (tileKMax) must not share bindings
# with the attn launchers' float SSBOs (binding dedup is positional in vulkan_lang).
const kMaxVk = vulkan:
  proc kMaxKernel(o: ptr UncheckedArray[uint32],
                  lengths: ptr UncheckedArray[uint16], mTile: uint32) {.global, workgroup: (32, 1, 1).} =
    o[0] = tileKMax(lengths, mTile)

# Metal twin of attnD64 — pins the MSL emission (blast radius for tile_ops edits).
const attnMsl = metal:
  proc attnD64(o: ptr UncheckedArray[float32],
               q, k, v: ptr UncheckedArray[float16], H, N: int32) {.global.} =
    attn_fwd(q, k, v, o, H, N, 64)

# ═════════════════════════════════════════════════════════════════════════
#  Host references (no libtorch): fp16-exact values, naive fp32 SDPA, kMax
# ═════════════════════════════════════════════════════════════════════════

func f16Exact(b, h, n, d, seed: int): float32 =
  ## Deterministic fp16-exact value in [-2, 2). Every value is an fp16 grid
  ## point, so fp32ToFp16 is exact and fp16ToFp32 round-trips losslessly.
  float32((seed * b + 7 * h + 11 * n + 13 * d) mod 32) / 8.0'f32 - 2.0'f32

func naiveAttn(q, k, v: seq[float32]; B, H, N, D: int): seq[float32] =
  ## O = softmax(Q·Kᵀ/√D)·V in fp32, mirroring attnRaw's exact arithmetic
  ## (q scaled by rsqrt(D)·log2(e) once, then exp2 after the row max).
  result = newSeq[float32](B * H * N * D)
  # Host-side exp2 is pow(2, x): std/math has no exp2, and crucible's is a device-only stub.
  let qMul = 1.0'f32 / sqrt(float32(D)) * 1.44269504089'f32
  for b in 0 ..< B:
    for h in 0 ..< H:
      for n in 0 ..< N:
        var scores = newSeq[float32](N)
        for m in 0 ..< N:
          var acc = 0.0'f32
          for d in 0 ..< D:
            acc += q[((b * H + h) * N + n) * D + d] * k[((b * H + h) * N + m) * D + d]
          scores[m] = acc * qMul
        var maxS = -3.402823466e38'f32
        for m in 0 ..< N:
          if scores[m] > maxS: maxS = scores[m]
        var sumE = 0.0'f32
        for m in 0 ..< N:
          scores[m] = pow(2.0'f32, scores[m] - maxS)
          sumE += scores[m]
        for d in 0 ..< D:
          var acc = 0.0'f32
          for m in 0 ..< N:
            acc += scores[m] * v[((b * H + h) * N + m) * D + d]
          result[((b * H + h) * N + n) * D + d] = acc / sumE

func tileKMaxRef(lengths: seq[uint16]; mTile: uint32): uint32 =
  ## ceil(max over the tile's 32 rows of Lengths / 16), the host twin of tileKMax.
  var m = 0'u32
  for i in 0 ..< 32:
    let l = uint32(lengths[mTile * 32'u32 + uint32(i)])
    if l > m: m = l
  result = (m + 15'u32) div 16'u32

# ═════════════════════════════════════════════════════════════════════════
#  Harness
# ═════════════════════════════════════════════════════════════════════════

proc glslangCheck(src: string; renameFrom: string; label: string) =
  ## Compile the shader with glslangValidator (-V, vulkan1.1 target) after
  ## renaming the target kernel to `main`, the same per-kernel recompilation
  ## the Vulkan engine does. Compile-only, -o /dev/null. Requires
  ## glslangValidator on PATH: a missing tool or any rejection fails the test.
  var s = src.replace("void " & renameFrom & "()", "void main()")
  let (tmpFile, tmpPath) = createTempFile("vk_tile_ops", ".comp")
  defer: tmpFile.close()
  tmpFile.write(s)
  tmpFile.flushFile()
  let (outp, exitCode) = execCmdEx(
    "glslangValidator -V --target-env vulkan1.1 " & quoteShell(tmpPath) & " -o /dev/null")
  doAssert exitCode == 0,
    "glslangValidator rejected " & label & ":\n" & outp & "\n--- shader ---\n" & s

proc runTest() =
  # ── emission part (Vulkan) ────────────────────────────────────────────────
  echo "── attnVk64 GLSL (" & $attnVk64.len & " chars) ──"
  echo attnVk64
  doAssert "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable" in attnVk64,
    "missing fp16 arithmetic extension:\n" & attnVk64
  doAssert "#extension GL_EXT_shader_16bit_storage : enable" in attnVk64,
    "missing 16bit_storage extension (fp16 SSBOs):\n" & attnVk64
  doAssert "float16_t" in attnVk64, "missing float16_t (fp16 SSBO / conversions):\n" & attnVk64
  doAssert "layout(local_size_x = 32" in attnVk64,
    "missing baked workgroup size:\n" & attnVk64
  doAssert "exp2(" in attnVk64 and "inversesqrt(" in attnVk64,
    "missing exp2/inversesqrt spellings (rsqrt is MSL's name; GLSL is inversesqrt):\n" & attnVk64
  doAssert "gl_LocalInvocationIndex" in attnVk64,
    "thread_index_in_threadgroup must emit gl_LocalInvocationIndex:\n" & attnVk64
  doAssert "-3.402823466e+38f" in attnVk64,
    "missing neg_infty fp32 literal:\n" & attnVk64
  # The device-fn ptr params must be renamed to the kernel's buffer names and
  # dropped from the signature. A leftover ptr param would emit `float16_t* o`
  # or `float* o` in the device signature.
  doAssert "void attnD64()" in attnVk64, "missing kernel entry point:\n" & attnVk64
  doAssert "float out_vk[]" in attnVk64,
    "out (GLSL-reserved) must become the out_vk SSBO member:\n" & attnVk64
  doAssert "attnRaw" in attnVk64, "missing device fn attnRaw:\n" & attnVk64
  doAssert "float16_t*" notin attnVk64,
    "device fn still carries a float16_t* param:\n" & attnVk64
  doAssert "float*" notin attnVk64,
    "device fn still carries a float* param:\n" & attnVk64

  echo "── attnVk128 GLSL (" & $attnVk128.len & " chars) ──"
  doAssert "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable" in attnVk128,
    "missing fp16 arithmetic extension:\n" & attnVk128
  doAssert "#extension GL_EXT_shader_16bit_storage : enable" in attnVk128,
    "missing 16bit_storage extension (fp16 SSBOs):\n" & attnVk128
  doAssert "float16_t" in attnVk128, "missing float16_t (fp16 SSBO / conversions):\n" & attnVk128
  doAssert "layout(local_size_x = 32" in attnVk128,
    "missing baked workgroup size:\n" & attnVk128

  echo "── attnVkShared GLSL (" & $attnVkShared.len & " chars) ──"
  doAssert attnVkShared.count("int H;") == 1,
    "push-constant H must be deduped across kernels:\n" & attnVkShared
  doAssert attnVkShared.count("int N;") == 1,
    "push-constant N must be deduped across kernels:\n" & attnVkShared
  doAssert "void attnD64a()" in attnVkShared, "missing attnD64a:\n" & attnVkShared
  doAssert "void attnD64b()" in attnVkShared, "missing attnD64b:\n" & attnVkShared

  echo "── kMaxVk GLSL (" & $kMaxVk.len & " chars) ──"
  echo kMaxVk
  doAssert "uint16_t lengths[]" in kMaxVk,
    "tileKMax must emit a uint16_t SSBO:\n" & kMaxVk
  doAssert "#extension GL_EXT_shader_16bit_storage : enable" in kMaxVk,
    "missing 16bit_storage extension (uint16 SSBO):\n" & kMaxVk
  doAssert "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable" in kMaxVk,
    "missing explicit_arithmetic_types_int16 extension (uint16 arithmetic):\n" & kMaxVk
  doAssert "subgroupShuffleDown(" in kMaxVk, "missing subgroupShuffleDown:\n" & kMaxVk
  doAssert "subgroupShuffle(" in kMaxVk, "missing subgroupShuffle:\n" & kMaxVk
  # GPU-B-001: kMax's tileKMax max-reduction is a subgroup shuffle tree —
  # its lane id must be the subgroup lane (gl_SubgroupInvocationID), fixed
  # to 32 lanes by the fail-loudly guard. The old gl_LocalInvocationIndex
  # spelling was the workgroup lane, silently wrong on devices whose
  # workgroup spans several subgroups.
  doAssert "if (gl_SubgroupSize < 32u) { return; }" in kMaxVk,
    "GPU-B-001: missing gl_SubgroupSize<32 fail-loudly guard:\n" & kMaxVk
  doAssert "gl_SubgroupInvocationID" in kMaxVk,
    "GPU-B-001: shuffle lane id must be gl_SubgroupInvocationID:\n" & kMaxVk
  doAssert "gl_LocalInvocationIndex" notin kMaxVk,
    "GPU-B-001: shuffle-reachable lane id must be fully rewritten:\n" & kMaxVk

  # ── emission part (Metal, blast radius) ─────────────────────────────────
  doAssert "simd_shuffle_down(" in attnMsl, "MSL subgroup shuffle spelling drifted:\n" & attnMsl
  doAssert "simd_shuffle(" in attnMsl, "MSL subgroup shuffle spelling drifted:\n" & attnMsl
  doAssert "thread_index_in_threadgroup" in attnMsl,
    "MSL thread_index spelling drifted:\n" & attnMsl
  doAssert "half(" in attnMsl, "MSL fp16 conversion spelling drifted:\n" & attnMsl
  doAssert "exp2(" in attnMsl, "MSL exp2 spelling drifted:\n" & attnMsl

  # ── glslangValidator (per-kernel, like the engine) ───────────────────────
  glslangCheck(attnVk64, "attnD64", "attnD64")
  glslangCheck(attnVk128, "attnD128", "attnD128")
  glslangCheck(attnVkShared, "attnD64a", "attnD64a")
  glslangCheck(attnVkShared, "attnD64b", "attnD64b")
  glslangCheck(kMaxVk, "kMaxKernel", "kMaxKernel")

  # ── on-device value checks (MoltenVK on Apple Silicon) ───────────────────
  var engine = bkVulkan.init()
  engine.ingest(attnVk64)

  const B = 1
  const H = 1
  const N = 16            # q-rows per head, one lane per q-row, 32 lanes
  let n = B * H * N * 64
  var q64 = newSeq[uint16](n)
  var k64 = newSeq[uint16](n)
  var v64 = newSeq[uint16](n)
  var qf = newSeq[float32](n)
  var kf = newSeq[float32](n)
  var vf = newSeq[float32](n)
  for i in 0 ..< n:
    let b = i div (H * N * 64)
    let h = (i div (N * 64)) mod H
    let s = (i div 64) mod N
    let d = i mod 64
    qf[i] = f16Exact(b, h, s, d, 1)
    kf[i] = f16Exact(b, h, s, d, 4)
    vf[i] = f16Exact(b, h, s, d, 8)
    q64[i] = fp32ToFp16(qf[i])
    k64[i] = fp32ToFp16(kf[i])
    v64[i] = fp32ToFp16(vf[i])
  var outF = newSeq[float32](n)
  engine.run << (grid: (1, H, B), blk: (32, 1)) >> (
    "attnD64", outF, (q64, k64, v64, int32(H), int32(N)))
  let ref64 = naiveAttn(qf, kf, vf, B, H, N, 64)
  var worst = 0.0'f32
  for i in 0 ..< n:
    let d = abs(outF[i] - ref64[i])
    if d > worst: worst = d
    doAssert d <= 1e-2'f32, &"attnD64[{i}]: got {outF[i]}, want {ref64[i]}"
  echo &"  OK — attnD64 run vs naive fp32 SDPA (worst |Δ| = {worst})"

  # D = 128
  engine.ingest(attnVk128)
  let n128 = B * H * N * 128
  var q128 = newSeq[uint16](n128)
  var k128 = newSeq[uint16](n128)
  var v128 = newSeq[uint16](n128)
  var qf128 = newSeq[float32](n128)
  var kf128 = newSeq[float32](n128)
  var vf128 = newSeq[float32](n128)
  for i in 0 ..< n128:
    let b = i div (H * N * 128)
    let h = (i div (N * 128)) mod H
    let s = (i div 128) mod N
    let d = i mod 128
    qf128[i] = f16Exact(b, h, s, d, 11)
    kf128[i] = f16Exact(b, h, s, d, 14)
    vf128[i] = f16Exact(b, h, s, d, 18)
    q128[i] = fp32ToFp16(qf128[i])
    k128[i] = fp32ToFp16(kf128[i])
    v128[i] = fp32ToFp16(vf128[i])
  var outF128 = newSeq[float32](n128)
  engine.run << (grid: (1, H, B), blk: (32, 1)) >> (
    "attnD128", outF128, (q128, k128, v128, int32(H), int32(N)))
  let ref128 = naiveAttn(qf128, kf128, vf128, B, H, N, 128)
  worst = 0.0'f32
  for i in 0 ..< n128:
    let d = abs(outF128[i] - ref128[i])
    if d > worst: worst = d
    doAssert d <= 1e-2'f32, &"attnD128[{i}]: got {outF128[i]}, want {ref128[i]}"
  echo &"  OK — attnD128 run vs naive fp32 SDPA (worst |Δ| = {worst})"

  # tileKMax: uint16 SSBO path
  var kEngine = bkVulkan.init()
  kEngine.ingest(kMaxVk)
  var lengths = newSeq[uint16](64)
  for i in 0 ..< 64:
    lengths[i] = uint16((i * 7) mod 40)
  for mTile in [0'u32, 1'u32]:
    var res: array[1, uint32]
    kEngine.run("kMaxKernel", res, (lengths, mTile))
    let want = tileKMaxRef(lengths, mTile)
    doAssert res[0] == want, &"kMax mTile={mTile}: got {res[0]}, want {want}"
    echo &"  OK — kMaxKernel run vs host reference (mTile={mTile}, ceil(max/16)={want})"

when isMainModule:
  runTest()

## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## Vulkan: device-fn ptr-param clone path (RID COV-A-001 + BUG-A-004 +
## BUG-B-002).
##
## Pass 3 (`vulkanBindDeviceFnPtrParams`) clones a device fn once per
## disagreeing ptr-arg tuple. That branch was entirely untested (COV-A-001);
## this file exercises it with one multi-kernel module (pointer-only, so the
## engine ingests it) and pins:
##   (a) both `_vk0`/`_vk1` clone entry points exist and the original fn is
##       gone (COV-A-001);
##   (b) ptr-arith call-site args substituted into clone bodies are folded
##       to SSBO element indexes — the old post-fold iterated the pre-clone
##       `reachable` snapshot and skipped clones, hitting a raw-pointer
##       raise at codegen (BUG-A-004);
##   (c) same-named idents at different SSBO positions group by symbol
##       identity, never display name — name-keyed grouping merged distinct
##       buffers into one clone, silently reading/writing the wrong SSBO
##       (BUG-B-002).
##
## Run:
##   nim cpp -r --hints:off --warnings:off \
##     --outdir:build/tests/vulkan --nimcache:nimcache/tests/vulkan \
##     workspace/crucible/tests/codegen/vulkan/test_vulkan_clone_path.nim

import std/[strutils, unittest]
import workspace/crucible
import workspace/ceramic/src/ptr_arithmetic

# ── (a)+(b): clone branch + clone-body ptr-index folding ─────────────────

const cloneArithVk = vulkan:
  proc scaleOff(src: ptr UncheckedArray[uint32], s: uint32,
                dst: ptr UncheckedArray[uint32]) {.device.} =
    dst[0] = src[0] * s

  proc kernelA(dst: ptr UncheckedArray[uint32],
               src: ptr UncheckedArray[uint32]) {.global.} =
    scaleOff(src +% 1'u32, 2'u32, dst)

  proc kernelB(dst: ptr UncheckedArray[uint32],
               src: ptr UncheckedArray[uint32]) {.global.} =
    scaleOff(src +% 2'u32, 3'u32, dst)

# ── (c): BUG-B-002 — same-named idents at swapped SSBO positions ─────────

const nameKeyVk = vulkan:
  proc copyFirst(src: ptr UncheckedArray[uint32],
                 dst: ptr UncheckedArray[uint32]) {.device.} =
    dst[0] = src[0]

  proc kernelA(res: ptr UncheckedArray[uint32],
               buf: ptr UncheckedArray[uint32]) {.global.} =
    copyFirst(buf, res)     # src = buf (binding 1), dst = res (binding 0)

  proc kernelB(buf: ptr UncheckedArray[uint32],
               res: ptr UncheckedArray[uint32]) {.global.} =
    copyFirst(buf, res)     # src = buf (binding 0), dst = res (binding 1)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Vulkan - device-fn ptr-param clone path":
    test "COV-A-001 + BUG-A-004: disagreeing ptr args clone per group, clones folded":
      # kernelA passes `src +% 1`, kernelB passes `src +% 2` — two disagreeing
      # ptr-arg tuples → scaleOff_vk0/_vk1. The substituted `src +% N` chains
      # must fold to SSBO indexes inside the clones (BUG-A-004) — an unfolded
      # cast[ptr] would raise at codegen.
      check "scaleOff_vk0" in cloneArithVk
      check "scaleOff_vk1" in cloneArithVk
      check "void scaleOff(" notin cloneArithVk
      check "src[(int(1U) + 0)]" in cloneArithVk
      check "src[(int(2U) + 0)]" in cloneArithVk
      var engine = bkVulkan.init()
      engine.ingest(cloneArithVk)
      var resA: array[1, uint32]
      let srcA = [10'u32, 20, 30]
      engine.run << (grid: (1, 1), blk: (1, 1)) >> ("kernelA", resA, (srcA,))
      # kernelA: dst[0] = src[1] * 2
      check resA[0] == 40'u32
      var resB: array[1, uint32]
      let srcB = [10'u32, 20, 30]
      engine.run << (grid: (1, 1), blk: (1, 1)) >> ("kernelB", resB, (srcB,))
      # kernelB: dst[0] = src[2] * 3
      check resB[0] == 90'u32

    test "BUG-B-002: same-named idents at swapped positions stay separate groups":
      # Both kernels call copyFirst(buf, res) with identically-named args,
      # but kernelB's `buf` is at binding 0 (canonical `res`) while
      # kernelA's `buf` is at binding 1. Name-keyed grouping merged them
      # into ONE clone bound to kernelA's buffers; iSym-keyed grouping
      # produces two clones and each kernel reads/writes its own buffer.
      check "copyFirst_vk0" in nameKeyVk
      check "copyFirst_vk1" in nameKeyVk
      check "void copyFirst(" notin nameKeyVk
      var engine = bkVulkan.init()
      engine.ingest(nameKeyVk)
      var resA: array[1, uint32]
      let bufA = [10'u32]
      engine.run << (grid: (1, 1), blk: (1, 1)) >> ("kernelA", resA, (bufA,))
      check resA[0] == 10'u32
      # kernelB: src = binding 0 (resB), dst = binding 1 (bufB): the kernel
      # writes binding 1 only, so the returned binding-0 buffer is untouched.
      # The old merged clone wrote binding 0 from binding 1 instead — resB
      # would become bufB[0] = 99. The fix keeps resB[0] == 10.
      var resB = [10'u32]
      let bufB = [99'u32]
      engine.run << (grid: (1, 1), blk: (1, 1)) >> ("kernelB", resB, (bufB,))
      check resB[0] == 10'u32

when isMainModule:
  runTest()

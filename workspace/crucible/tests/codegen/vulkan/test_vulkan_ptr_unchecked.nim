## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## Vulkan: `ptr UncheckedArray` codegen, independent of ceramic.
##
## Three single-kernel modules exercise the Vulkan-only legalization passes
## (passes_legalization_vulkan.nim) on the raw-pointer shapes the ceramic
## tile layer uses, without importing ceramic:
##   (a) add3 — a kernel with 3 differently-named ptr params locks in
##       position-based SSBO binding (binding N = param N);
##   (b) countUp — a device fn with a `var uint32` param, called twice,
##       locks in vulkanVarParamsToValue (GLSL has no references; the call
##       becomes `acc = bump(acc)` returning the mutated value);
##   (c) pairKernel — a value struct carrying a ptr field passed to a
##       device fn locks in vulkanFlattenStructPtrValues + per-call-site
##       binding (GLSL structs cannot hold pointer members, so the ptr
##       field becomes an SSBO expression and the struct is flattened).
##
## Run:
##   nim cpp -r --hints:off --warnings:off \
##     --outdir:build/tests/vulkan --nimcache:nimcache/tests/vulkan \
##     workspace/crucible/tests/codegen/vulkan/test_vulkan_ptr_unchecked.nim

import std/[strutils, unittest]
import workspace/crucible

type
  PtrPair = object
    data: ptr UncheckedArray[uint32]
    scale: uint32

const add3Vk = vulkan:
  proc add3(
      dst: ptr UncheckedArray[uint32],
      left: ptr UncheckedArray[uint32],
      right: ptr UncheckedArray[uint32]) {.global.} =
    dst[0] = left[0] + right[0]

const countUpVk = vulkan:
  proc bump(x: var uint32) {.device.} =
    x = x + 1'u32

  proc countUp(acc: ptr UncheckedArray[uint32]) {.global.} =
    var v: uint32 = 40
    bump(v)
    bump(v)
    acc[0] = v

const pairVk = vulkan:
  proc scalePair(p: PtrPair, dst: ptr UncheckedArray[uint32]) {.device.} =
    dst[0] = p.data[0] * p.scale

  proc pairKernel(
      dst: ptr UncheckedArray[uint32],
      src: ptr UncheckedArray[uint32]) {.global.} =
    let pair = PtrPair(data: src, scale: 3'u32)
    scalePair(pair, dst)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Vulkan - ptr UncheckedArray codegen (crucible-only)":
    test "kernel with 3 ptr params binds SSBOs by position":
      # 3 differently-named ptr params → 3 SSBOs, binding N = param N
      # (`dst` is binding 0, `right` is binding 2).
      check add3Vk.count("layout(set = 0, binding =") == 3
      check "layout(set = 0, binding = 0) buffer Buf0 { uint dst[]; };" in add3Vk
      check "layout(set = 0, binding = 2) buffer Buf2 { uint right[]; };" in add3Vk
      var engine = bkVulkan.init()
      engine.ingest(add3Vk)
      var res: array[1, uint32]
      let left = [10'u32]
      let right = [32'u32]
      engine.run << (grid: (1, 1), blk: (1, 1)) >> ("add3", res, (left, right))
      check res[0] == 42'u32

    test "device fn var uint32 param lowers to value + return":
      # bump(x: var uint32) becomes value-param + return-by-value
      # (GLSL has no references): `bump(v)` calls are rewritten to
      # `v = bump(v);`.
      check "uint bump(uint x);" in countUpVk
      check "v = bump(v);" in countUpVk
      var engine = bkVulkan.init()
      engine.ingest(countUpVk)
      var res: array[1, uint32]
      engine.run << (grid: (1, 1), blk: (1, 1)) >> ("countUp", res, ())
      check res[0] == 42'u32

    test "value struct with ptr field passed to a device fn":
      # PtrPair is flattened (ptr field → SSBO expression, scale → leaf
      # local) and the device fn's ptr params bind to the kernel SSBOs,
      # so no `struct PtrPair` (GLSL structs cannot hold pointer members)
      # is emitted: scalePair takes only the value leaf (`p_scale`) and
      # reads the SSBO names `dst`/`src` directly.
      check "struct PtrPair" notin pairVk
      check "dst[0] = (src[0] * p_scale);" in pairVk
      check "p__scale" notin pairVk
      var engine = bkVulkan.init()
      engine.ingest(pairVk)
      var res: array[1, uint32]
      let src = [14'u32]
      engine.run << (grid: (1, 1), blk: (1, 1)) >> ("pairKernel", res, (src,))
      check res[0] == 42'u32

when isMainModule:
  runTest()

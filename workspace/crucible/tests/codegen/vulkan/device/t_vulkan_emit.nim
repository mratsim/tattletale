## Vulkan {.emit.} raw GLSL injection: argument forms, payload byte-exactness, non-emit pragma discard, result exclusion.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan/device/t_vulkan_emit.nim

import std/[strutils, unittest]
import workspace/crucible

# Raw emit strings bypass the compiler: on Vulkan a pointer param is a buffer
# reference and glslang only accepts `ref.x[i]`, direct `ref[i]` is rejected.
# DSL code needs none of this, the compiler inserts the deref.

# ── single string literal, injected as raw GLSL ─────────────────────────────

const t1Vk = vulkan:
  proc t1Kernel(C: ptr UncheckedArray[uint32];
                A: ptr UncheckedArray[uint32];
                B: ptr UncheckedArray[uint32]) {.global, workgroup: (1, 1, 1).} =
    {.emit: "C.x[0] = A.x[0] + B.x[0];".}

# ── comma form, one interpolated ident (the `A` param) ──────────────────────

const t2Vk = vulkan:
  proc t2Kernel(C: ptr UncheckedArray[uint32];
                A: ptr UncheckedArray[uint32]) {.global, workgroup: (1, 1, 1).} =
    {.emit: ["C.x[0] = ", A, ".x[0] + 1;"].}

# ── comma form, interleaved ident + call + arithmetic ───────────────────────

const t3Vk = vulkan:
  proc doubleIt(v: uint32): uint32 =
    v * 2

  proc t3Kernel(C: ptr UncheckedArray[uint32];
                A: ptr UncheckedArray[uint32]) {.global, workgroup: (1, 1, 1).} =
    let off: uint32 = 2
    {.emit: ["C.x[0] = ", doubleIt(A[0]), " + ", off, ";"].}

# ── backtick form on a local variable ───────────────────────────────────────

const t4Vk = vulkan:
  proc t4Kernel(C: ptr UncheckedArray[uint32];
                A: ptr UncheckedArray[uint32]) {.global, workgroup: (1, 1, 1).} =
    let tmp: uint32 = A[0] + 1
    {.emit: "C.x[0] = `tmp` + 1;".}

# ── triple-quoted multiline string ──────────────────────────────────────────

const t5Vk = vulkan:
  proc t5Kernel(C: ptr UncheckedArray[uint32];
                A: ptr UncheckedArray[uint32];
                B: ptr UncheckedArray[uint32]) {.global, workgroup: (1, 1, 1).} =
    {.emit: """C.x[0] = A.x[0] + B.x[0];
C.x[1] = A.x[1] + B.x[1];""".}

# ── not silently dropped — the emit must override the DSL write ─────────────

const t6Vk = vulkan:
  proc t6Kernel(C: ptr UncheckedArray[uint32]) {.global, workgroup: (1, 1, 1).} =
    C[0] = 0
    {.emit: "C.x[0] = 42;".}

# ── backtick form on a kernel parameter ─────────────────────────────────────

const t7Vk = vulkan:
  proc t7Kernel(C: ptr UncheckedArray[uint32];
                x: ptr UncheckedArray[uint32]) {.global, workgroup: (1, 1, 1).} =
    {.emit: "C.x[0] = `x`.x[0] + 1;".}

# ── comma form on a kernel parameter ────────────────────────────────────────

const t8Vk = vulkan:
  proc t8Kernel(C: ptr UncheckedArray[uint32];
                x: ptr UncheckedArray[uint32]) {.global, workgroup: (1, 1, 1).} =
    {.emit: ["C.x[0] = ", x, ".x[0] + 1;"].}

# ── insertResult exclusion — a non-void device fn with only a trailing emit
#    must not become `result = <text>` ─────────────────────────────────────────

const t10Vk = vulkan:
  proc emit42(): uint32 =
    {.emit: "return 42;".}

  proc t10Kernel(C: ptr UncheckedArray[uint32]) {.global, workgroup: (1, 1, 1).} =
    C[0] = emit42()

# ── call-form emit `{.emit("...").}` is not silently dropped ────────────────

const t12Vk = vulkan:
  proc t12Kernel(C: ptr UncheckedArray[uint32]) {.global, workgroup: (1, 1, 1).} =
    C[0] = 0
    {.emit("C.x[0] = 42;").}

# ── a #define-style directive must pass through byte-exact. A `;` appended by the printer
#    would corrupt the macro (used below) ──────────────────────────────────────

const t14Vk = vulkan:
  proc t14Kernel(C: ptr UncheckedArray[uint32]) {.global, workgroup: (1, 1, 1).} =
    {.emit: "#define VULKAN_EMIT_TILE 32".}
    {.emit: "C.x[0] = VULKAN_EMIT_TILE + 1;".}

# ── duplicate emit items in one pragma — every item renders, in order ────────

const t15Vk = vulkan:
  proc t15Kernel(C: ptr UncheckedArray[uint32]) {.global, workgroup: (1, 1, 1).} =
    C[0] = 0
    {.emit: "C.x[0] = 1;", emit: "C.x[0] = 2;".}

# ── non-emit statement pragma inside a kernel stays a gpuDiscard ──────────────

const t16Vk = vulkan:
  proc t16Kernel(C: ptr UncheckedArray[uint32]) {.global, workgroup: (1, 1, 1).} =
    {.warning: "this statement pragma must be discarded, not emitted".}
    C[0] = 42

# ── Host code ────────────────────────────────────────────────────────────────

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Vulkan - {.emit.} raw GLSL injection":

    test "single string literal emits the sum":
      var engine = bkVulkan.newEngine()
      engine.ingest(t1Vk)
      var a: array[1, uint32] = [40'u32]
      var b: array[1, uint32] = [2'u32]
      let aDev = engine.allocDevice(uint32, a.len)
      let bDev = engine.allocDevice(uint32, b.len)
      let oDev = engine.allocDevice(uint32, 1)
      engine.copyToDeviceFromHost(aDev, addr a[0], a.len)
      engine.copyToDeviceFromHost(bDev, addr b[0], b.len)
      engine.launchKernel("t1Kernel", oDev, (aDev, bDev))
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 42

    test "comma form with one interpolated ident":
      var engine = bkVulkan.newEngine()
      engine.ingest(t2Vk)
      var a: array[1, uint32] = [41'u32]
      let aDev = engine.allocDevice(uint32, a.len)
      let oDev = engine.allocDevice(uint32, 1)
      engine.copyToDeviceFromHost(aDev, addr a[0], a.len)
      engine.launchKernel("t2Kernel", oDev, (aDev,))
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 42

    test "comma form with ident + call + arithmetic":
      var engine = bkVulkan.newEngine()
      engine.ingest(t3Vk)
      var a: array[1, uint32] = [20'u32]
      let aDev = engine.allocDevice(uint32, a.len)
      let oDev = engine.allocDevice(uint32, 1)
      engine.copyToDeviceFromHost(aDev, addr a[0], a.len)
      engine.launchKernel("t3Kernel", oDev, (aDev,))
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 42

    test "backtick form on a local variable":
      var engine = bkVulkan.newEngine()
      engine.ingest(t4Vk)
      var a: array[1, uint32] = [40'u32]
      let aDev = engine.allocDevice(uint32, a.len)
      let oDev = engine.allocDevice(uint32, 1)
      engine.copyToDeviceFromHost(aDev, addr a[0], a.len)
      engine.launchKernel("t4Kernel", oDev, (aDev,))
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 42

    test "triple-quoted multiline string":
      var engine = bkVulkan.newEngine()
      engine.ingest(t5Vk)
      var a: array[2, uint32] = [40'u32, 10'u32]
      var b: array[2, uint32] = [2'u32, 3'u32]
      let aDev = engine.allocDevice(uint32, a.len)
      let bDev = engine.allocDevice(uint32, b.len)
      let oDev = engine.allocDevice(uint32, 2)
      engine.copyToDeviceFromHost(aDev, addr a[0], a.len)
      engine.copyToDeviceFromHost(bDev, addr b[0], b.len)
      engine.launchKernel("t5Kernel", oDev, (aDev, bDev))
      var res: array[2, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 2)
      check res[0] == 42
      check res[1] == 13

    test "emit is not silently dropped (must override the DSL write)":
      var engine = bkVulkan.newEngine()
      engine.ingest(t6Vk)
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("t6Kernel", oDev, ())
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 42

    test "backtick form on a kernel parameter":
      var engine = bkVulkan.newEngine()
      engine.ingest(t7Vk)
      var x: array[1, uint32] = [41'u32]
      let xDev = engine.allocDevice(uint32, x.len)
      let oDev = engine.allocDevice(uint32, 1)
      engine.copyToDeviceFromHost(xDev, addr x[0], x.len)
      engine.launchKernel("t7Kernel", oDev, (xDev,))
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 42

    test "comma form on a kernel parameter":
      var engine = bkVulkan.newEngine()
      engine.ingest(t8Vk)
      var x: array[1, uint32] = [41'u32]
      let xDev = engine.allocDevice(uint32, x.len)
      let oDev = engine.allocDevice(uint32, 1)
      engine.copyToDeviceFromHost(xDev, addr x[0], x.len)
      engine.launchKernel("t8Kernel", oDev, (xDev,))
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 42

    test "trailing emit in a non-void device fn survives insertResult":
      var engine = bkVulkan.newEngine()
      engine.ingest(t10Vk)
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("t10Kernel", oDev, ())
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 42

    test "call-form emit is not silently dropped (must override the DSL write)":
      var engine = bkVulkan.newEngine()
      engine.ingest(t12Vk)
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("t12Kernel", oDev, ())
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 42

    test "#define-style payload is byte-exact (no `;` appended)":
      var engine = bkVulkan.newEngine()
      engine.ingest(t14Vk)
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("t14Kernel", oDev, ())
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 33

    test "every emit item in one pragma renders, in order, on separate lines":
      var engine = bkVulkan.newEngine()
      engine.ingest(t15Vk)
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("t15Kernel", oDev, ())
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 2

    test "non-emit statement pragma stays a gpuDiscard (kernel compiles and runs)":
      var engine = bkVulkan.newEngine()
      engine.ingest(t16Vk)
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("t16Kernel", oDev, ())
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 42

when isMainModule:
  runTest()

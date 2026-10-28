# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## HLSL printer fixtures: kernels through the registered `hlsl:` pipeline
## validate through dxc, the sole HLSL compiler, plus spirv-val over the SPIR-V
## that dxc emits, the Vulkan engine ingest path. Refusal fixtures print
## loudly through the printer on hand-built modules.
##
## Run:
##   nim c -r --outdir:build/tests tests/cirv2/printer/t_cirv2_printer_hlsl.nim

import std/[os, osproc, strutils, unittest]
import workspace/crucible
import workspace/crucible/src/codegen/backends/targets
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/backends/printer_hlsl

proc validate(name: string, src: string): bool =
  ## Runs dxc over the printed source, SPIR-V output then spirv-val, the two stages
  ## the Vulkan engine drives at ingest. Gives true when both are clean.
  let dxc = findExe("dxc")
  if dxc.len == 0:
    echo "dxc not found on PATH"
    return false
  let srcPath = getTempDir() / name & ".hlsl"
  let spvPath = getTempDir() / name & ".spv"
  writeFile(srcPath, src)
  let (outp, code) = execCmdEx(
    dxc & " -T cs_6_4 -E main -spirv -Fo " & spvPath & " " & srcPath)
  if code != 0:
    echo "dxc failed for " & name & ":\n" & outp & "\n" & src
    return false
  let (voutp, vcode) = execCmdEx("spirv-val " & spvPath)
  if vcode != 0:
    echo "spirv-val failed for " & name & ":\n" & voutp & "\n" & src
    return false
  result = true

# ═══════════════════════════════════════════════════════════════════════
#  Pipeline fixtures (public API: the registered `hlsl:` entry point)
# ═══════════════════════════════════════════════════════════════════════

const incSrc = hlsl(fvD3D12):
  proc kern_inc(dst, src: ptr UncheckedArray[float32], n: int32) {.global.} =
    var i: int32 = 0
    while i < n:
      dst[i] = src[i] + 1.0'f32
      i = i + 1

type Layout2 = object
  F0: int32
  F1: int32

const abiSrc = hlsl(fvD3D12):
  proc kern_abi(dst: ptr UncheckedArray[float32], ly: Layout2, alpha: float32) {.global.} =
    dst[0] = float32(ly.F0) + alpha

type View3 = object
  data: ptr UncheckedArray[float32]
  n: int32

const viewSrc = hlsl(fvD3D12):
  proc kern_view(dst: ptr UncheckedArray[float32], v: View3) {.global.} =
    dst[0] = v.data[0] + float32(v.n)

proc bump(x: var float32) =
  x = x + 1.0'f32

const byrefSrc = hlsl(fvD3D12):
  proc kern_byref(dst: ptr UncheckedArray[float32]) {.global.} =
    var a: float32 = 1.0'f32
    bump(a)
    dst[0] = a

const coordSrc = hlsl(fvD3D12):
  proc kern_coord(dst: ptr UncheckedArray[int32]) {.global.} =
    let gx = thread_position_in_grid.x
    let t = thread_index_in_threadgroup
    threadgroup_barrier()
    dst[0] = int32(gx + t)

const numwgSrc = hlsl(fvD3D12):
  proc kern_numwg(dst: ptr UncheckedArray[int32]) {.global.} =
    dst[0] = int32(threadgroups_per_grid.x + threadgroups_per_grid.y)

# ═══════════════════════════════════════════════════════════════════════
#  Refusal fixtures (hand-built modules through the printer)
# ═══════════════════════════════════════════════════════════════════════

proc buildTwoKernelsModule(): CirModule =
  let k1 = procD("kern_a", @[], tVoid(), @[retS()])
  let k2 = procD("kern_b", @[], tVoid(), @[retS()])
  k1.isKernel = true
  k2.isKernel = true
  moduleD("fixture_two", @[], @[k1, k2])

proc buildSmemModule(): CirModule =
  let kern = procD("kern_smem", @[], tVoid(), @[
    varD("scratch", tArray(8, tFloat(32)), nil, caThreadgroup),
    retS(),
  ])
  kern.isKernel = true
  moduleD("fixture_smem", @[], @[kern])

proc buildDeviceLocalModule(): CirModule =
  let kern = procD("kern_devlocal", @[], tVoid(), @[
    varD("d", tFloat(32), nil, caDevice),
    retS(),
  ])
  kern.isKernel = true
  moduleD("fixture_devlocal", @[], @[kern])

proc buildAsmModule(): CirModule =
  let kern = procD("kern_asm", @[], tVoid(), @[
    CirStmt(kind: sAsm, asmParts: @["mov x0, x1"], asmOperands: @[]),
    retS(),
  ])
  kern.isKernel = true
  moduleD("fixture_asm", @[], @[kern])

proc buildInt64Module(): CirModule =
  let kern = procD("kern_i64", @[
    param("n", tInt(64))], tVoid(), @[retS()])
  kern.isKernel = true
  moduleD("fixture_i64", @[], @[kern])

proc buildNumWorkgroupsModule(): CirModule =
  ## A threadgroups_per_grid read reaching the printer directly with no
  ## materialization pass run ahead of it
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, i32)
  let coordTy = tStruct("ivec3", fld("x", i32), fld("y", i32), fld("z", i32))
  let kern = procD("kern_numwg", @[
    param("dst", aptrD)], tVoid(), @[
    asgnS(patE(sym("dst", aptrD), intLit(0, i32)),
      dotE(sym("threadgroups_per_grid", coordTy, builtin = true,
        coordBuiltin = cbkThreadgroupsPerGrid), "x")),
    retS(),
  ])
  kern.isKernel = true
  moduleD("fixture_numwg", @[coordTy], @[kern])

proc buildCoordInDeviceFnModule(): CirModule =
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, i32)
  let coordTy = tStruct("ivec3", fld("x", i32), fld("y", i32), fld("z", i32))
  let fn = procD("dev_read", @[param("dst", aptrD)], tVoid(), @[
    asgnS(patE(sym("dst", aptrD), intLit(0, i32)),
      dotE(sym("thread_position_in_grid", coordTy, builtin = true,
        coordBuiltin = cbkThreadPositionInGrid), "x")),
    retS(),
  ])
  let kern = procD("kern_coordfn", @[], tVoid(), @[
    callS(sym("dev_read", tProcT(@[aptrD], tVoid())), @[sym("dst", aptrD)]),
    retS(),
  ])
  kern.isKernel = true
  moduleD("fixture_coordfn", @[coordTy], @[fn, kern])

proc buildRecursiveModule(): CirModule =
  let i32 = tInt(32)
  let fnTy = tProcT(@[i32], i32)
  let a = procD("dev_a", @[param("n", i32)], i32, @[
    retS(callE(sym("dev_b", fnTy, builtin = false), sym("n", i32))),
  ])
  let b = procD("dev_b", @[param("n", i32)], i32, @[
    retS(callE(sym("dev_a", fnTy, builtin = false), sym("n", i32))),
  ])
  let kern = procD("kern_rec", @[], tVoid(), @[
    callS(sym("dev_a", fnTy), @[intLit(1, i32)]),
    retS(),
  ])
  kern.isKernel = true
  moduleD("fixture_rec", @[], @[a, b, kern])

const
  twoErr = printHlslModule(buildTwoKernelsModule()).err
  smemErr = printHlslModule(buildSmemModule()).err
  devLocalErr = printHlslModule(buildDeviceLocalModule()).err
  asmErr = printHlslModule(buildAsmModule()).err
  int64Err = printHlslModule(buildInt64Module()).err
  numwgErr = printHlslModule(buildNumWorkgroupsModule()).err
  coordFnErr = printHlslModule(buildCoordInDeviceFnModule()).err
  recErr = printHlslModule(buildRecursiveModule()).err

suite "HLSL printer fixtures":

  test "read-write and read-only buffers on one global register ordinal":
    check incSrc.contains("RWStructuredBuffer<float> dst : register(u0, space0);")
    check incSrc.contains("StructuredBuffer<float> src : register(t1, space0);")
    check incSrc.contains("ConstantBuffer<Push> push : register(b2, space0);")
    check incSrc.contains("[numthreads(256, 1, 1)]")
    check incSrc.contains("void main()")
    check validate("hlsl_fixture_inc", incSrc)

  test "push block with a value struct and a scalar, reads through push":
    check abiSrc.contains("struct Push {")
    check abiSrc.contains("Layout2 ly;")
    check abiSrc.contains("float alpha;")
    check abiSrc.contains("push.ly.F0")
    check abiSrc.contains("push.alpha")
    check validate("hlsl_fixture_abi", abiSrc)

  test "pointer-holding struct param dissolves to its buffer and value leaves":
    check viewSrc.contains("StructuredBuffer<float> v_data : register(t1, space0);")
    check viewSrc.contains("v_data[0]")
    check viewSrc.contains("push.v_n")
    check (not viewSrc.contains("struct View3"))
    check validate("hlsl_fixture_view", viewSrc)

  test "var parameter binds inout, the call site passes the variable":
    check byrefSrc.contains("inout float x")
    # harvested device fns carry instantiation-hash names, the call site
    # passes the variable itself for the inout binding
    check byrefSrc.contains("bump_")
    check byrefSrc.contains("(a);")
    check validate("hlsl_fixture_byref", byrefSrc)

  test "coordinate builtins take semantics on main, barrier maps to the group fence":
    check coordSrc.contains("uint3 gid : SV_DispatchThreadID")
    check coordSrc.contains("uint gidx : SV_GroupIndex")
    check coordSrc.contains("int(gid.x)")
    check coordSrc.contains("GroupMemoryBarrierWithGroupSync();")
    check validate("hlsl_fixture_coord", coordSrc)

  test "threadgroups_per_grid materializes as a staged uniform input":
    check numwgSrc.contains("crucible_numworkgroups")
    check numwgSrc.contains("push.crucible_numworkgroups.x")
    check (not numwgSrc.contains("SV_NumWorkgroups"))
    check validate("hlsl_fixture_numwg", numwgSrc)

  test "loud refusal: two kernels in one module":
    check twoErr.contains("exactly one kernel")

  test "loud refusal: threadgroup-space local":
    check smemErr.contains("threadgroup space")

  test "loud refusal: device-space local":
    check devLocalErr.contains("device space")

  test "loud refusal: inline assembly":
    check asmErr.contains("inline assembly")

  test "loud refusal: 64-bit integer parameter":
    check int64Err.contains("64-bit")

  test "loud refusal: threadgroups_per_grid reaching the printer unmaterialized":
    check numwgErr.contains("threadgroups_per_grid")

  test "loud refusal: coordinate builtin in a device function":
    check coordFnErr.contains("device function")

  test "loud refusal: recursive device functions":
    check recErr.contains("recursive")

# ═══════════════════════════════════════════════════════════════════════
#  fvVulkan flavor fixtures, validate-only through dxc + spirv-val
# ═══════════════════════════════════════════════════════════════════════

const fvAxpySrc = hlsl(fvVulkan):
  proc kern_axpy(y, x: ptr UncheckedArray[float32], alpha: float32, n: int32) {.global.} =
    let i = thread_position_in_grid.x
    if i < n:
      y[i] = alpha * x[i] + y[i]

const fvNumwgSrc = hlsl(fvVulkan):
  proc kern_numwg(dst: ptr UncheckedArray[int32]) {.global.} =
    dst[0] = int32(threadgroups_per_grid.x)
    dst[1] = int32(threadgroups_per_grid.y)
    dst[2] = int32(threadgroups_per_grid.z)

type FvView = object
  data: ptr UncheckedArray[float32]
  n: int32

const fvViewSrc = hlsl(fvVulkan):
  proc kern_view(dst: ptr UncheckedArray[float32], v: FvView) {.global.} =
    dst[0] = v.data[0] + float32(v.n)

suite "HLSL printer fixtures, fvVulkan flavor":

  test "fvVulkan: BDA accesses over the pinned push-constant struct":
    check validate("hlsl_fixture_fv_axpy", fvAxpySrc)

  test "fvVulkan: the staged grid member rides the push-constant block":
    check validate("hlsl_fixture_fv_numwg", fvNumwgSrc)

  test "fvVulkan: a pointer-holding struct param dissolves to address and value leaves":
    check validate("hlsl_fixture_fv_view", fvViewSrc)

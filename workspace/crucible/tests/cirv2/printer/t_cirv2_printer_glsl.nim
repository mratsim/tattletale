# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## GLSL printer fixtures: hand-built CIRv2 modules print to GLSL 450 compute
## sources and validate through `glslangValidator`, the compiler the Vulkan
## engine drives at ingest.
##
## Run:
##   nim c -r --outdir:build/tests tests/cirv2/printer/t_cirv2_printer_glsl.nim

import std/[os, osproc, strutils, unittest]
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/backends/printer_glsl

proc validate(name: string, entry: string, src: string): bool =
  ## Runs glslangValidator over the printed source, true when clean.
  ## The flags match the Vulkan engine's ingest: entry point is the kernel name,
  ## source entry point is `main`, subgroup sources pass --target-env vulkan1.1.
  let srcPath = getTempDir() / name & ".comp"
  writeFile(srcPath, src)
  var args = "-V -e " & entry & " --source-entrypoint main"
  if "GL_KHR_shader_subgroup" in src:
    args.add " --target-env vulkan1.1"
  let cmd = "glslangValidator " & args & " " & srcPath
  let (outp, code) = execCmdEx(cmd)
  if code != 0:
    echo "glslangValidator failed for " & name & ":\n" & outp & "\n" & src
  result = code == 0

# ═══════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════

proc buildIncModule(): CirModule =
  ## kernel: for each i < n: dst[i] = src[i] + 1.0
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let kern = procD("kern_fixture_inc", @[
    param("dst", aptrD), param("src", aptrD), param("n", i32)], tVoid(), @[
    varD("i", i32, intLit(0, i32)),
    loopS(@[
      iteS(leE(sym("i", i32), subE(i32, sym("n", i32), intLit(1, i32))),
        @[], @[jmpS("exit")]),
      asgnS(patE(sym("dst", aptrD), sym("i", i32)),
        addE(f32, patE(sym("src", aptrD), sym("i", i32)),
          fltLit(1.0, f32))),
      asgnS(sym("i", i32), addE(i32, sym("i", i32), intLit(1, i32))),
    ]),
    labS("exit"),
    retS(),
  ])
  kern.isKernel = true
  moduleD("fixture_inc", @[], @[kern])

proc buildAbiModule(): CirModule =
  ## kernel: a device buffer, a by-value struct and a scalar in the push block,
  ## the body reads the struct fields and the buffer
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let layoutTy = tStruct("Layout_xy", fld("F0", i32), fld("F1", i32))
  let kern = procD("kern_fixture_abi", @[
    param("dst", aptrD), param("ly", layoutTy), param("alpha", f32)],
    tVoid(), @[
    varD("acc", f32, addE(f32, sym("alpha", f32),
      convE(f32, dotE(sym("ly", layoutTy), "F1")))),
    asgnS(patE(sym("dst", aptrD), intLit(0, i32)), sym("acc", f32)),
    retS(),
  ])
  kern.isKernel = true
  moduleD("fixture_abi", @[layoutTy], @[kern])

proc buildViewModule(): CirModule =
  ## kernel: a struct local with a device buffer member, written through the dot chain
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let viewTy = tStruct("View", fld("data", aptrD))
  let kern = procD("kern_fixture_view", @[
    param("dst", aptrD), param("a", f32)], tVoid(), @[
    varD("v", viewTy, oconstrE(viewTy, kv("data", sym("dst", aptrD)))),
    asgnS(patE(dotE(sym("v", viewTy), "data"), intLit(0, i32)),
      sym("a", f32)),
    retS(),
  ])
  kern.isKernel = true
  moduleD("fixture_view", @[viewTy], @[kern])

proc buildAliasModule(): CirModule =
  ## kernel: a ptr local bound to the address of a local struct, dereferenced on both sides of an assignment through the pointee's array member
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let arrTy = tArray(4, f32)
  let bufTy = tStruct("Buf", fld("data", arrTy))
  let ptrTy = tPtr(caThread, bufTy)
  let kern = procD("kern_fixture_alias", @[
    param("dst", aptrD)], tVoid(), @[
    varD("tmp", bufTy, oconstrE(bufTy, kv("data",
      aconstrE(arrTy, fltLit(1.0, f32), fltLit(2.0, f32),
        fltLit(3.0, f32), fltLit(4.0, f32))))),
    varD("p", ptrTy),
    asgnS(sym("p", ptrTy), addrE(ptrTy, sym("tmp", bufTy))),
    asgnS(atE(dotE(derefE(sym("p", ptrTy)), "data"), intLit(2, i32)),
      fltLit(9.0, f32)),
    asgnS(patE(sym("dst", aptrD), intLit(0, i32)),
      atE(dotE(derefE(sym("p", ptrTy)), "data"), intLit(2, i32))),
    retS(),
  ])
  kern.isKernel = true
  moduleD("fixture_alias", @[bufTy], @[kern])

proc buildCoordModule(): CirModule =
  ## kernel: coordinate builtin component reads and the flat thread index.
  ## A barrier and a threadgroup scratch local relocate to the module's program
  ## scope (the shape the GLSL lowering pass leaves)
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let coordTy = tStruct("ivec3_nim", fld("x", i32), fld("y", i32), fld("z", i32))
  let scratchTy = tArray(8, f32)
  let kern = procD("kern_fixture_coord", @[
    param("dst", aptrD)], tVoid(), @[
    varD("gx", i32, dotE(sym("thread_position_in_grid", coordTy, builtin = true,
      coordBuiltin = cbkThreadPositionInGrid), "x")),
    varD("t", i32, sym("thread_index_in_threadgroup", i32, builtin = true,
      coordBuiltin = cbkThreadIndexInThreadgroup)),
    callS(sym("threadgroup_barrier", tProcT(@[], tVoid()), builtin = true,
        synchroBuiltin = sbkThreadgroupBarrier), @[]),
    asgnS(patE(sym("dst", aptrD), intLit(0, i32)),
      addE(i32, sym("gx", i32), sym("t", i32))),
    retS(),
  ])
  kern.isKernel = true
  let m = moduleD("fixture_coord", @[coordTy], @[kern])
  m.globals = @[varD("scratch", scratchTy, nil, caThreadgroup)]
  m

proc buildReduceModule(): CirModule =
  ## kernel: a subgroup reduction, the KHR subgroup extensions
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let kern = procD("kern_fixture_reduce", @[
    param("dst", aptrD), param("v", f32)], tVoid(), @[
    asgnS(patE(sym("dst", aptrD), intLit(0, i32)),
      callE(sym("reduceAdd", tProcT(@[f32], f32), builtin = true,
        reductionBuiltin = rbkReduceAdd), sym("v", f32))),
    retS(),
  ])
  kern.isKernel = true
  moduleD("fixture_reduce", @[], @[kern])

proc buildTwoKernelsModule(): CirModule =
  let k1 = procD("kern_a", @[], tVoid(), @[retS()])
  let k2 = procD("kern_b", @[], tVoid(), @[retS()])
  k1.isKernel = true
  k2.isKernel = true
  moduleD("fixture_two", @[], @[k1, k2])

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

proc buildAliasNonLocalModule(): CirModule =
  ## A ptr local bound to the address of a buffer element: outside the alias set, GLSL cannot express the address
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let ptrTy = tPtr(caThread, f32)
  let kern = procD("kern_alias_bad", @[param("dst", aptrD)], tVoid(), @[
    varD("p", ptrTy),
    asgnS(sym("p", ptrTy), addrE(ptrTy, patE(sym("dst", aptrD), intLit(0, i32)))),
    retS(),
  ])
  kern.isKernel = true
  moduleD("fixture_alias_bad", @[], @[kern])

const
  incRes = printGlslModule(buildIncModule())
  abiRes = printGlslModule(buildAbiModule())
  viewRes = printGlslModule(buildViewModule())
  aliasRes = printGlslModule(buildAliasModule())
  coordRes = printGlslModule(buildCoordModule())
  reduceRes = printGlslModule(buildReduceModule())
  twoErr = printGlslModule(buildTwoKernelsModule()).err
  devLocalErr = printGlslModule(buildDeviceLocalModule()).err
  asmErr = printGlslModule(buildAsmModule()).err
  aliasBadErr = printGlslModule(buildAliasNonLocalModule()).err

suite "GLSL printer fixtures":

  test "loop with guard, device buffer load and store":
    check incRes.err == ""
    check incRes.src.startsWith("#version 450")
    check validate("fixture_inc", "kern_fixture_inc", incRes.src)

  test "push block with buffer, struct and scalar members":
    check abiRes.err == ""
    check "layout(push_constant, std430) uniform KernelParams" in abiRes.src
    check "Ref_float dst;" in abiRes.src
    check "Layout_xy ly;" in abiRes.src
    check "float alpha;" in abiRes.src
    check validate("fixture_abi", "kern_fixture_abi", abiRes.src)

  test "struct local with device buffer member":
    check viewRes.err == ""
    check "Ref_float data;" in viewRes.src
    check validate("fixture_view", "kern_fixture_view", viewRes.src)

  test "loud refusal: surviving pointer local names the lowering pass":
    ## GLSL lowering pass dissolves pointer locals, the bare printer refuses
    ## the survivors naming the pass. The dissolved emission is asserted in t_cirv2_lowering_glsl.nim.
    check aliasRes.err.contains("the GLSL lowering pass dissolves pointer locals")

  test "coordinate builtins, flat index, barrier, shared scratch":
    check coordRes.err == ""
    check "int(gl_GlobalInvocationID.x)" in coordRes.src
    check "int(gl_LocalInvocationIndex)" in coordRes.src
    check "shared float scratch[8];" in coordRes.src
    check "barrier();" in coordRes.src
    check validate("fixture_coord", "kern_fixture_coord", coordRes.src)

  test "subgroup reduction emits the KHR subgroup extensions":
    check reduceRes.err == ""
    check "GL_KHR_shader_subgroup_arithmetic" in reduceRes.src
    check "subgroupAdd(v)" in reduceRes.src
    check validate("fixture_reduce", "kern_fixture_reduce", reduceRes.src)

  test "loud refusal: two kernels in one module":
    check twoErr.contains("exactly one kernel")

  test "loud refusal: device-space local":
    check devLocalErr.contains("device space")

  test "loud refusal: inline assembly":
    check asmErr.contains("inline assembly")

  test "loud refusal: pointer local bound outside the alias set":
    check aliasBadErr.contains("the GLSL lowering pass dissolves pointer locals")

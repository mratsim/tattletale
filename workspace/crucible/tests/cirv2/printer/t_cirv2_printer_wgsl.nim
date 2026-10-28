# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## WGSL printer fixtures: hand-built CIRv2 modules print to WGSL sources.
## The assertions cover the text shape the engine and the naga compiler inside
## the vendored wgpu consume.
## There is no offline WGSL validator, the real check is the ceramic axpy suite on WebGPU.
##
## Run:
##   nim c -r --outdir:build/tests tests/cirv2/printer/t_cirv2_printer_wgsl.nim

import std/[strutils, unittest]
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/backends/printer_wgsl

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
  ## kernel: a device buffer, a by-value struct, a bool and a scalar as parameters,
  ## the body reads the struct fields, tests the bool and sizeof.
  ## The bool parameter read is hand-patched to `name != 0u`, the rewrite the WGSL lowering pass performs
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let layoutTy = tStruct("Layout_xy", fld("F0", i32), fld("F1", i32))
  let kern = procD("kern_fixture_abi", @[
    param("dst", aptrD), param("ly", layoutTy), param("flag", tBool()),
    param("alpha", f32)], tVoid(), @[
    varD("w", i32, sizeofE(f32)),
    varD("acc", f32, addE(f32, sym("alpha", f32),
      convE(f32, dotE(sym("ly", layoutTy), "F1")))),
    iteS(neqE(sym("flag", tBool()), intLit(0, tUint(32))), @[
      asgnS(patE(sym("dst", aptrD), intLit(0, i32)), sym("acc", f32)),
    ]),
    retS(),
  ])
  kern.isKernel = true
  moduleD("fixture_abi", @[layoutTy], @[kern])

proc buildViewModule(): CirModule =
  ## kernel: a struct local with a device buffer member, the write goes through the dot chain.
  ## A pointer-holding struct local refuses at the bare printer, the WGSL lowering pass dissolves it into per-field leaf rewrites
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
  ## kernel: a ptr local bound to the address of a local struct and dereferenced
  ## on both sides of an assignment through the pointee's array member.
  ##
  ## - the bare printer refuses the pointer local
  ## - the WGSL lowering pass dissolves it into its pointee before printing
  ## - t_cirv2_lowering_wgsl.nim asserts the dissolved emission through the pipeline
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
  ## kernel: coordinate builtin component reads, the flat thread index and a barrier.
  ## The workgroup scratch local lives in `m.globals`, the shape the WGSL lowering pass leaves after relocating it out of the kernel body
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let coordTy = tStruct("ivec3", fld("x", i32), fld("y", i32), fld("z", i32))
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

proc buildWholeVectorModule(): CirModule =
  ## kernel: a coordinate builtin read whole, WGSL composes the i32 vector
  ## from the u32 builtin parameter components
  let i32 = tInt(32)
  let coordTy = tStruct("ivec3", fld("x", i32), fld("y", i32), fld("z", i32))
  let kern = procD("kern_wholevector", @[], tVoid(), @[
    varD("g", coordTy, sym("thread_position_in_grid", coordTy, builtin = true,
      coordBuiltin = cbkThreadPositionInGrid)),
    retS(),
  ])
  kern.isKernel = true
  moduleD("fixture_wholevector", @[], @[kern])

proc buildReduceModule(): CirModule =
  ## kernel: a subgroup reduction, no directive, the device requests the feature
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

proc buildConstMemModule(): CirModule =
  ## kernel: a const_mem local printed from `m.globals` as a module-scope const declaration, the shape the WGSL lowering pass leaves
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let kern = procD("kern_fixture_constmem", @[
    param("dst", aptrD)], tVoid(), @[
    asgnS(patE(sym("dst", aptrD), intLit(0, i32)), sym("c", f32)),
    retS(),
  ])
  kern.isKernel = true
  let m = moduleD("fixture_constmem", @[], @[kern])
  m.globals = @[varD("c", f32, fltLit(1.5, f32), caConstant)]
  m

proc buildAtomicModule(): CirModule =
  ## kernel: an fp atomicFetchAdd on a device buffer element, WGSL has no float atomics
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let kern = procD("kern_fixture_atomic", @[
    param("dst", aptrD)], tVoid(), @[
    varD("old", i32, callE(sym("atomicFetchAdd",
      tProcT(@[aptrD, i32], i32), builtin = true,
      atomicsBuiltin = abkAtomicFetchAdd),
      patE(sym("dst", aptrD), intLit(0, i32)), intLit(1, i32))),
    retS(),
  ])
  kern.isKernel = true
  moduleD("fixture_atomic", @[], @[kern])

proc buildBoolConvModule(): CirModule =
  ## kernel: a conversion from a bool kernel parameter, WGSL has no bool-to-numeric conversion
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let kern = procD("kern_boolconv", @[
    param("dst", aptrD), param("flag", tBool())], tVoid(), @[
    asgnS(patE(sym("dst", aptrD), intLit(0, i32)),
      convE(f32, sym("flag", tBool()))),
    retS(),
  ])
  kern.isKernel = true
  moduleD("fixture_boolconv", @[], @[kern])

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
    asmS(@["mov x0, x1"], @[]),
    retS(),
  ])
  kern.isKernel = true
  moduleD("fixture_asm", @[], @[kern])

proc buildWorkgroupSizeModule(): CirModule =
  ## A threads_per_threadgroup read: WGSL has no workgroup_size builtin,
  ## only the @workgroup_size attribute on the entry point
  let i32 = tInt(32)
  let coordTy = tStruct("ivec3", fld("x", i32), fld("y", i32), fld("z", i32))
  let kern = procD("kern_wgsize", @[], tVoid(), @[
    varD("s", i32, dotE(sym("threads_per_threadgroup", coordTy, builtin = true,
      coordBuiltin = cbkThreadsPerThreadgroup), "x")),
    retS(),
  ])
  kern.isKernel = true
  moduleD("fixture_wgsize", @[], @[kern])

proc buildPtrParamModule(): CirModule =
  let f32 = tFloat(32)
  let kern = procD("kern_ptrparam", @[
    param("p", tPtr(caDevice, f32))], tVoid(), @[retS()])
  kern.isKernel = true
  moduleD("fixture_ptrparam", @[], @[kern])

const
  incRes = printWgslModule(buildIncModule())
  abiRes = printWgslModule(buildAbiModule())
  viewRes = printWgslModule(buildViewModule())
  aliasRes = printWgslModule(buildAliasModule())
  coordRes = printWgslModule(buildCoordModule())
  wholeVectorRes = printWgslModule(buildWholeVectorModule())
  reduceRes = printWgslModule(buildReduceModule())
  constMemRes = printWgslModule(buildConstMemModule())
  atomicErr = printWgslModule(buildAtomicModule()).err
  boolConvErr = printWgslModule(buildBoolConvModule()).err
  twoErr = printWgslModule(buildTwoKernelsModule()).err
  devLocalErr = printWgslModule(buildDeviceLocalModule()).err
  asmErr = printWgslModule(buildAsmModule()).err
  wgSizeErr = printWgslModule(buildWorkgroupSizeModule()).err
  ptrParamErr = printWgslModule(buildPtrParamModule()).err

suite "WGSL printer fixtures":

  test "loop with guard, device buffer load and store, scalar binding":
    check incRes.err == ""
    check "@group(0) @binding(0) var<storage, read_write> dst: array<f32>;" in incRes.src
    check "@group(0) @binding(2) var<storage, read> n: i32;" in incRes.src
    check "fn kern_fixture_inc() {" in incRes.src
    check "loop {" in incRes.src
    check "dst[i] = (src[i] + 1.0);" in incRes.src
    check "i = (i + 1);" in incRes.src
    check incRes.src.find("@group(0) @binding(0)") < incRes.src.find("fn kern_fixture_inc")

  test "kernel parameters: buffer, struct, bool, scalar":
    check abiRes.err == ""
    check "struct Layout_xy {" in abiRes.src
    check "F0: i32," in abiRes.src
    check "@group(0) @binding(1) var<storage, read> ly: Layout_xy;" in abiRes.src
    check "@group(0) @binding(2) var<storage, read> flag: u32;" in abiRes.src
    check "@group(0) @binding(3) var<storage, read> alpha: f32;" in abiRes.src
    check "var w: i32 = 4;" in abiRes.src
    check "var acc: f32 = (alpha + f32(ly.F1));" in abiRes.src
    check "if ((flag != 0u)) {" in abiRes.src
    check "dst[0] = acc;" in abiRes.src

  test "loud refusal: pointer-holding struct local names the lowering pass":
    ## Pointer-holding struct locals dissolve to per-field leaf rewrites,
    ## the bare printer refuses the survivor.
    ## t_cirv2_lowering_wgsl.nim asserts the dissolved emission through the pipeline.
    check viewRes.err.contains("the WGSL lowering pass dissolves pointer locals")

  test "loud refusal: pointer local names the lowering pass":
    ## Pointer locals dissolve into their pointees in the WGSL lowering pass,
    ## the bare printer refuses the survivor.
    ## t_cirv2_lowering_wgsl.nim asserts the dissolved emission through the pipeline.
    check aliasRes.err.contains("the WGSL lowering pass dissolves pointer locals")

  test "coordinate builtins, flat index, barrier, workgroup scratch":
    check coordRes.err == ""
    check "var<workgroup> scratch: array<f32, 8>;" in coordRes.src
    check "fn kern_fixture_coord(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_index) local_invocation_index: u32)" in coordRes.src
    check "var gx: i32 = i32(global_id.x);" in coordRes.src
    check "var t: i32 = i32(local_invocation_index);" in coordRes.src
    check "workgroupBarrier();" in coordRes.src
    check "enable subgroups" notin coordRes.src

  test "whole-vector coordinate read composes the vector from the components":
    check wholeVectorRes.err == ""
    check "var g: ivec3 = vec3<i32>(i32(global_id.x), i32(global_id.y), i32(global_id.z));" in wholeVectorRes.src

  test "subgroup reduction emits no directive":
    check reduceRes.err == ""
    check "dst[0] = subgroupAdd(v);" in reduceRes.src
    check "enable subgroups" notin reduceRes.src

  test "const_mem local hoists to a module-scope const":
    check constMemRes.err == ""
    check "const c: f32 = 1.5;" in constMemRes.src
    check constMemRes.src.find("const c: f32 = 1.5;") <
      constMemRes.src.find("fn kern_fixture_constmem")
    check constMemRes.src.contains("dst[0] = c;")

  test "loud refusal: fp atomicFetchAdd has no WGSL form":
    check atomicErr.contains("atomicFetchAdd")

  test "loud refusal: bool conversion":
    check boolConvErr.contains("bool")

  test "loud refusal: two kernels in one module":
    check twoErr.contains("exactly one kernel")

  test "loud refusal: device-space local":
    check devLocalErr.contains("device space")

  test "loud refusal: inline assembly":
    check asmErr.contains("inline assembly")

  test "loud refusal: threads_per_threadgroup has no WGSL builtin":
    check wgSizeErr.contains("threads_per_threadgroup")

  test "loud refusal: pointer kernel parameter":
    check ptrParamErr.contains("pointer")

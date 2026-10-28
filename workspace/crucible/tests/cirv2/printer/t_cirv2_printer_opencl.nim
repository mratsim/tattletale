# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## OpenCL C printer fixtures: hand-built CIRv2 modules print to OpenCL C
## sources and assert the text shape the engine and the device compiler consume.
## There is no offline OpenCL C validator here.
## The real check runs the ceramic axpy suite on the OpenCL engine.
##
## Run:
##   nim c -r --outdir:build/tests tests/cirv2/printer/t_cirv2_printer_opencl.nim

import std/[strutils, unittest]
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/backends/printer_opencl

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
  ## kernel: a device buffer, a by-value struct, a bool and a scalar as parameters, the body reads the struct fields, the buffer and sizeof
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
    asgnS(patE(sym("dst", aptrD), intLit(0, i32)),
      addE(f32, sym("acc", f32), convE(f32, sym("flag", tBool())))),
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
  ## kernel: coordinate builtin component reads, the flat thread index, a barrier and a threadgroup scratch local
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
    varD("scratch", scratchTy, nil, caThreadgroup),
    callS(sym("threadgroup_barrier", tProcT(@[], tVoid()), builtin = true,
        synchroBuiltin = sbkThreadgroupBarrier), @[]),
    asgnS(patE(sym("dst", aptrD), intLit(0, i32)),
      addE(i32, sym("gx", i32), sym("t", i32))),
    retS(),
  ])
  kern.isKernel = true
  moduleD("fixture_coord", @[coordTy], @[kern])

proc buildReduceModule(): CirModule =
  ## kernel: a subgroup reduction, the KHR subgroup extension
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

proc buildAtomicModule(): CirModule =
  ## kernel: an fp atomicFetchAdd on a device buffer element, the float atomics extension
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

proc buildConstMemModule(): CirModule =
  ## kernel: a const_mem local relocated into `CirModule.globals` by the OpenCL
  ## lowering pass prints as a program-scope __constant declaration.
  ## The fixture hand-populates `m.globals` the way the pass leaves the module.
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

proc buildWholeVectorModule(): CirModule =
  ## A coordinate builtin read whole: OpenCL C has no vector form of the canonical coordinates, the component reads are the contract.
  let i32 = tInt(32)
  let coordTy = tStruct("ivec3", fld("x", i32), fld("y", i32), fld("z", i32))
  let kern = procD("kern_wholevector", @[], tVoid(), @[
    varD("g", coordTy, sym("thread_position_in_grid", coordTy, builtin = true,
      coordBuiltin = cbkThreadPositionInGrid)),
    retS(),
  ])
  kern.isKernel = true
  moduleD("fixture_wholevector", @[], @[kern])

const
  incRes = printOpenclModule(buildIncModule())
  abiRes = printOpenclModule(buildAbiModule())
  viewRes = printOpenclModule(buildViewModule())
  aliasRes = printOpenclModule(buildAliasModule())
  coordRes = printOpenclModule(buildCoordModule())
  reduceRes = printOpenclModule(buildReduceModule())
  atomicRes = printOpenclModule(buildAtomicModule())
  constMemRes = printOpenclModule(buildConstMemModule())
  twoErr = printOpenclModule(buildTwoKernelsModule()).err
  devLocalErr = printOpenclModule(buildDeviceLocalModule()).err
  asmErr = printOpenclModule(buildAsmModule()).err
  wholeVectorErr = printOpenclModule(buildWholeVectorModule()).err

suite "OpenCL printer fixtures":

  test "loop with guard, device buffer load and store":
    check incRes.err == ""
    check "__kernel void kern_fixture_inc(__global float* restrict dst, __global float* restrict src, int n)" in incRes.src
    check "for (;;)" in incRes.src
    check "dst[i] = (src[i] + 1.0f);" in incRes.src

  test "kernel parameters: buffer, struct, bool, scalar":
    check abiRes.err == ""
    check "typedef struct" in abiRes.src
    check "int F0;" in abiRes.src
    check "} Layout_xy;" in abiRes.src
    check "__kernel void kern_fixture_abi(__global float* restrict dst, Layout_xy ly, int flag, float alpha)" in abiRes.src
    check "sizeof(float)" in abiRes.src
    check abiRes.src.contains("dst[0] = (acc + (float)flag);")

  test "struct local with device buffer member":
    check viewRes.err == ""
    check "__global float* data;" in viewRes.src
    check "v.data[0] = a;" in viewRes.src

  test "ptr local aliases its pointee local":
    check aliasRes.err == ""
    check aliasRes.src.contains("Buf* p;")
    check aliasRes.src.contains("p = (&tmp);")
    check aliasRes.src.contains("(*p).data[2] = 9.0f;")
    check aliasRes.src.contains("dst[0] = (*p).data[2];")

  test "coordinate builtins, flat index, barrier, local scratch":
    check coordRes.err == ""
    check "(int)get_global_id(0)" in coordRes.src
    check "(int)(get_local_id(2)*get_local_size(0)*get_local_size(1) + get_local_id(1)*get_local_size(0) + get_local_id(0))" in coordRes.src
    check "__local float scratch[8];" in coordRes.src
    check "barrier(CLK_LOCAL_MEM_FENCE);" in coordRes.src

  test "subgroup reduction emits the KHR subgroup extension":
    check reduceRes.err == ""
    check "#pragma OPENCL EXTENSION cl_khr_subgroups : enable" in reduceRes.src
    check "sub_group_reduce_add(v)" in reduceRes.src

  test "fp atomicFetchAdd emits the float atomics extension":
    check atomicRes.err == ""
    check "#pragma OPENCL EXTENSION cl_ext_float_atomics : enable" in atomicRes.src
    check "atomic_fetch_add((volatile __global atomic_float*)dst[0], 1)" in atomicRes.src

  test "const_mem local hoists to a program-scope __constant":
    check constMemRes.err == ""
    check "__constant float c = 1.5f;" in constMemRes.src
    check constMemRes.src.find("__constant float c = 1.5f;") <
      constMemRes.src.find("__kernel void")

  test "loud refusal: two kernels in one module":
    check twoErr.contains("exactly one kernel")

  test "loud refusal: device-space local":
    check devLocalErr.contains("device space")

  test "loud refusal: inline assembly":
    check asmErr.contains("inline assembly")

  test "loud refusal: whole-vector coordinate read":
    check wholeVectorErr.contains("whole-vector")

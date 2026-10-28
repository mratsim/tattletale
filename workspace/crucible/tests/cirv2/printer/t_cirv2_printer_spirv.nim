# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## SPIR-V printer fixtures: hand-built CIRv2 modules encode to binary words
## and validate through `spirv-val`.
##
## Run:
##   nim c -r --outdir:build/tests tests/cirv2/printer/t_cirv2_printer_spirv.nim

import std/[os, osproc, strutils, unittest]
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/backends/printer_spirv

proc validate(name: string, words: seq[uint32]): bool =
  ## Runs spirv-val over the encoded words, true when clean.
  let spvPath = getTempDir() / name & ".spv"
  var bytes = newSeq[uint8](words.len * 4)
  for i, w in words:
    bytes[i*4+0] = uint8(w and 0xFF)
    bytes[i*4+1] = uint8((w shr 8) and 0xFF)
    bytes[i*4+2] = uint8((w shr 16) and 0xFF)
    bytes[i*4+3] = uint8((w shr 24) and 0xFF)
  var f = open(spvPath, fmWrite)
  discard f.writeBuffer(bytes[0].addr, bytes.len)
  f.close()
  let (outp, code) = execCmdEx("spirv-val " & spvPath)
  if code != 0:
    echo "spirv-val failed for " & name & ":\n" & outp
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

proc buildMaxModule(): CirModule =
  ## kernel: dst[0] = a > b ? a : b expressed with ite and a temp local
  let f32 = tFloat(32)
  let aptrD = tAptr(caDevice, f32)
  let kern = procD("kern_fixture_max", @[
    param("dst", aptrD), param("a", f32), param("b", f32)], tVoid(), @[
    varD("m", f32),
    iteS(leE(sym("a", f32), sym("b", f32)),
      @[asgnS(sym("m", f32), sym("b", f32))],
      @[asgnS(sym("m", f32), sym("a", f32))]),
    asgnS(patE(sym("dst", aptrD), intLit(0, tInt(32))), sym("m", f32)),
    retS(),
  ])
  kern.isKernel = true
  moduleD("fixture_max", @[], @[kern])

proc buildVecModule(): CirModule =
  ## kernel: vdst[i] = vA + vB over an int32x2 vector held in a local
  let i32 = tInt(32)
  let v2 = tVec(2, i32)
  let aptrD = tAptr(caDevice, i32)
  let kern = procD("kern_fixture_vec", @[
    param("vdst", aptrD), param("a0", i32), param("a1", i32),
    param("b0", i32), param("b1", i32)], tVoid(), @[
    varD("vA", v2, oconstrE(v2, kv("F0", sym("a0", i32)), kv("F1", sym("a1", i32)))),
    varD("vB", v2, oconstrE(v2, kv("F0", sym("b0", i32)), kv("F1", sym("b1", i32)))),
    varD("vS", v2, addE(v2, sym("vA", v2), sym("vB", v2))),
    asgnS(patE(sym("vdst", aptrD), intLit(0, i32)),
      dotE(sym("vS", v2), "F0")),
    asgnS(patE(sym("vdst", aptrD), intLit(1, i32)),
      dotE(sym("vS", v2), "F1")),
    retS(),
  ])
  kern.isKernel = true
  moduleD("fixture_vec", @[], @[kern])

proc buildViewModule(): CirModule =
  ## kernel: a TensorView-shaped struct local whose pointer member indexes
  ## the device buffer, the store writes dst.data[0]
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
  moduleD("fixture_view", @[], @[kern])

proc buildTwoKernelsModule(): CirModule =
  let k1 = procD("kern_a", @[], tVoid(), @[retS()])
  let k2 = procD("kern_b", @[], tVoid(), @[retS()])
  k1.isKernel = true
  k2.isKernel = true
  moduleD("fixture_two", @[], @[k1, k2])

proc buildSmemModule(): CirModule =
  let i32 = tInt(32)
  let kern = procD("kern_smem", @[], tVoid(), @[
    varD("s", i32, nil, caThreadgroup),
    retS(),
  ])
  kern.isKernel = true
  moduleD("fixture_smem", @[], @[kern])

proc buildNarrowModule(): CirModule =
  let kern = procD("kern_narrow", @[param("x", tInt(8))], tVoid(), @[retS()])
  kern.isKernel = true
  moduleD("fixture_narrow", @[], @[kern])

const
  incRes = printSpirvModule(buildIncModule())
  maxRes = printSpirvModule(buildMaxModule())
  vecRes = printSpirvModule(buildVecModule())
  viewRes = printSpirvModule(buildViewModule())
  twoErr = printSpirvModule(buildTwoKernelsModule()).err
  smemErr = printSpirvModule(buildSmemModule()).err
  narrowErr = printSpirvModule(buildNarrowModule()).err

suite "SPIR-V printer fixtures":

  test "ite with guard, loop, device buffer load and store":
    check incRes.err == ""
    check incRes.words.len > 0
    check validate("fixture_inc", incRes.words)

  test "ite with both branches, scalar params":
    check maxRes.err == ""
    check maxRes.words.len > 0
    check validate("fixture_max", maxRes.words)

  test "vector arithmetic":
    check vecRes.err == ""
    check vecRes.words.len > 0
    check validate("fixture_vec", vecRes.words)

  test "struct local with device pointer member":
    check viewRes.err == ""
    check viewRes.words.len > 0
    check validate("fixture_view", viewRes.words)

  test "loud refusal: two kernels in one module":
    check twoErr.contains("exactly one kernel")

  test "loud refusal: shared-memory local":
    check smemErr.contains("shared/constant locals")

  test "loud refusal: narrow push member":
    check narrowErr.contains("narrower than 32 bits")

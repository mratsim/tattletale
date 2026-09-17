## Emitted discriminator tokens must track the instantiated body, not
## source position. Variant B mirrors variant A with inserted comments
## and shifted lines, and each kernel must still run its own factor.
##
## Run:
##   nim c -r --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/metal/test_metal_generic_inst_name_stability.nim

import std/unittest
import std/[strutils, sets]
import workspace/crucible

# ── Variant A ──

proc scaleA[Factor: static int](
    outp, inp: ptr UncheckedArray[float32]) {.device.} =
  outp[0] = inp[0] * float32(Factor)

const artA = metal:
  proc kA16(outp, inp: ptr UncheckedArray[float32]) {.global.} =
    scaleA[16](outp, inp)
  proc kA32(outp, inp: ptr UncheckedArray[float32]) {.global.} =
    scaleA[32](outp, inp)
  proc kA7(outp, inp: ptr UncheckedArray[float32]) {.global.} =
    scaleA[7](outp, inp)

# ── Variant B: variant A with filler comments and shifted lines ──

proc scaleB[Factor: static int](
    outp, inp: ptr UncheckedArray[float32]) {.device.} =
  # Filler comment line one.
  # Filler comment line two.

  outp[0] = inp[0] * float32(Factor)
  # Trailing filler comment.

  # Blank filler line.

const artB = metal:
  # Filler comment before the kernels.

  proc kB16(outp, inp: ptr UncheckedArray[float32]) {.global.} =
    # Filler comment inside the kernel.
    scaleB[16](outp, inp)

  proc kB32(outp, inp: ptr UncheckedArray[float32]) {.global.} =
    scaleB[32](outp, inp)
    # Trailing filler comment.

  proc kB7(outp, inp: ptr UncheckedArray[float32]) {.global.} =
    scaleB[7](outp, inp)

func discriminators(artifact, procName: string): HashSet[string] =
  ## Returns the trailing base58 discriminator token of every emitted device
  ## function named `procName`.
  var i = 0
  while true:
    let idx = artifact.find(procName & "_", i)
    if idx < 0:
      break
    var j = idx
    while j < artifact.len and artifact[j] in IdentChars:
      inc j
    var tok = ""
    for part in artifact[idx ..< j].split('_'):
      if part.len > 0:
        tok = part
    result.incl tok
    i = j

proc runKernel(artifact: string, launcher: string): seq[float32] =
  var engine = bkMetal.init()
  engine.ingest(artifact)
  var inp = newSeq[float32](1)
  inp[0] = 1.0'f32
  var outp = newSeq[float32](1)
  engine.run<<(grid: (1, 1), blk: (32, 1))>>(launcher, outp, (inp,))
  result = outp

proc runTest() =
  suite "Metal - generic instantiation name stability under source edits":
    test "discriminator tokens are unchanged by comment and line shifts":
      let dA = discriminators(artA, "scaleA")
      let dB = discriminators(artB, "scaleB")
      check dA.len == 3
      check dB.len == 3
      check dA == dB

    test "both variants compute their own factors at runtime":
      check runKernel(artA, "kA16") == @[16.0'f32]
      check runKernel(artA, "kA32") == @[32.0'f32]
      check runKernel(artA, "kA7") == @[7.0'f32]
      check runKernel(artB, "kB16") == @[16.0'f32]
      check runKernel(artB, "kB32") == @[32.0'f32]
      check runKernel(artB, "kB7") == @[7.0'f32]

when isMainModule:
  runTest()

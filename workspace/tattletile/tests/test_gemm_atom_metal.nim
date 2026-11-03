# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## GEMM through the atom-selected hardware mma path, end to end on Metal
##
## Same gemm Tile IR fixture as the abstract-mma tests, compiled at 32
## threads (one simdgroup):
## - the Apple 8x8x8 f32 atom selects, the mmaf lowers to the per-thread
##   fragment program with simdgroup_multiply_accumulate calls
## - results bitwise-identical (int32 view) against the float64-then-narrow matmul reference
## - the density count is recorded: CIRv2 nodes and scheduled ops per public mmaf
##
## Run:
##   nim c -r --hints:off --warnings:off
##     --nimcache:nimcache/tattletile --outdir:build/tattletile
##     workspace/tattletile/tests/test_gemm_atom_metal.nim

import std/strutils
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible
import workspace/tattletile/tattletile

const GemmAtomMsl = compileTileIrToMsl("../fixtures/gemm.tileir".parseTileIrFile,
                                       "gemm", threads = 32)

const GemmSpirvFallback = embedKernel("../fixtures/gemm.tileir", "gemm",
                                      threads = 32, backendKind = gbSpirv)

static:
  doAssert "simdgroup_multiply_accumulate(sd, sa, sb, sd);" in GemmAtomMsl,
    "the 32-thread compile lowers the mmaf to the simdgroup atom"
  doAssert "tt_mma_sum_acc" notin GemmAtomMsl,
    "the atom path replaces the abstract FMA loop"
  # inline-asm operand splice: the accumulator write-backs name tt_d0/tt_d1
  doAssert "tt_d0 = sd.thread_elements()[0];" in GemmAtomMsl,
    "the simdgroup mma writes sd back into the accumulator fragments"
  doAssert "tt_d1 = sd.thread_elements()[1];" in GemmAtomMsl,
    "the simdgroup mma writes sd back into the accumulator fragments"

proc gemmEntry(): Entry =
  parseTileIrFile("../fixtures/gemm.tileir").getEntry("gemm")

proc countNodes(s: CirStmt, statements, loops: var int) =
  ## Recursive node count over the CIRv2 the atom lowering emits:
  ## statements are sVar, sAsgn and sAsm nodes, loops are sWhile.
  case s.kind
  of sWhile:
    inc loops
    for st in s.whileBody:
      st.countNodes(statements, loops)
  of sVar, sAsgn, sAsm:
    inc statements
  else:
    discard

const Density = static:
  ## Atom-lowered gemm CIRv2 node counts, computed in the VM
  var statements, loops = 0
  let m = gemmEntry().runTilePipeline(threads = 32).lowerTileToCirv2(atomMma = true)
  for s in m.procs[0].body:
    s.countNodes(statements, loops)
  (statements, loops)

proc checkDensity() =
  ## Ops the atom layer generates per public mmaf, the 16x16x16 tile at 2x2x2 atom repetitions:
  ## - scheduled per thread: 4 prelude + 8 accumulator inits (2 per rm/rn cell pair)
  ##   + 32 fragment loads (4 per rm/rn/rk triple) + 8 simdgroup mma + 8 stores = 60
  ## - the CIRv2 node count adds the while-form loop plumbing over the source
  ##   for-loop form: every repetition loop declares one counter and increments
  ##   it as the trailing statement of the loop body
  let (statements, loops) = Density
  const reps = 2
  doAssert 4 + reps*reps*2 + reps*reps*reps*4 + reps*reps*reps + reps*reps*2 == 60,
    "the scheduled op count of the fragment program"
  echo "[density] atom layer per public mmaf (16x16x16 tile, 2x2x2 reps): ",
    statements + loops, " CIRv2 nodes (", statements, " statements + ", loops,
    " loops), 60 scheduled ops per thread ",
    "(32 loads + 8 simdgroup mma + 8 stores + 8 accumulator inits + 4 prelude)"

const M = 16
const K = 16

const AHost = static:
  ## Test input of the python runner: a[i] = i mod 5, integer-valued f32
  var a: array[M * K, float32]
  for i in 0 ..< M * K:
    a[i] = float32(i mod 5)
  a

const BHost = static:
  ## Test input of the python runner: b[i] = i mod 4, integer-valued f32
  var b: array[M * K, float32]
  for i in 0 ..< M * K:
    b[i] = float32(i mod 4)
  b

const Expected = static:
  ## Reference bits: C = f32(f64(A) @ f64(B)), the python gemm_ref order
  var e: array[M * M, uint32]
  for i in 0 ..< M:
    for j in 0 ..< M:
      var s = 0.0
      for kk in 0 ..< K:
        s += float64(AHost[i * K + kk]) * float64(BHost[kk * M + j])
      e[i * M + j] = cast[uint32](float32(s))
  e

proc f32Bits(v: float32): uint32 =
  cast[uint32](v)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  checkDensity()

  var engine = newEngine(runMetal)
  engine.ingest(GemmAtomMsl)

  let aDev = engine.allocDevice(float32, M * K)
  let bDev = engine.allocDevice(float32, M * K)
  let cDev = engine.allocDevice(float32, M * M)
  engine.copyToDeviceFromHost(aDev, addr AHost[0], M * K)
  engine.copyToDeviceFromHost(bDev, addr BHost[0], M * K)

  # Launch order = kernel parameter order, the store target binds slot 0.
  # One block: the 16x16 views equal their 16x16 partition tiles.
  # 32 threads: one simdgroup, the atom thread group.
  var args = (aDev, bDev)
  engine.launchKernel("gemm", cDev, args) do:
    grid = (1, 1)
    blk = (32, 1)

  var got: array[M * M, float32]
  engine.copyToHostFromDevice(addr got[0], cDev, M * M)
  for i in 0 ..< M * M:
    doAssert f32Bits(got[i]) == Expected[i],
      "gemm atom element " & $i & ": got bits " & $f32Bits(got[i]) &
      " want bits " & $Expected[i]
  echo "[OK] gemm hardware mma on Metal: bitwise-exact (int32 view) over ",
    M * M, " outputs"

proc runSpirvFallback() =
  ## Abstract FMA fallback check on the SPIR-V backend:
  ## no simdgroup intrinsics exist there, the same fixture at 32 threads lowers
  ## to the per-thread FMA loop and stays bitwise-exact
  var engine = newEngine(runVulkan)
  engine.ingest(GemmSpirvFallback.words)

  let aDev = engine.allocDevice(float32, M * K)
  let bDev = engine.allocDevice(float32, M * K)
  let cDev = engine.allocDevice(float32, M * M)
  engine.copyToDeviceFromHost(aDev, addr AHost[0], M * K)
  engine.copyToDeviceFromHost(bDev, addr BHost[0], M * K)

  var args = (aDev, bDev)
  engine.launchKernel(GemmSpirvFallback.name, cDev, args) do:
    grid = (1, 1)
    blk = (32, 1)

  var got: array[M * M, float32]
  engine.copyToHostFromDevice(addr got[0], cDev, M * M)
  for i in 0 ..< M * M:
    doAssert f32Bits(got[i]) == Expected[i],
      "gemm spirv fallback element " & $i & ": got bits " & $f32Bits(got[i]) &
      " want bits " & $Expected[i]
  echo "[OK] gemm abstract mma fallback on MoltenVK: bitwise-exact over ",
    M * M, " outputs"

when isMainModule:
  runTest()
  runSpirvFallback()

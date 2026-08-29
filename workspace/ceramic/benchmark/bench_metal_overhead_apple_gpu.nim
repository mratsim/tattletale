## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT.)
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0.)
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Benchmark: Metal dispatch-path overhead decomposition (Apple GPU)
##
## Splits the per-call cost of the MetalEngine run contract into allocation, host
## memcpy, and encode/commit/wait, at tile-GEMM buffer sizes: fp32 output, fp16
## inputs. bench_matmul_apple_gpu's K=16 probe measures the same total, this file
## attributes it to components.
##
## Probes, per shape (median of `NbSamples` samples after `WarmupSamples` warmups):
##   alloc/free      4 fresh MTLBuffers (out + 2 inputs + scalars), no copies, no kernel
##   alloc+memcpy    same buffers + copyMem in and back, no kernel
##   trivial kernel  full run contract, signature-identical kernel writing one element
##   K=16 gemm       real fusedGemm, one k-block, cross-check
##
## Kernel probes bind every device arg page-aligned with a page-multiple byte
## length, so all of them wrap no-copy exactly like bench_matmul_apple_gpu. A copied
## arg would bill the run a transfer the GEMM bench never pays.
##
## Usage:
##   nim cpp -r --hints:off --warnings:off \
##     --outdir:build/bench_metal_overhead_apple_gpu \
##     --nimcache:nimcache/bench_metal_overhead_apple_gpu \
##     workspace/ceramic/benchmark/bench_metal_overhead_apple_gpu.nim

import std/[monotimes, times, algorithm, strutils, strformat]

import workspace/crucible
# initMetal/allocBuffer/releaseBuffer come from `exec/metal_runtime`,
# which the engine module re-exports. The {.all.} adds the engine's private symbols,
# which nothing here uses.
import workspace/crucible/src/runtime/engines/metal {.all.}
import ../src/kernels/k_tile_gemm

# ═════════════════════════════════════════════════════════════════════════
#  Config
# ═════════════════════════════════════════════════════════════════════════

const
  ProblemSizes = [2048, 4096]
  NbSamples = 10
  WarmupSamples = 3
  ProbeK = 16            # K of the k-block probes: one k-block, so their A/B
                         # are N×ProbeK strips, not full squares

const touchKernel = "touch"
const gemmKernel = "fusedGemm"

# One metal block = one library: ingest replaces the previous artifact,
# so this single source holds both probe kernels.
const probeMsl = metal:
  # Dispatch probe kernel: every invoke stores the identical value and D[0] is never
  # read back. The concurrent same-value store is deliberate, a per-invoke guard
  # would bill the measurement its own serialization.
  proc touch(D: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16],
             N, K, M: int32) {.global.} =
    D[0] = float32(A[0]) + float32(B[0])

  proc fusedGemm(D: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16],
                 N, K, M: int32) {.global.} =
    matmul(D, A, B, N, K, M)

proc getpagesize(): cint {.importc: "getpagesize", header: "<unistd.h>".}

proc posix_memalign(memptr: ptr pointer, alignment: csize_t, size: csize_t): cint
  {.importc: "posix_memalign", header: "<stdlib.h>".}

func roundPages(nbytes: int, ps = getpagesize()): int =
  ## Byte length rounded up to a host-page multiple, 16 KiB on Apple Silicon.
  ## The device-visible length must be a page multiple for a no-copy binding.
  (nbytes + ps - 1) div ps * ps

proc allocPages(nbytes: int): pointer =
  ## Allocation that is page-aligned and has a byte length that is a page multiple,
  ## the two requirements of no-copy binding. Any other length gets an allocated
  ## buffer and a copy.
  let ps = getpagesize()
  var p: pointer = nil
  doAssert posix_memalign(addr p, csize_t(ps), csize_t(roundPages(nbytes, ps))) == 0,
    "posix_memalign failed for " & $nbytes & " bytes"
  p

# ═════════════════════════════════════════════════════════════════════════
#  Benchmark helpers
# ═════════════════════════════════════════════════════════════════════════

func median(xs: seq[float64]): float64 =
  ## Middle element of the sorted samples: index len div 2, which for an even length
  ## is the upper of the two middle values.
  let s = sorted(xs)
  s[s.len div 2]

proc bench(run: proc(), ops: float64): tuple[gflops, medianUs: float64] =
  ## Warmup runs (discarded) then timed samples, median of the samples. `ops`
  ## is the FLOP count of one timed unit and reaches `gflops` only, which is why
  ## the cost-measuring probes pass 1.0 and read `medianUs`.
  for _ in 0 ..< WarmupSamples:
    run()

  var times = newSeq[float64](NbSamples)
  for s in 0 ..< NbSamples:
    let t0 = getMonoTime()
    run()
    let t1 = getMonoTime()
    times[s] = float64((t1 - t0).inNanoseconds) / 1e9

  let med = median(times)
  ((ops / 1e9) / med, med * 1e6)

func mb(nbytes: int): string =
  &"{nbytes.float64 / (1024.0 * 1024.0):.0f} MB"

# ═════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════

proc runBench() =   # engines are RAII, so keep them function-local
  var engine = bkMetal.init()
  engine.ingest(probeMsl)

  echo "=".repeat(72)
  echo "  Metal dispatch-path overhead decomposition — Apple GPU"
  echo &"  Device: {engine.deviceName()}"
  echo "=".repeat(72)
  echo ""

  for N in ProblemSizes:
    let
      outSize = 4 * N * N      # fp32 C/D bytes
      inSize = 2 * N * N       # fp16 A/B bytes per matrix
      scalarSize = 3 * 16      # 3 int32 args at 16-byte constant slots
      probeOps = float64(2 * N * N * ProbeK)  # K=16 probe: one k-block only

    # Output arg: page-aligned with a page-multiple byte length, like the K=16 probe
    # inputs. Seq storage gives no page-alignment guarantee, so a run
    # could alloc+copy+readback the output while the inputs wrap no-copy, an asymmetric
    # measurement. Kernel-visible length stays N×N.
    var hostOut = cast[ptr UncheckedArray[float32]](allocPages(outSize))
    var hostA = newSeq[uint16](N * N)
    var hostB = newSeq[uint16](N * N)
    for i in 0 ..< hostA.len:
      hostA[i] = uint16(0x3C00 + (i and 3))   # fp16 1.0..1.003, valid operands
      hostB[i] = uint16(0x3C00 + ((i + 1) and 3))
    var scalarPack = newSeq[byte](scalarSize)

    var dArg = PtrArg[float32](buf: hostOut, len: N * N, off: 0)

    echo &"  {N}³ buffers: out {mb(outSize)} fp32, in {mb(inSize)} fp16 ×2, scalars {scalarSize} B"

    # K=16 probes read an N×ProbeK strip of each input, so the arg length names
    # the operand the kernel reads, not a full square. Page alignment and a length
    # that is a page multiple keep the strips on the no-copy path.
    let
      probeElems = N * ProbeK
      probeBytes = probeElems * sizeof(uint16)
      probeLen = roundPages(probeBytes) div sizeof(uint16)
    var probeA = cast[ptr UncheckedArray[uint16]](allocPages(probeBytes))
    var probeB = cast[ptr UncheckedArray[uint16]](allocPages(probeBytes))
    for i in 0 ..< probeElems:
      probeA[i] = uint16(0x3C00 + (i and 3))    # fp16 1.0..1.003
      probeB[i] = uint16(0x3C00 + ((i + 1) and 3))
    var aKArg = PtrArg[uint16](buf: probeA, len: probeLen, off: 0)
    var bKArg = PtrArg[uint16](buf: probeB, len: probeLen, off: 0)

    echo &"  K=16 probe inputs: {probeElems} fp16 ×2 = {probeBytes div 1024} KiB each"

    # ── Probe 1: allocation/free cycle. allocBuffer/releaseBuffer only, 4
    # fresh MTLBuffers per iteration on a bare device, no copies, no kernel.
    block:
      let dev = initMetal().device
      proc runAlloc() =
        var outBuf = allocBuffer(dev, outSize)
        var aBuf = allocBuffer(dev, inSize)
        var bBuf = allocBuffer(dev, inSize)
        var sBuf = allocBuffer(dev, scalarSize)
        releaseBuffer(outBuf)
        releaseBuffer(aBuf)
        releaseBuffer(bBuf)
        releaseBuffer(sBuf)
      let r = bench(runAlloc, 1.0)
      echo &"    alloc/free      {int(r.medianUs):>6d} μs   (4 fresh MTLBuffers per call)"

    # ── Probe 2: allocation + host memcpy. Same 4 buffers per iteration,
    # copyMem in (out + 2 inputs + scalars) + copyMem back (out), no kernel.
    block:
      let dev = initMetal().device
      proc runCopy() =
        var outBuf = allocBuffer(dev, outSize)
        var aBuf = allocBuffer(dev, inSize)
        var bBuf = allocBuffer(dev, inSize)
        var sBuf = allocBuffer(dev, scalarSize)
        copyMem(outBuf.data, addr hostOut[0], outSize)
        copyMem(aBuf.data, addr hostA[0], inSize)
        copyMem(bBuf.data, addr hostB[0], inSize)
        let dst = cast[ptr UncheckedArray[byte]](sBuf.data)
        copyMem(addr dst[0], addr scalarPack[0], scalarSize)
        copyMem(addr hostOut[0], outBuf.data, outSize)
        releaseBuffer(outBuf)
        releaseBuffer(aBuf)
        releaseBuffer(bBuf)
        releaseBuffer(sBuf)
      let r = bench(runCopy, 1.0)
      echo &"    alloc+memcpy    {int(r.medianUs):>6d} μs   (+ copyMem in/out, no kernel)"

    # ── Probe 3: trivial kernel. Full run contract
    # (no-copy binds, scalar pack, encode/commit/wait) with a kernel
    # of the GEMM's exact signature writing one element, same grid/blk as the GEMM.
    block:
      proc runTouch() =
        engine.run << (grid: (N div 32, N div 32), blk: (32, 1)) >>
          (touchKernel, dArg, (aKArg, bKArg, int32(N), int32(ProbeK), int32(N)))
      let r = bench(runTouch, 1.0)
      echo &"    trivial kernel  {int(r.medianUs):>6d} μs   (full run contract, ~zero compute)"

    # ── Probe 4: K=16 gemm. Real fusedGemm at one k-block, cross-checks
    # the trivial-kernel total against bench_matmul_apple_gpu's K=16 overhead probe.
    block:
      proc runK16() =
        engine.run << (grid: (N div 32, N div 32), blk: (32, 1)) >>
          (gemmKernel, dArg, (aKArg, bKArg, int32(N), int32(ProbeK), int32(N)))
      let r = bench(runK16, probeOps)
      echo &"    K=16 gemm       {int(r.medianUs):>6d} μs   (cross-check)"
    echo ""

  echo "Done."

when isMainModule:
  runBench()

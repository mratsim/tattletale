# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Two-pass sum, Metal-only bench ceiling.
##
## - one partial per threadgroup (`sum_partials`),
##   one threadgroup folds the partials (`sum_finalize`)
## - `launchSumTwoPasses`: one end-to-end sum in one command buffer,
##   both dispatches encoded back-to-back, one commit, one wait

import std/[importutils, tables]
import workspace/crucible/src/runtime/errors
import workspace/crucible/src/runtime/exec/metal_runtime
import workspace/crucible/src/runtime/engines/metal{.all.}
import workspace/crucible/src/runtime/launch/launch_serialization
import workspace/crucible/src/runtime/engines/engines_common
import workspace/crucible/src/abis/objc_abi as objc

privateAccess(MetalEngine)
privateAccess(MetalCache)
privateAccess(SerializationCtx)

const sumTwoPassSrc* = """
using namespace metal;

// Two-pass ceiling, pass 1: threadgroup g sums its disjoint strided
// subset into partials[g]. Grid (G, 1), blk (32, 1).
kernel void sum_partials(
        device const float * src [[buffer(1)]],
        device       float * partials [[buffer(0)]],
        constant uint64_t & np [[buffer(2)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        uint3 tpg [[threadgroups_per_grid]]) {
    float acc = 0.0f;
    for (uint64_t i = uint64_t(tgpig.x) * ntg.x + tpitg.x; i < np; i += uint64_t(tpg.x) * ntg.x) {
        acc += src[i];
    }
    acc = simd_sum(acc);
    if (tpitg.x == 0) {
        partials[tgpig.x] = acc;
    }
}

// Two-pass ceiling, pass 2: one threadgroup folds the G partials into
// dst. Grid (1, 1), blk (32, 1).
kernel void sum_finalize(
        device const float * partials [[buffer(1)]],
        device       float * dst [[buffer(0)]],
        constant uint32_t & G [[buffer(2)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    float acc = 0.0f;
    for (uint i = tpitg.x; i < G; i += ntg.x) {
        acc += partials[i];
    }
    acc = simd_sum(acc);
    if (tpitg.x == 0) {
        dst[0] = acc;
    }
}
"""

proc launchSumTwoPasses*(engine: MetalEngine, partialsBuf, srcBuf, dstBuf: objc.ID, np: uint64, G: uint32) =
  ## One end-to-end two-pass sum.
  ## Both dispatches go in a single command buffer, one commit, one wait.
  ##
  ## - the command buffer serializes the pair so pass 2 reads final partials
  ##
  ## Args:
  ## - partialsBuf, srcBuf, dstBuf: raw buffer handles (DeviceMem.mem)
  ## - np: element count
  ## - G: threadgroup (partials) count
  objc.withMemPool:
    var psoP: objc.ID
    if engine.cache.psos.hasKey("sum_partials"):
      psoP = engine.cache.psos["sum_partials"]
    else:
      psoP = compilePipelineState(engine.ctx.device, engine.cache.library, "sum_partials")
      engine.cache.psos["sum_partials"] = psoP
    var psoF: objc.ID
    if engine.cache.psos.hasKey("sum_finalize"):
      psoF = engine.cache.psos["sum_finalize"]
    else:
      psoF = compilePipelineState(engine.ctx.device, engine.cache.library, "sum_finalize")
      engine.cache.psos["sum_finalize"] = psoF

    let cmdBuf = objc.msgSend(engine.ctx.queue, objc.`$$`("commandBuffer"))
    let encoder = objc.msgSend(cmdBuf, objc.`$$`("computeCommandEncoder"))
    discard objc.msgSend(encoder, objc.`$$`("setComputePipelineState:"), psoP)
    var dctxP = SerializationCtx(encoder: encoder)
    bindArgs(dctxP, partialsBuf, (srcBuf, np), store, objc.ID, bkMetal)
    let gridSizeP = objc.MTLSize(width: objc.NSUInteger(G), height: 1, depth: 1)
    let blkSize = objc.MTLSize(width: 32, height: 1, depth: 1)
    discard objc.msgSend(encoder, objc.`$$`("dispatchThreadgroups:threadsPerThreadgroup:"),
                    gridSizeP, blkSize)

    discard objc.msgSend(encoder, objc.`$$`("setComputePipelineState:"), psoF)
    var dctxF = SerializationCtx(encoder: encoder)
    bindArgs(dctxF, dstBuf, (partialsBuf, G), store, objc.ID, bkMetal)
    let gridSizeF = objc.MTLSize(width: 1, height: 1, depth: 1)
    discard objc.msgSend(encoder, objc.`$$`("dispatchThreadgroups:threadsPerThreadgroup:"),
                    gridSizeF, blkSize)

    discard objc.msgSend(encoder, objc.`$$`("endEncoding"))
    discard objc.msgSend(cmdBuf, objc.`$$`("commit"))
    discard objc.msgSend(cmdBuf, objc.`$$`("waitUntilCompleted"))

    let status = objc.msgSendUInt(cmdBuf, objc.`$$`("status"))
    if status != MTLCommandBufferStatusCompleted:
      var detail = "no NSError object provided"
      let err = objc.msgSend(cmdBuf, objc.`$$`("error"))
      if not objc.isNil(err):
        detail = objc.nsStringToNimString(objc.msgSend(err, objc.`$$`("localizedDescription")))
      failLoud("Metal two-pass sum: command buffer failed (status " & $status & "): " & detail)

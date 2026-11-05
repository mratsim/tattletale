# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tensorcery execution context: the deferred-execution record.
##
## An op call marshals its arguments once into a flat byte blob, one OpEntry per op.
## The record stores kernel id, blob bytes, output tensor id and launch config, data only.
##
## eval() drains every pending entry in ONE backend submission and waits for it.
## The backend encodes each entry from its blob and dispatches once per op in record order.
## Reading a tensor materializes it, so eval also runs implicitly on read.

import
  workspace/tensorcery/src/raw/dtypes

type
  KernelId* = enum
    ## Kernels the record can reference, one entry per recorded op.
    opAdd
    opMul

  LaunchCfg* = object
    ## Threadgroup dispatch of one recorded op.
    gridX*, gridY*, gridZ*: int32
    blkX*, blkY*, blkZ*: int32

  OpEntry* = object
    ## One recorded op: data only, one dispatch unit at eval.
    kernel*: KernelId
    backend*: BackendKind  ## owning backend of the entry's dispatch
    blob*: seq[byte]   ## flat image of the op's args object, marshaled at enqueue
    output*: int       ## tensor id of the entry's output, equal to its producing op id
    cfg*: LaunchCfg

  DrainFn* = proc(ctx: Ctx) {.nimcall.}
    ## Backend submission path.
    ## Contract: encodes the backend's own pending entries from their blobs,
    ## one submission per backend total. Wired once by the backend, never per op.

  Ctx* = ref object
    ## Execution context: the op record plus the per-backend drain wiring.
    record*: seq[OpEntry]  ## pending ops, in enqueue order
    drainedTo: int         ## op id of the last drained entry
    opSeq: int             ## op id of the last enqueued entry
    drains*: array[BackendKind, DrainFn]  ## per-backend submission paths, nil until wired

proc enqueue*(ctx: Ctx, kernel: KernelId, blob: seq[byte], cfg: LaunchCfg, backend: BackendKind): int =
  ## Appends one op entry to the record.
  ##
  ## Returns the op id. The op id doubles as the tensor id of the entry's output.
  ctx.opSeq.inc
  ctx.record.add OpEntry(kernel: kernel, backend: backend, blob: blob,
      output: ctx.opSeq, cfg: cfg)
  ctx.opSeq

proc isPending*(ctx: Ctx, opId: int): bool =
  ## Returns true when the op `opId` produced has not been drained yet.
  opId > ctx.drainedTo

proc getPendingCount*(ctx: Ctx): int =
  ## Returns the number of ops waiting in the record since the last eval.
  ctx.record.len

proc eval*(ctx: Ctx) =
  ## Computes every recorded op synchronously, one submission per backend holding pending entries.
  ##
  ## Contract: zero pending entries submit nothing.
  ## After the drains return, every output in the record is readable and the record clears.
  if ctx.record.len == 0:
    return
  for backend in BackendKind:
    if ctx.drains[backend] != nil:
      ctx.drains[backend](ctx)
  ctx.drainedTo = ctx.opSeq
  ctx.record.setLen(0)

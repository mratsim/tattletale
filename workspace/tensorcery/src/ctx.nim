# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tensorcery execution context: the deferred-execution record, which is also the autograd tape.
##
## An op call marshals its arguments once into a flat byte blob, one OpEntry per op, data only.
## eval() drains every pending entry in ONE backend submission and waits for it.
## Reading a tensor materializes it, so eval also runs implicitly on read.
##
## The record persists across eval. backward() appends grad entries to it.
## reset() clears it at the step boundary, and op ids and tensor ids keep counting across resets.

import
  workspace/tensorcery/src/raw/dtypes

type
  KernelId* = enum
    ## Kernels the record can reference, one entry per recorded op.
    opAdd
    opMul       ## scalar multiply, kernel k_mul
    opMul2      ## elementwise product of two tensors, kernel k_mul2
    opFill      ## fill with ones, kernel k_fill
    opReshape   ## pure-metadata reshape, never dispatched
    opPermute   ## pure-metadata dimension permutation, never dispatched
    opSlice     ## pure-metadata range selection, never dispatched

  LaunchCfg* = object
    ## Threadgroup dispatch of one recorded op.
    gridX*, gridY*, gridZ*: int32
    blkX*, blkY*, blkZ*: int32

  OpEntry* = object
    ## One recorded op: data only, one dispatch unit at eval.
    id*: int           ## op id of the entry, referenced by the drain frontier
    kernel*: KernelId
    backend*: BackendKind  ## owning backend of the entry's dispatch
    blob*: seq[byte]   ## flat image of the op's args object, marshaled at enqueue
    inputs*: array[4, int]  ## tensor ids of the op's tensor inputs, 0 for an absent input
    output*: int       ## tensor id of the entry's output
    isGrad*: bool      ## true when the entry belongs to a backward() grad segment
    cfg*: LaunchCfg

  DrainFn* = proc(ctx: Ctx) {.nimcall.}
    ## Backend submission path.
    ## Contract: encodes the backend's own pending entries from their blobs,
    ## one submission per backend total. Wired once by the backend, never per op.

  Ctx* = ref object
    ## Execution context: the op record plus the per-backend drain wiring.
    record*: seq[OpEntry]  ## ops in enqueue order, drained entries stay until reset
    drainedTo: int         ## op id of the last drained entry
    opSeq: int             ## op id of the last enqueued entry
    tensorSeq: int         ## tensor id of the last created tensor
    drains*: array[BackendKind, DrainFn]  ## per-backend submission paths, nil until wired

proc enqueue*(ctx: Ctx, kernel: KernelId, blob: seq[byte], cfg: LaunchCfg, backend: BackendKind,
    output: int, inputA = 0, inputB = 0, inputC = 0, inputD = 0, isGrad = false): int =
  ## Appends one op entry to the record.
  ##
  ## - `output`: the tensor id of the entry's output, allocated through newTid before the call
  ## - `inputA` to `inputD`: tensor ids of the op's tensor inputs, 0 for an absent input
  ## - `isGrad`: marks entries appended by backward()
  ##
  ## Returns the op id of the entry.
  ctx.opSeq.inc
  ctx.record.add OpEntry(id: ctx.opSeq, kernel: kernel, backend: backend, blob: blob,
      inputs: [inputA, inputB, inputC, inputD], output: output, isGrad: isGrad, cfg: cfg)
  ctx.opSeq

proc isShapeOp*(kernel: KernelId): bool =
  ## Returns true for the pure-metadata kernels: their entries record the view change
  ## for the backward walk and every backend drain skips them.
  kernel in {opReshape, opPermute, opSlice}

proc newTid*(ctx: Ctx): int =
  ## Returns a fresh tensor id, unique within the context for the process lifetime.
  ctx.tensorSeq.inc
  ctx.tensorSeq

proc isPending*(ctx: Ctx, opId: int): bool =
  ## Returns true when the op `opId` produced has not been drained yet.
  opId > ctx.drainedTo

proc hasPending*(ctx: Ctx, backend: BackendKind): bool =
  ## Returns true when the record contains an undrained entry owned by `backend`
  ## that a drain can dispatch. Shape-op entries never dispatch, so a record
  ## holding only them submits nothing.
  for entry in ctx.record:
    if entry.backend == backend and ctx.isPending(entry.id) and not isShapeOp(entry.kernel):
      return true

proc getPendingCount*(ctx: Ctx): int =
  ## Returns the number of entries waiting in the record since the last eval, across all backends.
  for entry in ctx.record:
    if ctx.isPending(entry.id):
      result.inc

proc reset*(ctx: Ctx) =
  ## Clears the record: the training-step boundary.
  ##
  ## - gradients already read stay valid
  ## - the next step records into an empty record
  ctx.record.setLen(0)

proc eval*(ctx: Ctx) =
  ## Computes every pending op synchronously, one submission per backend holding pending entries.
  ##
  ## Contract: zero pending entries submit nothing, on every backend.
  ## After the drains return, every pending output is readable and the record stays
  ## for backward() to walk, the next eval drains the grad entries appended after it.
  for backend in BackendKind:
    if ctx.drains[backend] != nil and ctx.hasPending(backend):
      ctx.drains[backend](ctx)
  ctx.drainedTo = ctx.opSeq

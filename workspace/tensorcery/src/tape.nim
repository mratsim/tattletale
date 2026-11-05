# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Per-context tensor registry behind the backward walk.
##
## The record names its inputs and outputs by tensor id only.
## The walk resolves an id to the erased implementation through the registry.
## The registry contains every tensor the record can reference:
##
## - leaves and op outputs, registered at creation
## - gradients backward() produced, in the grad map keyed by the differentiated tensor
##
## Ctx cannot name AnyTensorImpl while TensorImpl names Ctx, so both maps live here keyed
## by the context identity and clear together at the step boundary.
## Single-threaded use, matching the rest of the package.

import std/tables

import pkg/iface

import
  workspace/tensorcery/src/raw/impl,
  workspace/tensorcery/src/ctx

var
  registries: Table[pointer, Table[int, AnyTensorImpl]]
  gradMaps: Table[pointer, Table[int, AnyTensorImpl]]

proc register*(ctx: Ctx, tid: int, impl: AnyTensorImpl) =
  ## Records that tensor id `tid` resolves to `impl` for the backward walk.
  registries.mgetOrPut(cast[pointer](ctx), initTable[int, AnyTensorImpl]())[tid] = impl

proc tapeHas*(ctx: Ctx, tid: int): bool =
  ## Returns true when the registry resolves `tid`.
  registries.hasKey(cast[pointer](ctx)) and
      registries[cast[pointer](ctx)].hasKey(tid)

proc tapeGet*(ctx: Ctx, tid: int): AnyTensorImpl =
  ## Returns the implementation registered under tensor id `tid`.
  registries[cast[pointer](ctx)][tid]

proc setGrad*(ctx: Ctx, tid: int, impl: AnyTensorImpl) =
  ## Stores `impl` as the gradient of the tensor with id `tid`, replacing any previous one.
  gradMaps.mgetOrPut(cast[pointer](ctx), initTable[int, AnyTensorImpl]())[tid] = impl

proc hasGrad*(ctx: Ctx, tid: int): bool =
  ## Returns true when backward() recorded a gradient for tensor id `tid`.
  gradMaps.hasKey(cast[pointer](ctx)) and
      gradMaps[cast[pointer](ctx)].hasKey(tid)

proc getGrad*(ctx: Ctx, tid: int): AnyTensorImpl =
  ## Returns the gradient recorded for tensor id `tid`.
  gradMaps[cast[pointer](ctx)][tid]

proc clear*(ctx: Ctx) =
  ## Drops the registry and the grad map of `ctx`: the step boundary of the tape state.
  registries.del(cast[pointer](ctx))
  gradMaps.del(cast[pointer](ctx))

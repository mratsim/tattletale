# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/json,
  std/os,
  std/tables,
  pkg/iface,
  workspace/libtorch,
  workspace/transformers/src/layers/attn

iface *Model:
  proc forward(input: TorchTensor, positions: TorchTensor, cache: var KVCache): TorchTensor

## Model registry - populated by each model module at initialization
var ModelRegistry* {.compileTime.}: Table[string, proc(modelPath: string, device: DeviceKind): Model {.nimcall.}]

proc loadModel*(modelPath: string, device = kCPU): Model =
  # Pass the compile-time -> runtime boundary
  # and make the var {.compiletime.} a const at runtime
  const registry = static(ModelRegistry)

  let cfg = modelPath.joinPath("config.json").parseFile()
  let archs = cfg["architectures"]

  if archs.len == 0:
    raise newException(ValueError, "[ttt] No architectures found in config.json")

  if archs.len > 1:
    raise newException(ValueError, "[ttt] Multiple architectures not supported")

  let arch = archs[0].getStr()

  if not registry.hasKey(arch):
    raise newException(ValueError, "[ttt] Unknown architecture: " & arch)

  let loader = registry[arch]
  loader(modelPath, device)
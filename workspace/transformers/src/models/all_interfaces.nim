# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/tables,
  pkg/iface,
  workspace/libtorch,
  ../stateful/inference_context

iface *Model:
  proc forward(ctx: var InferenceContext, input_ids: Tensor): Tensor

var ModelRegistry* {.compileTime.}: Table[string, proc(modelPath: string, device: DeviceKind): Model {.nimcall.}]
  ## Model registry - populated by each model module at initialization
  ## Ensure it is used after all individual model files are imported
  ## or it will miss the ones loaded after

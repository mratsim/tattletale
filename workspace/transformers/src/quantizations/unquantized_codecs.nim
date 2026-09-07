# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Unquantized BF16 codec: identity loaders (pass-through from safetensor).
##
## No dtype conversion. Tensors are returned in whatever dtype
## the safetensor stores them in (typically BF16 for modern models).

import
  std/tables,
  std/options,
  pkg/packedjson,
  workspace/safetensors,
  workspace/libtorch,
  ../layers/all_reexports,
  workspace/safetensors/src/collections,
  ./all_interfaces

# ─── Linear ─────────────────────────────────────────────────────────────

proc loadUnquantLinear(
    view: SafetensorsCollection, prefix: string, cfg: JsonNode, device: DeviceKind
): Linear =
  let w = view.getTensorOwned(prefix & ".weight", device)
  let b = if view.hasTensor(prefix & ".bias"):
            some(view.getTensorOwned(prefix & ".bias", device))
          else:
            none(Tensor)
  Linear.init(w, b)


# ─── LMHead ────────────────────────────────────────────────────────────

proc loadUnquantLmHead(
    view: SafetensorsCollection, device: DeviceKind
): LMHead =
  if view.hasTensor("lm_head.weight"):
    LMHead.init(view.getTensorOwned("lm_head.weight", device))
  else:
    nil

# ─── Registration ───────────────────────────────────────────────────────

static:
  QuantLoaderRegistry[qBF16] = QuantLoaders(
    linear: loadUnquantLinear,
    lmHead: loadUnquantLmHead,
    deployDtype: kBFloat16,
  )

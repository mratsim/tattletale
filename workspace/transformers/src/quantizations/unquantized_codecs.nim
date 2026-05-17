# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Unquantized BF16 codec — identity loaders (pass-through from safetensor).
##
## No dtype conversion; tensors are returned in whatever dtype the
## safetensor stores them in (typically BF16 for modern models).

import
  std/tables,
  std/options,
  pkg/packedjson,
  workspace/safetensors,
  workspace/libtorch,
  ../layers/all_reexports,
  ./all_interfaces

# ─── Linear ─────────────────────────────────────────────────────────────

proc loadUnquantLinear(
    st: Safetensor, prefix: string, cfg: JsonNode
): Linear =
  let w = st.getTensorOwned(prefix & ".weight")
  let b = if st.tensors.hasKey(prefix & ".bias"):
            some(st.getTensorOwned(prefix & ".bias"))
          else:
            none(Tensor)
  Linear.init(w, b)

# ─── RMSNorm ────────────────────────────────────────────────────────────

proc loadUnquantRmsNorm(
    st: Safetensor, prefix: string, cfg: JsonNode
): Tensor =
  st.getTensorOwned(prefix & ".weight")

# ─── Embedding ──────────────────────────────────────────────────────────

proc loadUnquantEmbedding(
    st: Safetensor, prefix: string, cfg: JsonNode
): Tensor =
  st.getTensorOwned(prefix & ".weight")

# ─── Registration ───────────────────────────────────────────────────────

static:
  QuantLoaderRegistry[qBF16] = QuantLoaders(
    linear: loadUnquantLinear,
    rmsNorm: loadUnquantRmsNorm,
    embedding: loadUnquantEmbedding,
    activationDtype: kBFloat16,
  )

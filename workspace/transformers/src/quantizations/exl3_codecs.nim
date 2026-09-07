# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## EXL3 layer-specific serialization.
##
## Maps safetensors tensor layout → Linear object construction.
## This is the ONLY file that knows EXL3 safetensors key paths
## (like `.trellis`, `.suh`, `.svh`).

import
  std/options,
  std/tables,
  pkg/packedjson,
  workspace/safetensors,
  workspace/libtorch as F,
  workspace/transformers/src/layers/linear,
  workspace/transformers/src/layers/lmhead,
  workspace/transformers/src/quantizations/all_interfaces,
  workspace/transformers/src/quantizations/exl3,
  workspace/safetensors/src/collections

# ─── Helpers ─────────────────────────────────────────────────

func derive_K(trellis: F.Tensor): int =
  ## Derive bitrate K from trellis shape.
  ## trellis must have at least 3 dimensions.
  if trellis.dim < 3:
    raise newException(ValueError,
      "[ttt] derive_K: trellis must have at least 3 dimensions, got " & $trellis.dim)
  let d2 = trellis.size(2)
  let numerator = d2 * 16
  if numerator mod 256 != 0:
    raise newException(ValueError,
      "[ttt] derive_K: trellis.size(2)*16 must be divisible by 256, got size(2)=" & $d2)
  d2 * 16 div 256

func derive_cb(has_mcg, has_mul1: bool): int =
  ## Codebook variant: 0=default, 1=MCG, 2=MUL1.
  if has_mcg: 1
  elif has_mul1: 2
  else: 0

# ─── Linear projection loader ────────────────────────────────

proc loadExl3Linear(
    view: SafetensorsCollection, prefix: string, cfg: JsonNode, device: DeviceKind
): Linear =
  ## Load one EXL3-quantized linear projection from safetensors.
  ## EXL3 always operates in float16 — weight, scales, activations.
  let trellis = view.getTensorOwned(prefix & ".trellis", device)
  let suh = view.getTensorOwned(prefix & ".suh", device).to(kFloat16)
  let svh = view.getTensorOwned(prefix & ".svh", device).to(kFloat16)
  let bias =
    if view.hasTensor(prefix & ".bias"):
      some(view.getTensorOwned(prefix & ".bias", device))
    else:
      none(F.Tensor)

  let K = derive_K(trellis)
  let has_mcg = view.hasTensor(prefix & ".mcg")
  let has_mul1 = view.hasTensor(prefix & ".mul1")
  let cb = derive_cb(has_mcg, has_mul1)
  let in_f = suh.size(0)
  let out_f = svh.size(0)

  let w = exl3_reconstruct(trellis, K, cb, in_f, out_f).contiguous()  # [in_f, out_f] for F.mm layout

  Linear.init(
    weight = w,
    bias,
    suh,
    svh
  )

# ─── LM Head loader (EXL3-quantized) ──────────────────────────

proc loadExl3LmHead(
    view: SafetensorsCollection, device: DeviceKind
): LMHead =
  ## Load EXL3-quantized lm_head from safetensors.
  ## lm_head is huge ([1024, 151936]), reconstruct on CPU to avoid OOM on GPU.
  ## Result weight goes to the requested device.
  if not (view.hasTensor("lm_head.trellis") or view.hasTensor("lm_head.suh") or
          view.hasTensor("lm_head.svh")):
    return nil
  let trellis = view.getTensorOwned("lm_head.trellis", kCPU)
  let suh = view.getTensorOwned("lm_head.suh", kCPU).to(kFloat16)
  let svh = view.getTensorOwned("lm_head.svh", kCPU).to(kFloat16)
  let K = derive_K(trellis)
  let cb = derive_cb(view.hasTensor("lm_head.mcg"), view.hasTensor("lm_head.mul1"))
  let in_f = suh.size(0)
  let out_f = svh.size(0)

  let w = exl3_reconstruct(trellis, K, cb, in_f, out_f).contiguous()
  LMHead.init(
    weight = w.to(device),
    suh = suh.to(device),
    svh = svh.to(device),
  )

# ─── Registration ─────────────────────────────────────────────

static:
  QuantLoaderRegistry[qExl3] = QuantLoaders(
    linear: loadExl3Linear,
    lmHead: loadExl3LmHead,
    deployDtype: kFloat16,
  )

# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Models module - imports all model implementations and provides the generic loadModel proc
##
## Import order matters:
## 1. all_interfaces - defines ModelRegistry and Model iface
## 2. Individual models (qwen3, etc.) - populate ModelRegistry via static blocks
## 3. This file - uses ModelRegistry in loadModel (after it's populated)

import ./models/all_reexports
export all_reexports

import std/json
import std/os
import std/tables
import std/strutils
import std/sequtils
import workspace/libtorch as F
import workspace/toktoktok
import ./stateful/orchestrator
import ./samplers

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

proc parseTorchDtype(s: string): ScalarKind =
  ## Parse dtype string from config.json to ScalarKind enum.
  ## Based on transformers dtype string format (lowercase).
  case s.toLowerAscii()
  of "bfloat16": kBfloat16
  of "float16", "half": kFloat16
  of "float32", "float": kFloat32
  of "float64", "double": kFloat64
  of "uint8", "byte": kUint8
  of "int8", "char": kInt8
  of "int16", "short": kInt16
  of "int32", "int": kInt32
  of "int64", "long": kInt64
  of "bool": kBool
  of "complexfloat16", "complexf16": kComplexF16
  of "complexfloat32", "complexf32", "complexfloat": kComplexF32
  of "complexfloat64", "complexf64", "complexdouble": kComplexF64
  of "qint8": kQint8
  of "quint8": kQuint8
  of "qint32": kQint32
  else:
    raise newException(ValueError, "[ttt] Unknown torch_dtype: " & s)

proc generate*(
        model: Model,
        prompt: string,
        device: DeviceKind | Device = kCPU,
        temp = 1.0f,
        maxTokens = 200,
        maxContextLen: int = -1): string =
  let cfg = model.getConfig()
  let device = model.getDeviceKind()
  let maxCtx = if maxContextLen < 0: cfg.max_position_embeddings else: maxContextLen
  let numPoolPages = computeNumPages(maxCtx, concurrentRequests = 1)
  var orc = init(Orchestrator, cfg.num_hidden_layers, 1,
                 cfg.num_key_value_heads, maxCtx, cfg.head_dim,
                 numPoolPages, parseTorchDtype(cfg.torch_dtype), device)
  defer: orc.endSequence()

  let dtype = parseTorchDtype(cfg.torch_dtype)

  # Tokenize prompt with special tokens
  var ids = model.getTokenizer().encode(prompt)
  let startPos = ids.len

  orc.startSequence(ids.mapIt(it.uint32))
  # TODO: change tokenization to uint32/int32 directly — no point using 64-bit
  #   when vocabulary is at most ~230K (fits in 2^18).
  #   This would avoid the temporary seq allocation from mapIt.

  # === PREFILL: forward on full prompt ===
  let inputIds = F.toTensor([ids]).to(device)
  let logits = model.forward(orc.active_context, inputIds)
  let lastLogits = logits.narrow(1, startPos - 1, 1).squeeze(1)
  var nextToken = sample(lastLogits, temp)
  ids.add(nextToken)

  # === DECODE LOOP: forward on 1 token at a time ===
  while ids.len < startPos + maxTokens:
    # Set position for this decode step
    orc.decodeStep(ids.len - 1, nextToken.uint32, device)

    # Forward on single token: [1, 1]
    let singleToken = F.toTensor([[nextToken]]).to(device)
    let stepLogits = model.forward(orc.active_context, singleToken)

    # Sample next token from [1, 1, vocab] -> [vocab]
    let stepLastLogits = stepLogits.squeeze(0).squeeze(0)
    nextToken = sample(stepLastLogits, temp)
    ids.add(nextToken)

    stdout.write model.getTokenizer().decodeToString([ids[^1]])

    if ids[^1] == cfg.eosTokenId:
      break

  model.getTokenizer().decodeToString(ids)

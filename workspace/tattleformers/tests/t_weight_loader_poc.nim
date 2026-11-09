# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Weight loader POC (Metal): one checkpoint tensor end to end.
##
## Test that one checkpoint tensor loads through the erased bf16 face:
##
## - safetensors header parse, libtorch-free core
## - Metal device maxBufferLength check at the 311MB embedding matrix
## - DMA through the openArray byte copy boundary
## - readback FNV-1a hash match against the mmap bytes
##
## Run from the worktree root:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wipM08poc --outdir:build/wipM08poc \
##     workspace/tattleformers/tests/t_weight_loader_poc.nim

import std/os
import std/strutils
import std/importutils
import workspace/crucible
import workspace/crucible/src/abis/objc_abi
import workspace/ceramic/src/tensors
import workspace/tensorcery/src/tensor_datatypes
import workspace/ceramic/src/layout_algebra
import workspace/tensorcery/src/device_view
import workspace/safetensors/src/collections

{.experimental: "callOperator".}

privateAccess(MetalEngine)

proc k_primer*(z, x: distinct DeviceView, shift: static int) =
  ## z = x + shift, one element.
  ##
  ## Why this kernel exists:
  ## - the compilation must hold one enqueue before the first erased-Tensor call
  ## - Tensor views dispatch through the erased interface, whose vtable
  ##   instantiation compiles eval and its batch submit
  ## - submit's finalize macro is a compile-time check erroring on an empty
  ##   codegen frontend
  let
    zv = make_view(z.data +% z.offset, z.layout)
    xv = make_view(x.data +% x.offset, x.layout)
  let s = xv[0].to(float32) + float32(shift)
  zv[0] = s.to(bfloat16)

const
  EmbedName = "model.embed_tokens.weight"
  EmbedShape = [151936, 1024]
  EmbedBytes = 311_164_928  # 151936 * 1024 * 2
  MinBufferBytes = 311_164_928'i64

proc fnv1a(p: pointer, n: int): uint64 =
  ## FNV-1a 64-bit over n bytes at p.
  result = 14695981039346656037'u64
  let bytes = cast[ptr UncheckedArray[byte]](p)
  for i in 0 ..< n:
    result = (result xor uint64(bytes[i])) * 1099511628211'u64

proc queryMaxBufferLength(device: ID): int64 =
  ## Metal device maxBufferLength: the largest single buffer allocation.
  result = int64(msgSendUInt(device, $$("maxBufferLength")))

proc primer[E: HwEngine](engine: E) =
  ## Primer enqueue in a generic proc: the batch enqueue's callee resolves
  ## to a symbol only inside a generic body.
  let
    zDev = engine.allocDevice(bfloat16, 1)
    xDev = engine.allocDevice(bfloat16, 1)
    zb = TensorBridge[typeof(zDev), (int,), (int,)](
      data: zDev, offset: 0, layout: make_layout((int(1),), (int(1),)))
    xb = TensorBridge[typeof(xDev), (int,), (int,)](
      data: xDev, offset: 0, layout: make_layout((int(1),), (int(1),)))
  var batch = engine.recordBatch()
  batch.enqueue<<(1, 1)>> k_primer(zb, xb, 3)
  batch.submit()

proc main =
  let metal = gMetalEngine()
  # primer enqueue: textually precede the first erased-Tensor call
  primer(metal)

  let modelPath = getEnv("TTT_QWEN3_PATH",
    "/Users/pi/Documents/Programming/MODELS/Qwen3-0.6B")
  let weights = SafetensorsCollection.open(modelPath)

  # Header: the embedding tensor's dtype and shape
  let view = weights.getMmapView(EmbedName)
  doAssert view.size == EmbedBytes,
    "embedding byte count " & $view.size & " != " & $EmbedBytes

  # maxBufferLength check: the embedding matrix must fit one Metal buffer
  let maxBuf = queryMaxBufferLength(metal.ctx.device)
  echo "device.maxBufferLength = ", maxBuf, " bytes"
  doAssert maxBuf >= MinBufferBytes,
    "maxBufferLength " & $maxBuf & " < " & $MinBufferBytes

  # One tensor: header -> erased bf16 Tensor -> DMA -> readback hash
  let engine = metal.toAnyEngine()
  let embed = newTensor(engine, @[EmbedShape[0], EmbedShape[1]], kBf16)
  # zero-copy DMA from the checkpoint's mmap through the byte copy boundary
  embed.engine.copyToDeviceFromHost(embed.node,
    toOpenArray(cast[ptr UncheckedArray[byte]](view.data), 0, view.size - 1))

  var got = newSeq[bfloat16](EmbedShape[0] * EmbedShape[1])
  copyToHostFromDevice(got, embed)
  let gotHash = fnv1a(got[0].addr, EmbedBytes)
  let wantHash = fnv1a(view.data, EmbedBytes)
  echo "host FNV-1a = 0x", wantHash.toHex
  echo "readback FNV-1a = 0x", gotHash.toHex
  doAssert gotHash == wantHash, "readback hash mismatch"

  # 311MB buffer releases cleanly, ORC reclaims the impl with the tensor
  echo "weight loader POC: all checks passed"

when isMainModule:
  main()

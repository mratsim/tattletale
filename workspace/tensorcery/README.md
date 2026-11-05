# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Tensorcery

A deferred-execution tensor library with autograd for CPU, Metal and Vulkan.

## Package surface

Tensorcery records compute instead of running it.
Every op call marshals its arguments into a flat byte blob and appends one
`OpEntry` to the context record.
`ctx.eval()` drains the pending entries in ONE submission per backend.

- `Tensor`: a thin value with context, erased implementation, runtime dtype, device
- op refusals: cross-context, cross-device, mixed-dtype and shape-mismatched operands are refused by name
- shape ops (`reshape`, `permute`, `transpose`, `slice`, `flatten`): pure
  metadata over shared storage, never enqueue
- optimizer updates (`sgdUpdate`, `adamUpdate`): terminal in-place ops

## Autograd

The record doubles as the autograd tape. `backward(loss)` appends the gradient
segment to the same record.

- one training step = forward ops + grad segment + optimizer updates
- one `ctx.eval()` drains all of them in one submission per backend
- Metal and Vulkan expose a `submissionCount` counter, CPU a completion counter
- tests assert the exact count

Op set: `+`, `*`, `* scalar`, `gather`, `layernorm`, `gelu`, `softmax`,
`crossEntropy`, `where`, plus the shape ops and the optimizer updates above.

## Microgpt example

`examples/microgpt.nim` trains a small GPT on the Shakespeare corpus.
After training it generates text.

- d_model 64, 2 layers, 4 heads, block size 16, char-level vocabulary of 65
- training with in-place adam, one eval per step, submission count asserted
- temperature sampling generation
- a forward-only Vulkan pass after training

- no matmul exists on the Tensor face
- every linear maps through the gather-expand matmul:
  `examples/microgpt_model.nim` gathers both operands with constant indices
  into one rank-4 shape, multiplies elementwise and reduces with pairwise
  adds over contiguous half-slices
- the same example code runs on every backend because the op set is uniform

Run from the worktree root:

```
nim c -r --hints:off --warnings:off --path:"$PWD" --outdir:build/tests \
  --nimcache:nimcache/microgpt workspace/tensorcery/examples/microgpt.nim
```

Knobs:

- optional arguments: corpus path, step count, generation character count
- default corpus: `_references_nim_libs/arraymancer/examples/ex06_shakespeare_input.txt` at the workspace root
- backend switch: the `Backend` const near the top of the example, one line
  to change (`bkCpu` or `bkMetal`)

Vulkan pass:

- `VulkanLeg = true` adds a forward-only Vulkan pass after training
- the pass copies the trained weights with `.to(bkVulkan)`, re-checks the loss
  and generates text in one Vulkan submission per eval
- fused backward kernels and the in-place optimizer updates have no Vulkan
  kernels yet and refuse with a named assertion. Training runs on CPU or Metal
  while Vulkan runs forward work

Known limitations:

- Metal device buffers are not freed yet. The resident set grows by each
  step's transient buffers over 60 steps while CPU memory stays flat
- the vocabulary is char-level. The pure-Nim BPE codec round-trips the corpus
  in `tests/t_bpe_roundtrip.nim`
- no trained tokenizer ships an in-tree vocabulary. A 50k-token vocabulary
  would multiply the gather-expand head beyond reasonable cost

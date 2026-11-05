# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Tensorcery

A deferred-execution tensor library with autograd for CPU, Metal and Vulkan.

## What tensorcery is

Tensorcery records compute instead of running it. Every op call marshals its
arguments into a flat byte blob and appends one `OpEntry` to the context record;
`ctx.eval()` drains the pending entries in ONE submission per backend and waits
for completion. The record is also the autograd tape: `backward(loss)` appends
the gradient segment to the same record, and one eval drains forward ops,
gradient ops and optimizer updates together.

- `Tensor` is a thin value: context, erased implementation, runtime dtype, device.
- Ops refuse cross-context, cross-device, mixed-dtype and shape-mismatched
  operands, naming the combo.
- Shape ops (`reshape`, `permute`, `transpose`, `slice`, `flatten`) are pure
  metadata over shared storage and never enqueue.
- Optimizer updates (`sgdUpdate`, `adamUpdate`) are terminal in-place ops.
- The op set: `+`, `*`, `* scalar`, `gather`, `layernorm`, `gelu`, `softmax`,
  `crossEntropy`, `where`, plus the shape ops and the optimizer updates.

## The one-eval-per-step contract

One training step is one record and one submission per backend holding pending
entries: the forward ops, the gradient segment `backward()` appended, and the
in-place optimizer updates drain together in a single `ctx.eval()`. The
backends expose `submissionCount` (Metal, Vulkan) and a completion counter
(CPU) so tests assert the exact count.

## Microgpt example

`examples/microgpt.nim` trains a small GPT (d_model 64, 2 layers, 4 heads,
block size 16, char-level vocabulary from the Shakespeare corpus) and generates
text. No matmul exists on the Tensor face, so every linear maps through the
gather-expand matmul in `examples/microgpt_model.nim`: constant-index gathers
broadcast both operands to one rank-4 shape, an elementwise product and pairwise
adds over contiguous half-slices reduce. The same example code runs on every
backend because the op set is uniform.

Run from the worktree root:

```
nim c -r --hints:off --warnings:off --path:"$PWD" --outdir:build/tests \
  --nimcache:nimcache/microgpt workspace/tensorcery/examples/microgpt.nim
```

Optional arguments: corpus path, step count, generation character count.
The corpus defaults to
`_references_nim_libs/arraymancer/examples/ex06_shakespeare_input.txt`
at the workspace root.

Backend switch: the `Backend` const at the top of the example, one line
(`bkCpu` or `bkMetal`). `VulkanLeg = true` adds a forward-only Vulkan pass
after training: it copies the trained weights with `.to(bkVulkan)`, re-checks
the loss in one Vulkan submission and generates text.

Measured with seed 42, 60 steps: the mean loss drops from 3.96 (steps 1-5
average) to 3.05 (steps 56-60 average) on both CPU and Metal, with
bit-closely agreeing trajectories. Generation produces Shakespeare-shaped
text (speaker names, line breaks) on all three backends.

## Current status

- CPU and Metal: full training pipeline, one submission per step, gradcheck
  batteries in `tests/`.
- Vulkan: the forward op set only. Fused backward kernels and the in-place
  optimizer updates have no Vulkan kernels yet and refuse with a named
  assertion, so training runs on CPU or Metal and Vulkan runs forward work.
- Metal device buffers are not freed yet (`src/backends/metal.nim`), so the
  resident set grows by each step's transient buffers; 60 steps stay far
  inside physical memory and CPU memory is flat.
- The microgpt vocabulary is char-level (65 symbols). The pure-Nim BPE codec
  (`workspace/toktoktok`) round-trips the corpus (`tests/t_bpe_roundtrip.nim`),
  but a trained tokenizer ships no in-tree vocabulary and a 50k-token
  vocabulary multiplies the gather-expand head beyond reasonable cost.

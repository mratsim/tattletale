---
name: writing-gpu-kernels
description: Accumulated, evidence-backed rules for writing GPU kernels in this workspace — register budgets, schedule rules, measured dead ends, probe discipline. Every tip carries its measurement.
license: MIT
metadata:
  audience: kernel authors
  workflow: kernel-development
---

## What I do

I accumulate durable lessons of kernel-writing ops as evidence-backed rules, covering the SDPA perf tracer, GEMM, and tile kernels.
Each tip states its measurement and the op that produced it.
Tips graduate in, nothing enters here without a number behind it.

## When to use me

- writing or retuning any kernel under `workspace/ceramic/src/kernels/`
- pricing a schedule change before benching it
- deciding whether a hypothesis is worth a test

## Register law (first-order constraint)

| Law | Measurement |
|---|---|
| live set under the spill cliff decides speed | cliff at 186 to 210 regs/thread on M4 Max, larger shapes regress 2x to 3x |
| price every schedule feature in registers before benching | batched-softmax over 2 kv-tiles lost 35% at +65 regs/thread |
| live set from emitted MSL struct sizes, fragment = tile elements / 32 lanes | toolchain register reporting is unusable (AGX hides allocation) |
| chunking a reduction shrinks fragments | QK mma over Dk took the k-fragment from 64 to 16 regs, +15% |

## Schedule rules a fusion pass cannot find

These exist only when the composite algorithm is authored as one schedule: the argument for composable algebra over op fusion.

- **Skip-when-unchanged rescale**: if the running max didn't change, skip the rescale+exp entirely.
- Measured +22% f32, +55% bf16 full, +45% causal (C17D11 entry 7), the single biggest lever this op found.
- **Band-edge-only masking**: apply the causal band test only to edge tiles, +7% causal bf16 (C17D11 entry 6).

Detection and loop-shape rules:

- Skip detection must be exact and group-wide: reduce a 0/1 flag through `simd_shuffle_xor`, 5 steps for 32 lanes.
- Fold per-tile invariants out of the kv loop: Q chunk loads, band bounds, and the per-iteration scalar ops that bind once registers fit.
- The 2048^2 residual is 60 to 80 scalar ops per iteration per lane.

## Measured dead ends (do not retry without new hardware)

- **Load pipelining / K double-buffering**: -24% to neutral, the schedule is ALU-bound and not memory-bound (C17D11 entries 4 and 5).
- **V-load hoist above softmax**: -28% both dtypes (entry 4).
- **kv-tile batching**: register cost exceeds the arithmetic win (entry 9).

## Numerics traps

- `exp2(tiny)` rounds to exactly 1.0f, so skip-detection must read the pre-exp2 delta, not the post-exp2 value (fixture suite caught it).
- `xor` reduction is parity, not OR: a mask test written with xor-reduce passes half its cases (caught by the fixture suite).
- Negative slot indices are legal (reversed views): the invariant is *logical index within the domain*, never sign.

## int is the backend int

Kernel and launch signatures take `int`, never `int32`/`int64`:

- the `int` type lowers to the backend default (int32 on GPUs),
  so no conversion is needed
- the signature reads the same on every backend
- narrow explicitly only at an actual ABI boundary. State the narrowing
  in one sentence

## Test discipline

- **The fixture suite is non-negotiable**: every merged change reruns the fixture suite before its TFLOPS entry counts as evidence.
- **A/B best-of**: every comparison is measured both orders, co-tenanted machines drift between runs.
- Record refusals-to-believe in the ledger: a dropped hypothesis with numbers is a deliverable while an untested intuition is not.

Extraction discipline:

- **Verify extraction before believing analysis.**
- A session was lost to a phantom "missing K-loop", an awk pattern stopping at the wrong closing brace.
- If source analysis contradicts green fixtures, suspect the extraction first and design a numeric test against a f64 CPU reference.

## Per-backend notes

- **Metal/M4 Max**: no tensor cores, no half-rate units, fp16/bf16/fp32 all run at the fp32-core rate.
- A dtype-rate hypothesis is wrong by construction: fp32-input GEMM measured 12.25 to 13.36 TF vs fp16 12.95 to 14.29 (C17D11 test 0).

Vulkan and MSL notes:

- **MoltenVK zeroes some property structs** (e.g. reported subgroupSize = 0).
- Do not assume subgroup size 32 in Vulkan kernels, RADV wave32/64 differ.
- Warp specialization is impossible in MSL and TileLang's Metal path disables pipelining, so hand-build any overlap (C17D11 study).

## Shape selection with 1-length modes

Tile shapes measured at 2048^2 with the size-1 compose branch honored (1-length tiler modes are ordinary inputs):

- f32 (8, 8): 8.25 full / 7.67 causal TF
- bf16 (8, 16): 12.55 full / 11.28 causal TF
- both over 2x the (16, 16) shape measured in the same window on the same branch
- NT=32 at MT=8 spills to 0.8 TF (f32): the rule scales down to MT=8

The same window deflated the (16, 16) baseline about 33% below its recorded quiet-machine value.
The quiet-machine f32 (8, 8) peak plausibly exceeds the 8.8 TF target, re-measure co-tenant-free before claiming it.

## Layout-algebra contract: size-1 compose branch

`compose` with a size-1 tiler mode (N==1) evaluates A at one point.
A tiler stride in coordinate-stride basis form is a point of A's basis space, its coefficients are the coordinate components.
Pass the coefficient tuple as the coordinate.

- feeding the form to `crd2idx` as an int degrades to a sentinel for runtime strides and to a tuple-valued sum for basis-strided layouts
- the composed mode then pairs a scalar shape with a nested stride tree
- every later walk (prefix product scan, tuple-stream zip) rejects an incongruent pair like this one
- with the contract honored, 1-length tiler modes (repOuter = MT div atomM = 1) compose and index and emit like every other mode

## Builtins: the payload-less pragma form

A `{.builtin.}` written without a payload sems as `builtin(nil)` over the pragma template's default argument.

- the distinct-scalar builtin resolver must match the defaulted form
- the resolver misses it, float16 and bfloat16 conversions emit calls to undeclared MSL helper functions
- every kernel touching those types fails its Metal library compile at runtime

## Generated-MSL extraction trap

- Extracting functions from generated MSL: brace-match or bound to the next `void`/`kernel` line.
- A `/^}$/` pattern stops at the first column-0 closer, which nested blocks always have.
- That extraction trap cost a full session on a phantom missing K loop.

## Intra-kernel kv-split design (designed, not built)

- In-threadgroup split: blk=(64,1), warp g owns kv range g of 2 on the q-tile, partial (m, l, acc) merged in `{.smem.}` buffers.
- The buffers take 2x8 KB and the merge waits on `threadgroup_barrier()`.
- Merge weights are `exp2(m_g - m*)` per band entry and apply per lane because both warps share the fragment layout.
- Unpriced before building: the stats-to-acc-cell entry correspondence across the rowStats and oMap per-lane layouts.

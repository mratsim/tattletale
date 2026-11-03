---
name: writing-gpu-kernels
description: "Evidence-backed rules for writing GPU kernels in this workspace: register budgets, schedule rules, shape selection, layout-algebra contracts. Append a rule only with a measured number behind it."
license: MIT
metadata:
  audience: developers
  workflow: engineering
---

# Writing GPU Kernels

Measured rules for kernel work on M4 Max (Metal first-class).
A rule enters with its number or it does not enter.

## Register budget

Price every schedule feature in regs per thread on paper before writing MSL:
regs/lane = tile elements x element bytes / (32 lanes x 4 B).

- the M4 Max spill cliff sits between ~186 regs (fast) and ~210 regs/thread
- a change adding a live tile, about +30 to +65 regs, loses even when it
  removes arithmetic
- the live set is a peak over phases: peak = max(QK phase, PV phase)

## Shape selection with 1-length modes

Tile shapes measured at 2048^2 with the one-point compose contract honored
(1-length tiler modes are ordinary inputs):

- f32 (8, 8): 8.25 full / 7.67 causal TF
- bf16 (8, 16): 12.55 full / 11.28 causal TF
- both over 2x the (16, 16) shape measured in the same window on the same branch
- NT=32 at MT=8 spills to 0.8 TF (f32): the rule scales down to MT=8

The same window deflated the (16, 16) baseline about 33% below its recorded
quiet-machine value. The quiet-machine f32 (8, 8) peak plausibly exceeds the 8.8 TF target.
Re-measure co-tenant-free before claiming it.

## Layout-algebra contract: one-point compose

`compose` with a one-point scalar tiler (N==1) evaluates A at one point.
A tiler stride in coordinate-stride basis form is a point of A's basis space,
its coefficients are the coordinate components. Pass the coefficient tuple as the coordinate.

- feeding the form to `crd2idx` as an int degrades to a sentinel for runtime
  strides and to a tuple-valued sum for basis-strided layouts
- the composed mode then pairs a scalar shape with a nested stride tree
- every later walk (prefix product scan, tuple-stream zip) rejects the incongruent pair
- with the contract honored, 1-length tiler modes (repOuter = MT div atomM = 1) compose, index and emit like every other mode

## Builtins: the payload-less pragma form

A `{.builtin.}` written without a payload sems as `builtin(nil)` over the pragma template's default argument.

- the distinct-scalar builtin resolver must match the defaulted form
- the resolver misses it, float16 and bfloat16 conversions emit calls to undeclared MSL helpers
- every kernel touching those types fails its Metal library compile at runtime

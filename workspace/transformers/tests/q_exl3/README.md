# EXL3 tests

The fixture suites enforce through the two harness functions.

- assertStats against the committed 004 uniform-stats frames
- assertArgMax against the committed 005 decision frames
- the 03 logits decisions frame stays in the 002 decimal shape and loads through loadArgmaxRecords
- suites run on the suite device via `testDevice()`, the granular tasks
  live in `config.nims` (`nim test_tf_exl3_qwen3_00_codec`, ...)

- the recorded payloads come from the production EXL3 CUDA kernel (`exllamav3_ext`)
- replays on any device compare through the kinded depth allowances, the depth
  argument carries the composed reordered stages since the recorded reference
- the EXL3-00 codec payload keeps its bit-for-bit contract, no other exl3
  assert runs an exact class

One EXL3 linear composes 3 reordered stages (pre-Hadamard FWHT, fp16 GEMM, post-Hadamard FWHT).
See the 01 layer internals suite for the derivation.

Check the family contracts in `../testgen/FIXTURE_GENERATION.md`.

# EXL3 tests

The fixture suites compare through the recorded-summary surface (fingerprint
stats sidecars, chain checkpoint bands, decision projections, greedy step
checks) and run on any device via `testDevice()`; the granular tasks live in
`config.nims` (`nim test_tf_exl3_qwen3_00_codec`, ...).

The recorded payloads come from the production EXL3 CUDA kernel
(`exllamav3_ext`), so only a CUDA replay is the bit-exact reference class: on
the recording device the suites run the elementwise ulp rows and the full
decision projection, on any other device the chain checkpoint band with the
ulp unit taken in fp16, because EXL3 dequantizes to fp16. The unregistered
suites `t_exl3_qwen3_04_long_residual_3_blocks.nim` and
`t_exl3_qwen3_06_t2t_inference.nim` keep their raw-compare CUDA shape and stay
box-only.

Check the family contracts in `../testgen/FIXTURE_GENERATION.md` and the
budget derivation in `../harness/SPEC.md`.

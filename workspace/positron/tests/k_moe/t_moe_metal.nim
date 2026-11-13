## MoE frontend Metal driver: router numerics, dispatch metadata, weighted combine, both tier paths, seeded fuzz.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/positron/tests/k_moe/t_moe_metal.nim

import std/random
import workspace/crucible
import moe_template

proc main =
  var rng = initRand(FuzzSeed)
  var engine = newEngine(runMetal)
  # Router numerics: softmax and sigmoid, with and without bias, with and without scaling
  moeRouterCase(engine, 1, 8, 3, moeSoftmax, 0.0'f32, 1.0'f32, rng, "router softmax no-bias")
  moeRouterCase(engine, 4, 16, 6, moeSigmoid, 0.0'f32, 1.0'f32, rng, "router sigmoid no-bias")
  moeRouterCase(engine, 4, 16, 6, moeSoftmax, 0.5'f32, 1.0'f32, rng, "router softmax bias")
  moeRouterCase(engine, 4, 16, 6, moeSigmoid, 0.5'f32, 2.5'f32, rng, "router sigmoid bias scaled")
  # topk > E: the excess slots carry -1 and weight 0
  moeRouterCase(engine, 2, 3, 6, moeSoftmax, 0.0'f32, 1.0'f32, rng, "router topk over E")
  # E = 1
  moeRouterCase(engine, 2, 1, 1, moeSoftmax, 0.0'f32, 1.0'f32, rng, "router single expert")
  moeRouterBiasSelectCase(engine)
  moeRouterTieCase(engine)
  echo "[OK] moe router numerics"

  # Dispatch metadata + gather: T=1, empty experts, topk > E, duplicate expert in slots
  moeDispatchCase(engine, @[@[3]], 8, 40, rng, "dispatch single token")
  moeDispatchCase(engine, @[@[0, 1, 2, 3, 4, 5], @[5, 4, 3, 2, 1, 0]], 6, 24, rng,
    "dispatch every expert once, empty none")
  moeDispatchCase(engine, @[@[0, 0, 0, 1, 1, 2], @[2, 2, 1, 0, 0, 0]], 5, 16, rng,
    "dispatch duplicate experts, expert 3 and 4 empty")
  moeDispatchCase(engine, @[@[0, 1, 2, -1, -1, -1], @[2, -1, 1, -1, 0, -1]], 4, 12, rng,
    "dispatch unused slots")
  moeDispatchCase(engine, @[@[0, 1, 2, 0, 1, 2], @[1, 2, 0, 1, 2, 0]], 3, 8, rng,
    "dispatch topk over E")
  echo "[OK] moe dispatch"

  # Combine: the weighted un-permute sum over the dispatch inverse map
  moeCombineCase(engine, 1, 6, 8, 64, rng, "combine single token")
  moeCombineCase(engine, 8, 6, 64, 48, rng, "combine ragged N")
  moeCombineCase(engine, 4, 3, 5, 130, rng, "combine ragged N edge, one lane partly past")
  echo "[OK] moe gather_add"

  # moeForward end to end: both tier paths against the whole-layer float64 reference
  moeForwardCase(engine, float32, 4, 6, 8, 24, 32, moeSoftmax, 0.0'f32, 1.0'f32, rng,
    "e2e gemv tier softmax")
  moeForwardCase(engine, float32, 4, 6, 8, 24, 32, moeSigmoid, 0.5'f32, 2.5'f32, rng,
    "e2e gemv tier sigmoid scaled")
  moeForwardCase(engine, float32, 32, 6, 8, 24, 32, moeSoftmax, 0.5'f32, 1.0'f32, rng,
    "e2e gemm tier softmax")
  moeForwardCase(engine, float32, 32, 6, 8, 24, 32, moeSigmoid, 0.0'f32, 1.0'f32, rng,
    "e2e gemm tier sigmoid")
  moeForwardCase(engine, float16, 32, 6, 8, 24, 32, moeSoftmax, 0.5'f32, 1.0'f32, rng,
    "e2e gemm tier f16")
  moeForwardGraphCaptureCase(engine, float32)
  echo "[OK] moeForward e2e"

  # Fixture replay
  moeRouterReplayCase(engine, "router_softmax_nobias")
  moeRouterReplayCase(engine, "router_softmax_bias_scale")
  moeRouterReplayCase(engine, "router_sigmoid_bias")
  moeRouterReplayCase(engine, "router_topk_over_e")
  moeDispatchReplayCase(engine, "dispatch_single_token")
  moeDispatchReplayCase(engine, "dispatch_empty_experts")
  moeDispatchReplayCase(engine, "dispatch_topk_over_e")
  moeForwardReplayCase(engine, float32, "e2e_gemm_f32")
  moeForwardReplayCase(engine, float32, "e2e_gemv_f32")
  moeForwardReplayCase(engine, float16, "e2e_gemm_f16")

  # Analytic fuzz
  moeRouterFuzz(engine)
  moeDispatchFuzz(engine)
  moeForwardFuzz(engine)
  echo "[OK] moe fixtures + fuzz"

when isMainModule:
  main()

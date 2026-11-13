## Positron grouped_gemv Metal driver: folded-combine fixture replay, tier and form boundaries, seeded fuzz.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests \
##     --nimcache:nimcache/tests \
##     workspace/positron/tests/k_grouped_via_expertIds/t_kernel_grouped_gemv_metal.nim

import std/random
import workspace/crucible
import grouped_gemv_template

proc main =
  var engine = newEngine(runMetal)
  # Metal simdgroup lanes: fp16 and fp32 operands, both accumulate in fp32
  groupedGemvReplayCase(engine, float32, "grouped_decode_f32", kSplits = 4)
  # kSplits = K boundary: every slice one element wide
  groupedGemvReplayCase(engine, float32, "grouped_decode_f32", kSplits = 64)
  groupedGemvReplayCase(engine, float32, "grouped_sentinel_ragged_f32", kSplits = 4)
  groupedGemvReplayCase(engine, float32, "grouped_wide_f32", kSplits = 4)
  groupedGemvReplayCase(engine, float32, "grouped_single_token_f32", kSplits = 4)
  groupedGemvReplayCase(engine, float16, "grouped_sentinel_ragged_f16", kSplits = 4)
  # Half the contract grid on the block axis must still cover every n-block
  var rng = initRand(20261113)
  undersizedGridCase(engine, float32, 2, 2, 8, 64, 512, rng, "undersized grid")
  undersizedGridCase(engine, float16, 2, 2, 8, 64, 512, rng, "undersized grid")
  fuzzCase(engine)
  echo "[OK] grouped_gemv: fixtures (fp32, fp16), tier boundary, fuzz"

when isMainModule:
  main()

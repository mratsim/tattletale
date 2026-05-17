# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/strformat,
  workspace/libtorch as F,
  workspace/transformers/src/samplers

proc testExtremeLogit*() =
  ## If one logit dominates, Gumbel-max always picks it
  let logits = F.toTensor([[ 0.0, 0.0, 0.0, 1000.0 ]])  # index 3 dominates
  for _ in 0 ..< 1000:
    let idx = logits.sample(temp = 1.0f)
    doAssert idx == 3, &"expected 3, got {idx}"
  echo "✅ extreme logit → always argmax"

proc testLowTempIsGreedy*() =
  ## Low temperature → converges to argmax (greedy)
  let logits = F.toTensor([[ 1.0, 2.5, 0.3, 4.0, 3.2 ]])  # argmax = 3
  let greedy = logits.argmax().item(int)

  for temp in @[0.1f, 0.01f, 0.001f]:
    for _ in 0 ..< 100:
      let idx = logits.sample(temp = temp)
      doAssert idx == greedy, &"temp={temp}: expected {greedy}, got {idx}"
  echo "✅ low temp → matches argmax (greedy)"

proc testDistributionMatch*() =
  ## Gumbel-max produces the SAME distribution as softmax
  let logits = F.toTensor([[ 0.0, 1.0, 2.0 ]])  # softmax ≈ [0.090, 0.245, 0.665]
  let probs = logits.softmax(dim = -1)
  const N = 50_000
  var counts = [0, 0, 0]

  for _ in 0 ..< N:
    let idx = sample(logits, temp = 1.0f)
    inc counts[idx]

  # Check each bin is within ±5% of expected
  for i in 0 .. 2:
    let expected = probs[0, i].item(float) * float(N)
    let observed = float(counts[i])
    let relError = abs(observed - expected) / expected
    doAssert relError < 0.05, &"bin {i}: observed={observed:.0f}, expected={expected:.0f}, error={relError:.4f}"

  echo "✅ distribution matches softmax (±5%)"

proc main*() =
  testExtremeLogit()
  testLowTempIsGreedy()
  testDistributionMatch()
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "All sampler tests passed"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

when isMainModule:
  main()

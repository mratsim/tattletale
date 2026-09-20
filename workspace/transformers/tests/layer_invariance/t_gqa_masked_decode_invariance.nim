# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run through the test_tf_layer_invariance_gqa_masked_decode task in config.nims.
##
## Contract, grouped-query attention with an explicit visibility-band
## mask handling query and key sequences of different lengths.
##
## - the decode shape, one query row against a key-value history longer
##   than the window, exercises the kv-expansion spelling at kvLen != qLen
## - the reference comparison locks the head order of the expansion
##   against an independent index_select spelling
##
## Scope note, this suite does not guard the sliding-window decode
## truncation branch, the decode-truncation contract is fixture-verified
## through the tier suites against the reference stack

import
  std/options,
  std/strformat,
  workspace/libtorch as F,
  workspace/transformers/src/layers/attn_ssm/grouped_query_attention,
  workspace/transformers/tests/layer_utils

proc main() =
  let dev = testDevice()
  echo &"GQA masked decode invariance, device {dev}"

  let gqa = GroupedQueryAttention.init(
    num_qo_head = 8, num_kv_head = 1, head_dim = 128)

  # Decode shape, one query row against a key-value history of 556
  # rows. The window 512 binds, so the reference dispatches the masked
  # decode path here.
  let qLen = 1
  let kvLen = 556
  let q = setupStimulusTensor(qLen, 8, 128, 0.0'f32, dev)
  let k = setupStimulusTensor(kvLen, 1, 128, 0.5'f32, dev)
  let v = setupStimulusTensor(kvLen, 1, 128, 1.0'f32, dev)
  let mask = windowedCausalMask(qLen, kvLen, kvLen - 1, 512, dev)

  let gqaOut = gqa.forward(q, k, v,
    is_causal = false, attn_mask = some(mask), enable_gqa = true)
  if gqaOut.size(0) != 1 or gqaOut.size(1) != qLen or gqaOut.size(2) != 8 * 128:
    echo &"FAIL: masked decode output shape is ({gqaOut.size(0)}, " &
      &"{gqaOut.size(1)}, {gqaOut.size(2)}), expected (1, 1, 1024)"
    quit(1)

  # Reference spelling, each kv head repeated for its query-head group
  # through an independent index_select and the same SDPA call
  let groupIdx = block:
    var idx = newSeq[int64](8)
    for g in 0 ..< 8:
      idx[g] = 0'i64
    idx.toTensor().to(dev)
  let kRep = k.permute([0, 2, 1, 3]).index_select(1, groupIdx)
  let vRep = v.permute([0, 2, 1, 3]).index_select(1, groupIdx)
  let qAttn = q.permute([0, 2, 1, 3])
  let refOut = F.scaled_dot_product_attention(qAttn, kRep, vRep,
    attn_mask = some(mask), dropout_p = 0.0'f64, is_causal = false,
    scale = some(gqa.softmax_scale), enable_gqa = false)
    .permute([0, 2, 1, 3]).reshape([1, qLen, 8 * 128])

  let diff = (gqaOut.to(F.kFloat32) - refOut.to(F.kFloat32)).abs()
  let worst = diff.max().item(float32)
  # Both spellings feed the same SDPA kernel value-identical copies so
  # the outputs agree with no drift budget at all
  if worst != 0.0'f32:
    echo &"FAIL: masked decode output diverges from the index_select " &
      &"reference, maxabs {worst}"
    quit(1)

  # Prefill shape under the band, qLen == kvLen and the expansion
  # sizes both dims from the key length, this must keep working.
  let qPre = setupStimulusTensor(556, 8, 128, 0.0'f32, dev)
  let kPre = setupStimulusTensor(556, 1, 128, 0.5'f32, dev)
  let vPre = setupStimulusTensor(556, 1, 128, 1.0'f32, dev)
  let maskPre = windowedCausalMask(556, 556, 0, 512, dev)
  let outPre = gqa.forward(qPre, kPre, vPre,
    is_causal = false, attn_mask = some(maskPre), enable_gqa = true)
  if outPre.size(1) != 556 or outPre.size(2) != 8 * 128:
    echo &"FAIL: masked prefill output shape is ({outPre.size(0)}, " &
      &"{outPre.size(1)}, {outPre.size(2)}), expected (1, 556, 1024)"
    quit(1)

when isMainModule:
  main()

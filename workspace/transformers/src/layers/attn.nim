# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/math,
  std/options,
  workspace/libtorch as F,
  workspace/libtorch/src/abi/neural_nets,
  workspace/libtorch/src/torch_tensors_sugar,
  workspace/libtorch/src/torch_tensors_overloads,
  workspace/transformers/src/layers/linear,
  workspace/transformers/src/layers/norm,
  ./rope

type
  GroupedQueryAttention* = object
    head_dim*: int
    num_qo_head*: int
    num_kv_head*: int
    num_kv_groups*: int
    qo_attn_dim*: int
    kv_attn_dim*: int
    softmax_scale*: float64

  KVCache* = object
    keys*: TorchTensor
    values*: TorchTensor

  RopeGQAttention* = object
    q_proj*: Linear
    k_proj*: Linear
    v_proj*: Linear
    o_proj*: Linear
    attn*: GroupedQueryAttention
    rotary*: RotaryPositionEmbedding
    q_norm*: Option[RmsNorm]
    k_norm*: Option[RmsNorm]
    kv_cache*: KVCache

func init*(_: type GroupedQueryAttention, num_qo_head, num_kv_head, head_dim: int): GroupedQueryAttention =
  let num_kv_groups = num_qo_head div num_kv_head
  GroupedQueryAttention(
    head_dim: head_dim,
    num_qo_head: num_qo_head,
    num_kv_head: num_kv_head,
    num_kv_groups: num_kv_groups,
    qo_attn_dim: num_qo_head * head_dim,
    kv_attn_dim: num_kv_head * head_dim,
    softmax_scale: 1.0'f64 / sqrt(head_dim.float64)
  )

func forward*(
  self: GroupedQueryAttention,
  q: TorchTensor,
  k: TorchTensor,
  v: TorchTensor,
  is_causal: bool = true,
  attn_mask = none(TorchTensor),
  dropout_p = 0.0'f64
): TorchTensor =
  # Backend: permute to (batch, head, seq, head_dim), ensure dtype, SDPA, reshape
  # Input q,k,v: (batch, seq, num_head, head_dim)
  let batch = q.size(0)
  let seq_len = q.size(1)
  
  var q_attn = q.permute([0, 2, 1, 3])
  var k_attn = k.permute([0, 2, 1, 3])
  let v_attn = v.permute([0, 2, 1, 3])
  
  let target_dtype = v_attn.scalarType()
  let q_final = q_attn.to(target_dtype)
  let k_final = k_attn.to(target_dtype)
  
  let attn_out = F.scaled_dot_product_attention(
    q_final, k_final, v_attn,
    attn_mask = attn_mask,
    dropout_p = dropout_p,
    is_causal = is_causal,
    scale = some(self.softmax_scale),
    enable_gqa = self.num_kv_groups > 1
  )
  
  let attn_perm = attn_out.permute([0, 2, 1, 3])
  result = attn_perm.reshape([batch, seq_len, self.qo_attn_dim])

func init*(_: type KVCache): KVCache =
  KVCache()

proc reset*(self: var KVCache) =
  self.keys = F.empty(0)
  self.values = F.empty(0)

proc append*(self: var KVCache, k, v: TorchTensor): (TorchTensor, TorchTensor) =
  if self.keys.numel == 0:
    self.keys = k
    self.values = v
  else:
    self.keys = F.cat([self.keys, k], 1)
    self.values = F.cat([self.values, v], 1)
  (self.keys, self.values)

func init*(
  _: type RopeGQAttention,
  q_weight, k_weight, v_weight, o_weight, q_norm_weight, k_norm_weight: TorchTensor,
  num_qo_head, num_kv_head, head_dim: int,
  rotary: RotaryPositionEmbedding,
  rms_norm_eps = 1e-6'f64
): RopeGQAttention =
  let q_proj = Linear.init(q_weight)
  let k_proj = Linear.init(k_weight)
  let v_proj = Linear.init(v_weight)
  let o_proj = Linear.init(o_weight)

  let has_qk_norm = rotary.head_dim == head_dim
  let norm_dtype = kBFloat16
  let q_norm =
    if has_qk_norm: some(RmsNorm.init(weight = q_norm_weight.to(norm_dtype), eps = rms_norm_eps))
    else: none(RmsNorm)
  let k_norm =
    if has_qk_norm: some(RmsNorm.init(weight = k_norm_weight.to(norm_dtype), eps = rms_norm_eps))
    else: none(RmsNorm)

  let attn = GroupedQueryAttention.init(
    num_qo_head = num_qo_head,
    num_kv_head = num_kv_head,
    head_dim = head_dim
  )

  RopeGQAttention(
    q_proj: q_proj,
    k_proj: k_proj,
    v_proj: v_proj,
    o_proj: o_proj,
    attn: attn,
    rotary: rotary,
    q_norm: q_norm,
    k_norm: k_norm,
    kv_cache: KVCache.init()
  )

proc reset_cache*(self: var RopeGQAttention) =
  self.kv_cache.reset()

proc forward*(
  self: var RopeGQAttention,
  x: TorchTensor,
  rope_offset: int
): TorchTensor =
  # Use separate Q, K, V projections (matching HF/Qwen3)
  let q = self.q_proj.forward(x)
  var k_new = self.k_proj.forward(x)
  var v_new = self.v_proj.forward(x)
   
  let batch = x.size(0)
  let seq_len = x.size(1)
 
  # KV cache transformation:
  # - Prefill: cache is empty, we append new KV, returning (k_new, v_new)
  # - Decode: cache has prior KV, we append new KV, returning full concatenated KV
  (k_new, v_new) = self.kv_cache.append(k_new, v_new)
 
  # Reshape to (batch, seq, heads, head_dim)
  # Note: for decode, k_new/v_new now has seq_len = 1 + cache_size
  let q_reshaped = q.reshape([batch, seq_len, self.attn.num_qo_head, self.attn.head_dim])
  let k_reshaped = k_new.reshape([batch, k_new.size(1), self.attn.num_kv_head, self.attn.head_dim])
  let v_reshaped = v_new.reshape([batch, v_new.size(1), self.attn.num_kv_head, self.attn.head_dim])
 
  # Apply q/k norm (on reshaped tensor before permute)
  var q_norm_input = q_reshaped
  var k_norm_input = k_reshaped
  if self.q_norm.isSome:
    q_norm_input = self.q_norm.get().forward(q_reshaped)
  if self.k_norm.isSome:
    k_norm_input = self.k_norm.get().forward(k_reshaped)
 
  # Apply RoPE using the rotary cache with offset into the cache
  let (q_rot, k_rot) = self.rotary.apply_rope(q_norm_input, k_norm_input, rope_offset)
 
  # Pass to backend (GroupedQueryAttention) which handles permute/dtype/SDPA/reshape
  let attn_out_reshaped = self.attn.forward(q_rot, k_rot, v_reshaped, is_causal = true)
  result = self.o_proj.forward(attn_out_reshaped)

proc forward*(
  self: var RopeGQAttention,
  x: TorchTensor,
  cos: TorchTensor,
  sin: TorchTensor
): TorchTensor =
  # For testing: use pre-computed cos/sin from fixture
  let q = self.q_proj.forward(x)
  let k_new = self.k_proj.forward(x)
  let v_new = self.v_proj.forward(x)
   
  let batch = x.size(0)
  let seq_len = x.size(1)
 
  let q_reshaped = q.reshape([batch, seq_len, self.attn.num_qo_head, self.attn.head_dim])
  let k_reshaped = k_new.reshape([batch, k_new.size(1), self.attn.num_kv_head, self.attn.head_dim])
  let v_reshaped = v_new.reshape([batch, v_new.size(1), self.attn.num_kv_head, self.attn.head_dim])
 
  var q_norm_input = q_reshaped
  var k_norm_input = k_reshaped
  if self.q_norm.isSome:
    q_norm_input = self.q_norm.get().forward(q_reshaped)
  if self.k_norm.isSome:
    k_norm_input = self.k_norm.get().forward(k_reshaped)
 
  let (q_rot, k_rot) = apply_rope_impl(q_norm_input, k_norm_input, cos, sin)
  
  # Pass to backend (GroupedQueryAttention) which handles permute/dtype/SDPA/reshape
  let attn_out_reshaped = self.attn.forward(q_rot, k_rot, v_reshaped, is_causal = true)
  result = self.o_proj.forward(attn_out_reshaped)

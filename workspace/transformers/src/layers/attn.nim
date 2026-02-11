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
  workspace/transformers/src/layers/rmsnorm

type
  RotaryConfig* = object
    head_dim*: int
    rotary_dim*: int
    max_position*: int
    base*: float
    scaling_cached*: TorchTensor

func init*(_: type RotaryConfig, head_dim, rotary_dim, max_position: int, base: float): RotaryConfig =
  RotaryConfig(head_dim: head_dim, rotary_dim: rotary_dim, max_position: max_position, base: base)

proc apply_rope*(self: RotaryConfig, q, k: TorchTensor, positions: TorchTensor): (TorchTensor, TorchTensor) =
  result = (q, k)

type
  MultiHeadAttention* = object
    layer_id*: int
    head_dim*: int
    num_qo_heads*: int
    num_kv_heads*: int
    num_kv_groups*: int
    qo_attn_dim*: int
    kv_attn_dim*: int
    rotary*: RotaryConfig
    q_norm*: Option[RmsNorm]
    k_norm*: Option[RmsNorm]

func init*(_: type MultiHeadAttention, layer_id, num_qo_heads, num_kv_heads, head_dim: int, rotary: RotaryConfig, q_norm, k_norm: Option[RmsNorm]): MultiHeadAttention =
  let num_kv_groups = num_qo_heads div num_kv_heads
  MultiHeadAttention(
    layer_id: layer_id,
    head_dim: head_dim,
    num_qo_heads: num_qo_heads,
    num_kv_heads: num_kv_heads,
    num_kv_groups: num_kv_groups,
    qo_attn_dim: num_qo_heads * head_dim,
    kv_attn_dim: num_kv_heads * head_dim,
    rotary: rotary,
    q_norm: q_norm,
    k_norm: k_norm
  )

proc forward*(self: MultiHeadAttention, q, k, v: TorchTensor, positions: TorchTensor): TorchTensor =
  let batch = q.size(0)
  let seq_len = q.size(1)

  let q_reshaped = q.reshape([batch, seq_len, self.num_qo_heads, self.head_dim])
  let k_reshaped = k.reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
  let v_reshaped = v.reshape([batch, seq_len, self.num_kv_heads, self.head_dim])

  if self.q_norm.isSome:
    discard self.q_norm.get().forward(q_reshaped)
  if self.k_norm.isSome:
    discard self.k_norm.get().forward(k_reshaped)

  let q_attn = q_reshaped.permute([1, 0, 2, 3])
  let k_attn = k_reshaped.permute([1, 0, 2, 3])
  let v_attn = v_reshaped.permute([1, 0, 2, 3])

  let (q_rot, k_rot) = self.rotary.apply_rope(q_attn, k_attn, positions)

  var attn_out: TorchTensor
  if self.num_kv_groups > 1:
    let k_expanded = k_attn.repeat_interleave(self.num_kv_groups, 0)
    let v_expanded = v_attn.repeat_interleave(self.num_kv_groups, 0)
    attn_out = F.scaled_dot_product_attention(
      q_attn, k_expanded, v_expanded, is_causal = true, enable_gqa = true
    )
  else:
    attn_out = F.scaled_dot_product_attention(
      q_attn, k_attn, v_attn, is_causal = true
    )

  result = attn_out.permute([1, 0, 2, 3]).reshape([batch, seq_len, self.qo_attn_dim])

type
  KVCache* = object
    keys*: TorchTensor
    values*: TorchTensor

func init*(_: type KVCache): KVCache =
  KVCache()

proc append*(self: var KVCache, k, v: TorchTensor): (TorchTensor, TorchTensor) =
  self.keys = F.cat([self.keys, k], 1)
  self.values = F.cat([self.values, v], 1)
  result = (self.keys, self.values)

proc reset*(self: var KVCache) =
  self.keys = F.init(TorchTensor)
  self.values = F.init(TorchTensor)

type
  RopeMHAttention* = object
    qkv*: Linear
    o_proj*: Linear
    attn*: MultiHeadAttention
    kv_cache*: KVCache

func init*(_: type RopeMHAttention, q_weight, k_weight, v_weight, o_weight: TorchTensor, num_qo_heads, num_kv_heads, head_dim: int, rotary: RotaryConfig, rms_norm_eps: float = 1e-6): RopeMHAttention =
  let qkv_fused = F.cat([q_weight, k_weight, v_weight], 0)
  let qkv = Linear.init(qkv_fused)
  let o_proj = Linear.init(o_weight)

  let has_qk_norm = rotary.rotary_dim == head_dim
  let q_norm: Option[RmsNorm] = if has_qk_norm: some(RmsNorm.init(weight = F.ones([head_dim], kFloat32), eps = rms_norm_eps)) else: none[RmsNorm]
  let k_norm: Option[RmsNorm] = if has_qk_norm: some(RmsNorm.init(weight = F.ones([head_dim], kFloat32), eps = rms_norm_eps)) else: none[RmsNorm]

  let attn = MultiHeadAttention.init(
    layer_id = 0,
    num_qo_heads = num_qo_heads,
    num_kv_heads = num_kv_heads,
    head_dim = head_dim,
    rotary = rotary,
    q_norm = q_norm,
    k_norm = k_norm
  )

  RopeMHAttention(
    qkv: qkv,
    o_proj: o_proj,
    attn: attn,
    kv_cache: KVCache.init()
  )

func reset_cache*(self: var RopeMHAttention) =
  self.kv_cache.reset()

proc forward*(self: var RopeMHAttention, x: TorchTensor, positions: TorchTensor, use_cache: bool): TorchTensor =
  let qkv_out = self.qkv.forward(x)
  let batch = x.size(0)
  let seq_len = x.size(1)
  let qo_dim = self.attn.qo_attn_dim
  let kv_dim = self.attn.kv_attn_dim

  let q = qkv_out[0..<qo_dim]
  let k_new = qkv_out[qo_dim..<qo_dim+kv_dim]
  let v_new = qkv_out[qo_dim+kv_dim..^1]

  var k_full: TorchTensor
  var v_full: TorchTensor

  if use_cache:
    (k_full, v_full) = self.kv_cache.append(k_new, v_new)
  else:
    k_full = k_new
    v_full = v_new
    self.kv_cache.keys = F.init(TorchTensor)
    self.kv_cache.values = F.init(TorchTensor)

  let q_reshaped = q.reshape([batch, seq_len, self.attn.num_qo_heads, self.attn.head_dim])
  let k_reshaped = k_full.reshape([batch, k_full.size(1), self.attn.num_kv_heads, self.attn.head_dim])
  let v_reshaped = v_full.reshape([batch, v_full.size(1), self.attn.num_kv_heads, self.attn.head_dim])

  let q_attn = q_reshaped.permute([1, 0, 2, 3])
  let k_attn = k_reshaped.permute([1, 0, 2, 3])
  let v_attn = v_reshaped.permute([1, 0, 2, 3])

  discard self.attn.rotary.apply_rope(q_attn, k_attn, positions)

  var attn_out: TorchTensor
  if self.attn.num_kv_groups > 1:
    let k_expanded = k_attn.repeat_interleave(self.attn.num_kv_groups, 0)
    let v_expanded = v_attn.repeat_interleave(self.attn.num_kv_groups, 0)
    attn_out = F.scaled_dot_product_attention(
      q_attn, k_expanded, v_expanded, is_causal = true, enable_gqa = true
    )
  else:
    attn_out = F.scaled_dot_product_attention(
      q_attn, k_attn, v_attn, is_causal = true
    )

  let attn_flat = attn_out.permute([1, 0, 2, 3]).reshape([batch * seq_len, self.attn.qo_attn_dim])
  result = self.o_proj.forward(attn_flat)

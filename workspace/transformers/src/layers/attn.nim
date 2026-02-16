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
    qkv*: Linear
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
  let enable_gqa = self.num_kv_groups > 1
  var key = k
  var value = v
  if enable_gqa:
    key = k.repeat_interleave(self.num_kv_groups, 1)
    value = v.repeat_interleave(self.num_kv_groups, 1)
  F.scaled_dot_product_attention(
    q, key, value,
    attn_mask = attn_mask,
    dropout_p = dropout_p,
    is_causal = is_causal,
    scale = some(self.softmax_scale),
    enable_gqa = enable_gqa
  )

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
  q_weight, k_weight, v_weight, o_weight: TorchTensor,
  num_qo_head, num_kv_head, head_dim: int,
  rotary: RotaryPositionEmbedding,
  rms_norm_eps = 1e-6'f64
): RopeGQAttention =
  let qkv_fused = F.cat([q_weight, k_weight, v_weight], 0)
  let qkv = Linear.init(qkv_fused)
  let o_proj = Linear.init(o_weight)

  let has_qk_norm = rotary.head_dim == head_dim
  let q_norm =
    if has_qk_norm: some(RmsNorm.init(weight = F.ones([head_dim], kFloat32), eps = rms_norm_eps))
    else: none(RmsNorm)
  let k_norm =
    if has_qk_norm: some(RmsNorm.init(weight = F.ones([head_dim], kFloat32), eps = rms_norm_eps))
    else: none(RmsNorm)

  let attn = GroupedQueryAttention.init(
    num_qo_head = num_qo_head,
    num_kv_head = num_kv_head,
    head_dim = head_dim
  )

  RopeGQAttention(
    qkv: qkv,
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
  position: TorchTensor,
  use_cache: bool
): TorchTensor =
  let qkv_out = self.qkv.forward(x)
  # doAssert qkv_out.shape.asNimView() == @[2, 8, 4096], "qkv_out shape"
  let batch = x.size(0)
  let seq_len = x.size(1)

  # Input: qkv_out is (batch, seq, qo_attn_dim + 2*kv_attn_dim)
  # Shape: (2, 8, 2048 + 2*1024) = (2, 8, 4096)
  # For 3D tensor: t[_, _, start..<stop] to slice last dim only
  let q = qkv_out[_, _, 0..<self.attn.qo_attn_dim]     # (2, 8, 2048)
  let k_new = qkv_out[_, _, self.attn.qo_attn_dim..<self.attn.qo_attn_dim + self.attn.kv_attn_dim]  # (2, 8, 1024)
  let v_new = qkv_out[_, _, self.attn.qo_attn_dim + self.attn.kv_attn_dim.._]  # (2, 8, 1024)
  # Expected: q,k_new,v_new = (batch, seq, qo_attn_dim/kv_attn_dim) = (2, 8, 2048/1024)
  doAssert q.size(0) == batch and q.size(1) == seq_len and q.size(2) == self.attn.qo_attn_dim, "q slice shape"
  doAssert k_new.size(0) == batch and k_new.size(1) == seq_len and k_new.size(2) == self.attn.kv_attn_dim, "k_new slice shape"
  doAssert v_new.size(0) == batch and v_new.size(1) == seq_len and v_new.size(2) == self.attn.kv_attn_dim, "v_new slice shape"

  var k_full: TorchTensor
  var v_full: TorchTensor

  if use_cache:
    (k_full, v_full) = self.kv_cache.append(k_new, v_new)
  else:
    k_full = k_new
    v_full = v_new
    self.kv_cache.reset()

  # Reshape to (batch, seq, heads, head_dim)
  let q_reshaped = q.reshape([batch, seq_len, self.attn.num_qo_head, self.attn.head_dim])   # (2, 8, 16, 128)
  let k_reshaped = k_full.reshape([batch, k_full.size(1), self.attn.num_kv_head, self.attn.head_dim])  # (2, 8, 8, 128)
  let v_reshaped = v_full.reshape([batch, v_full.size(1), self.attn.num_kv_head, self.attn.head_dim])  # (2, 8, 8, 128)

  doAssert q_reshaped.shape.asNimView() == @[batch, seq_len, self.attn.num_qo_head, self.attn.head_dim], "q_reshaped shape"
  doAssert k_reshaped.shape.asNimView() == @[batch, k_full.size(1), self.attn.num_kv_head, self.attn.head_dim], "k_reshaped shape"
  doAssert v_reshaped.shape.asNimView() == @[batch, v_full.size(1), self.attn.num_kv_head, self.attn.head_dim], "v_reshaped shape"

  # Permute to (batch, head, seq, head_dim) for SDPA: (0,2,1,3)
  var q_attn = q_reshaped.permute([0, 2, 1, 3])  # (2, 16, 8, 128)
  var k_attn = k_reshaped.permute([0, 2, 1, 3])  # (2, 8, 8, 128)
  let v_attn = v_reshaped.permute([0, 2, 1, 3])  # (2, 8, 8, 128)

  doAssert q_attn.shape.asNimView() == @[batch, self.attn.num_qo_head, seq_len, self.attn.head_dim], "q_attn shape"
  doAssert k_attn.shape.asNimView() == @[batch, self.attn.num_kv_head, k_full.size(1), self.attn.head_dim], "k_attn shape"
  doAssert v_attn.shape.asNimView() == @[batch, self.attn.num_kv_head, v_full.size(1), self.attn.head_dim], "v_attn shape"

  if self.q_norm.isSome:
    q_attn = self.q_norm.get().forward(q_attn)
  if self.k_norm.isSome:
    k_attn = self.k_norm.get().forward(k_attn)

  let pos_offset = position[0, 0].item(int)  # First batch, first position as offset
  let (q_rot, k_rot) = self.rotary.apply_rope(q_attn, k_attn, pos_offset)

  # Ensure q, k, v have same dtype (v may be bfloat16 from safetensors weights)
  let target_dtype = v_attn.scalarType()
  let q_final = q_rot.to(target_dtype)
  let k_final = k_rot.to(target_dtype)

  # SDPA output: (batch, head, seq, head_dim) -> permute back to (batch, seq, head*head_dim)
  # Expected: attn_out = (2, 16, 8, 128), attn_perm = (2, 8, 16, 128), attn_out_reshaped = (2, 8, 2048)
  let attn_out = self.attn.forward(q_final, k_final, v_attn, is_causal = true)
  let attn_perm = attn_out.permute([0, 2, 1, 3])
  let attn_out_reshaped = attn_perm.reshape([batch, seq_len, self.attn.qo_attn_dim])
  result = self.o_proj.forward(attn_out_reshaped)

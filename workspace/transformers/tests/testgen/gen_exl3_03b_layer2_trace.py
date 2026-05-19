#!/usr/bin/env python3
"""Generate step-by-step layer 02 trace fixture for EXL3."""
import torch, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from safetensors.torch import load_file as st_load, save_file as st_save
from q_exl3_common import reconstruct_orig_exl3
from exllamav3.ext import exllamav3_ext as ext

DEVICE = 'cuda:0'
BASE = os.path.join(os.path.dirname(__file__), '..')
FIXTURE_DIR = os.path.join(BASE, 'fixtures', 'exl3-layer02-trace')

def rms(x, w, eps=1e-6):
    x2=x.reshape(-1,x.shape[-1]); y=torch.empty_like(x2)
    ext.rms_norm(x2,w,y,eps,0.0,1.0,False,False)
    return y.reshape(x.shape)

def hp(x,w,suh,svh):
    b,s=x.shape[0],x.shape[1]; x2=x.reshape(-1,x.shape[-1]); o2=torch.empty_like(x2)
    ext.had_r_128(x2,o2,suh,None,1.0)
    g2=torch.empty(x2.shape[0],w.shape[1],dtype=x.dtype,device=x.device)
    ext.hgemm(o2,w,g2); h=torch.empty_like(g2)
    ext.had_r_128(g2,h,None,svh,1.0)
    return h.reshape(b,s,-1)

def build_proj(pk):
    t=raw.get(f'{pk}.trellis'); K=int(t.shape[2])*16//256; kt,nt,_=t.shape
    mcg=raw.get(f'{pk}.mcg') is not None; mul=raw.get(f'{pk}.mul1') is not None
    return (reconstruct_orig_exl3(t,K,mcg,mul,(kt*16,nt*16)),
            raw.get(f'{pk}.suh'), raw.get(f'{pk}.svh'))

raw=st_load(os.path.join(BASE,'hf_models','Qwen3-0.6B-EXL3-5bpw','model.safetensors'),device=DEVICE)
fix=st_load(os.path.join(BASE,'fixtures','exl3-ids-inference','Qwen3-0.6B-EXL3-5bpw','layer-02.safetensor'),device=DEVICE)
x=fix['layer_input']; b,s,d=x.shape

P='model.layers.2'
w_q,suh_q,svh_q=build_proj(f'{P}.self_attn.q_proj')
w_k,suh_k,svh_k=build_proj(f'{P}.self_attn.k_proj')
w_v,suh_v,svh_v=build_proj(f'{P}.self_attn.v_proj')
w_o,suh_o,svh_o=build_proj(f'{P}.self_attn.o_proj')
w_g,suh_g,svh_g=build_proj(f'{P}.mlp.gate_proj')
w_u,suh_u,svh_u=build_proj(f'{P}.mlp.up_proj')
w_d,suh_d,svh_d=build_proj(f'{P}.mlp.down_proj')
ln_w=raw.get(f'{P}.input_layernorm.weight')
pln_w=raw.get(f'{P}.post_attention_layernorm.weight')
qn_w=raw.get(f'{P}.self_attn.q_norm.weight')
kn_w=raw.get(f'{P}.self_attn.k_norm.weight')

t={}
t['input_hidden_states']=x.clone()

# Step 1: input_layernorm
h=rms(x,ln_w); t['after_input_layernorm']=h.clone()

# Step 2: QKV
q=hp(h,w_q,suh_q,svh_q); k=hp(h,w_k,suh_k,svh_k); v=hp(h,w_v,suh_v,svh_v)
t['q_proj_out']=q.clone(); t['k_proj_out']=k.clone(); t['v_proj_out']=v.clone()
q=q.reshape(b,s,16,128); k=k.reshape(b,s,8,128); v_h=v.reshape(b,s,8,128)

# Step 3: QK norms
q=rms(q,qn_w); k=rms(k,kn_w)
t['after_q_norm']=q.reshape(b,s,-1).clone(); t['after_k_norm']=k.reshape(b,s,-1).clone()

# Step 4: RoPE
hd=128; inv=1.0/(1e6**(torch.arange(0,hd,2,device=DEVICE).float()/hd))
pos=torch.arange(s,device=DEVICE).float()[:,None]
ang=pos*inv[None,:]
cos=torch.cat([ang.cos(),ang.cos()],-1).half(); sin=torch.cat([ang.sin(),ang.sin()],-1).half()
cos4=cos.unsqueeze(0).unsqueeze(0); sin4=sin.unsqueeze(0).unsqueeze(0)
qt=q.transpose(1,2); kt=k.transpose(1,2)
def rh(x): h2=x.shape[-1]//2; return torch.cat([-x[...,h2:],x[...,:h2]],-1)
qr=qt*cos4+rh(qt)*sin4; kr=kt*cos4+rh(kt)*sin4
t['after_rope_q']=qr.transpose(1,2).reshape(b,s,-1).clone()
t['after_rope_k']=kr.transpose(1,2).reshape(b,s,-1).clone()

# Step 5: SDPA
attn=torch.nn.functional.scaled_dot_product_attention(qr,kr,v_h.transpose(1,2),is_causal=True,scale=hd**-0.5,enable_gqa=True)
t['attn_output']=attn.transpose(1,2).reshape(b,s,-1).clone()

# Step 6: O projection
ao=attn.transpose(1,2).reshape(b,s,-1)
o=hp(ao,w_o,suh_o,svh_o); t['after_o_proj']=o.clone()

# Step 7: Residual
res=x+o; t['after_residual']=res.clone()

# Step 8: Post-attention RMSNorm
h2=rms(res,pln_w); t['after_post_layernorm']=h2.clone()

# Step 9: MLP
gate=hp(h2,w_g,suh_g,svh_g); up=hp(h2,w_u,suh_u,svh_u)
t['mlp_gate_out']=gate.clone(); t['mlp_up_out']=up.clone()
act=torch.nn.functional.silu(gate)*up; t['mlp_activation']=act.clone()
down=hp(act,w_d,suh_d,svh_d); t['mlp_down_out']=down.clone()

# Step 10: Long residual stream outputs
t['output']=down.clone(); t['output_residual']=res.clone()

# Verify
y=down+res; expected=fix['layer_output']
print(f'Final vs fixture: {(y-expected).abs().max().item():.6f}')

os.makedirs(FIXTURE_DIR,exist_ok=True)
path=os.path.join(FIXTURE_DIR,'layer02_trace.safetensor')
st_save(t, path)
print(f'Saved: {path}')
for k,v in t.items(): print(f'  {k}: {v.shape}')

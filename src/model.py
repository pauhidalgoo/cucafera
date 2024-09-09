"""
This is the cucafera model. The initial version of this file contained some error (we couldn't find exactly where, we suspect it was a problem with 
the dimensions when computing the attention), so this one is based on the transformers library code.

https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

It uses the improvements from the LLAMA3 series.
It has:
- GQA
- GeGLU * (llama uses SwiGLU)
- RoPE

I still would need to implement KV-caching to improve inference type.

# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import math
import inspect

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


"""
ORIGINAL BAD SELF-ATTENTION (the RoPE was also different)
class CausalSelfAttentionGQA(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.head_dim = config.n_embd // config.n_head
        self.n_kv_heads = config.n_kv_heads # Nombre de grups de query
        self.n_head = config.n_head

        shape = (config.n_head + 2 * config.n_kv_heads) * self.head_dim

        self.wq = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False )
        self.wk = nn.Linear(config.n_embd, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.n_embd, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)
        self.cache = None
        self.queries_per_kv = self.n_head // self.n_kv_heads

    def forward(self, x, freqs_cis, mask, return_attention=False):
        B, T, C = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(B, T, self.n_head, self.head_dim)
        xk = xk.view(B, T, self.n_kv_heads, self.head_dim)
        xv = xv.view(B, T, self.n_kv_heads, self.head_dim)

        xq = apply_rope(xq, freqs_cis)
        xk = apply_rope(xk, freqs_cis)

        xk = repeat_kv(xk, self.queries_per_kv)
        xv = repeat_kv(xv, self.queries_per_kv)
        print("aquí si")
        
        xq, xk, xv = (x.transpose(1, 2) for x in (xq, xk, xv))

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        # output projection
        proj = self.wo(output)
        if return_attention:
            return proj, scores
        return proj
"""


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CausalSelfAttentionGQA(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.config = config
        self.num_heads = config.n_head
        self.num_kv_heads = config.n_kv_heads
        self.head_dim = config.n_embd // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        

        self.q_proj = nn.Linear(config.n_embd, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.n_embd, bias=False)


        #self.sliding_window_size = config.sliding_window_size     initially we also wanted to try the alternate sliding window from Gemma
        self.max_seq_len = config.block_size

    def forward(self, x, cos_sin: tuple, return_attention = None):
        B, T, C = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1,2)

        cos, sin = cos_sin
        q, k = apply_rope(q, k, cos, sin)

        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        scores = torch.matmul(q, k.transpose(2,3)) / math.sqrt(self.head_dim)

        mask = True
        if mask is not None:  # the mask is not correct, it needs to be [B, C, T, T]
            mask = torch.full((T, T), float("-inf"), device=x.device)
            mask = torch.triu(mask, diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(1)
            scores = scores + mask

        scores = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(scores, v)

        if output.size() != (B, self.num_heads, T, self.head_dim):
            raise ValueError(f"ALGO HA ANAT MALAMENT, output té dimensions {output.size()}")
        
        output = output.transpose(1,2).contiguous()
        output = output.reshape(B, T, -1)

        output = self.o_proj(output)

        if return_attention:
            return output, scores
        return output, None
        

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d)) # weight

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        hidden_states = x * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


def compute_rope_default(config, device):
    base = config.rope_theta
    dim = config.n_embd // config.n_head

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device)/dim))
    return inv_freq, 1.0



class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config, device = None):
        super().__init__()
        self.max_position_embeddings = config.max_position_embeddings
        self.factor = config.scaling_factor
        self.base = config.rope_theta
        self.dim = config.n_embd // config.n_head
        self.config = config
        self.rope_init_fn = compute_rope_default

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
    
    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

        

class MLP(nn.Module):
    """
    class GeGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        # Silu és el mateix que swift function
        return F.gelu(gate) * x
    
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate_proj   = nn.Linear(config.n_embd, config.intermediate_size, bias=False) # gate_proj
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False) # down proj
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False) # up proj
    
    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh")* self.up_proj(x)) 


class Block(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.n_embd, config.norm_eps) # input_layernorm
        self.self_attn = CausalSelfAttentionGQA(config)
        self.post_attention_layernorm = RMSNorm(config.n_embd, config.norm_eps) # post_attention_layernorm
        self.mlp = MLP(config)
    
    def forward(self, x, cos_sin):
        residual = x
        x = self.input_layernorm(x)
        x, attn_weights = self.self_attn(x, cos_sin)

        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x

class Cucafera(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


        self.padding_idx = config.pad_token_id
        self.embed_tokens =nn.Embedding(config.vocab_size, config.n_embd, self.padding_idx)

        self.layers = nn.ModuleList([Block(config) for i in range(config.n_layer)])

        self.norm =  RMSNorm(config.n_embd, config.norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.embed_tokens.weight = self.lm_head.weight # Linkeddddd

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean = 0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is smaller"

        x = self.embed_tokens(idx)


        position_ids = torch.arange(T, device = x.device).unsqueeze(0)

        


        cos_sin = self.rotary_emb(x, position_ids)

        for block in self.layers:
            x = block(x, cos_sin)
        
        x = self.norm(x)

        logits = self.lm_head(x).float()

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss
    
    def configure_optimizers(self, learning_rate, weight_decay=0.1, betas=(0.9, 0.95), device_type='cuda'):
        # From Andrej Karpathy's NanoGPT
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=1e-8, fused=use_fused)
        return optimizer


@dataclass
class CucaferaConfig:
    block_size: int = 2048
    vocab_size: int = 65536
    n_layer: int = 30
    n_head: int = 8
    n_embd: int = 768 # hidden_size
    intermediate_size = 2048 # 3072
    n_kv_heads: int = 4 # nombre de grups de query
    norm_eps: int = 1e-05
    rope_theta: float = 10000.0
    use_scaled_rope: bool = False
    scaling_factor: int = 1
    max_batch_size: int = 16
    max_seq_len:int = 2048
    max_position_embeddings:int = 2048
    pad_token_id:int = 3

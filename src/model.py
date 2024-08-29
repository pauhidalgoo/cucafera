"""
This is THE model.

It combines improvements from the LLAMA3 series, with others from Gemma2.
It has:
- GQA
- Sliding window attention every two layers
- GeGLU (thinking to maybe use SwiGLU or ReGLU)
- RoPE

I still would need to implement KV-caching to improve inference type.
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class CausalSelfAttentionGQA(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.head_dim = config.n_embd // config.n_head


        shape = (config.n_head + 2 * config.n_kv_heads) * self.head_dim

        self.c_attn = nn.Linear(config.n_embd, shape, bias=False) # q_proj, k_proj i v_proj joined
        self.o_proj = nn.Linear(self.head_dim * config.n_head, config.n_embd, bias=False) # o_proj
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        

        self.n_kv_heads = config.n_kv_heads # Nombre de grups de query

        assert self.n_head % self.n_kv_heads == 0, "n_head must be divisible by n_group"
        self.queries_per_kv = self.n_head // self.n_kv_heads # n_rep

        #self.sliding_window_size = config.sliding_window_size
        self.max_seq_len = config.block_size

    def forward(self, x, freqs_cis):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        total_qkv = self.queries_per_kv + 2 # cada grup té a més de queries, 1 key i 1 value
        qkv = qkv.view(B, T, self.n_kv_heads, total_qkv, self.head_dim)
        qkv = qkv.permute(0, 2, 3, 1, 4)

        q, k, v = qkv.split((self.queries_per_kv, 1, 1), dim=2)

        q = q.reshape(B, -1, self.n_head, self.head_dim)  # (B, T, n_h, hs)
        k = k.reshape(B, -1, self.n_kv_heads, self.head_dim)  # (B, T, nh_kv, hs)
        v = v.reshape(B, -1, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        if self.n_kv_heads != self.n_head:
            k = repeat_kv(k, n_rep=self.queries_per_kv)
            v = repeat_kv(v, n_rep=self.queries_per_kv)

        # (B, n_h, T, hs)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        """
            SLIDING WINDOW DISCARDED (FOR NOW) TO MATCH EXACTLY LLAMAFORCAUSALLM
            scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)

            all_ones = torch.ones((T, T), device=q.device)
            sliding_mask = torch.triu(all_ones, -self.sliding_window_size + 1) * torch.tril(all_ones, self.sliding_window_size - 1)
            sliding_mask = sliding_mask.unsqueeze(0).unsqueeze(0)
            mask = torch.where(sliding_mask == 1, torch.zeros_like(scores), torch.full_like(scores, float("-inf")))
            scores = scores + mask
            scores = F.softmax(scores.float(), dim=-1).type_as(q)
            # [B, n_h, T, hs]
            y = torch.matmul(scores, v)
        """

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        return y


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d)) # weight

    def forward(self, x):
        norm = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).type_as(x)
        return self.scale.to(x.device) * (x / norm)
    

def rope_scaling(freqs: torch.Tensor):
    """
    From https://github.com/karpathy/nano-llama31/blob/master/llama31.py
    """
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 2048
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

def precompute_rope(dim, end, theta, use_scaled):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) # theta
    
    if use_scaled:
        inv_freq = rope_scaling(inv_freq)

    position_ids = torch.arange(end, device=inv_freq.device, dtype=torch.float32) # en alguns llocs, seq_idx
    
    freqs = torch.outer(position_ids, inv_freq)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    freqs_cis_real = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return freqs_cis_real

def apply_rope(x, freqs_cis):
    """
    Alt: Gemma implementation:
    def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1),
                    dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2],
                          -1).transpose(1, 2)
    return x_out

    
    """


    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    # xshaped is (bs, seqlen, n_heads, head_dim/2, 2), e.g. (4, 8, 32, 64, 2)

    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    # freqs_cis becomes (1, seqlen, 1, head_dim/2, 2), e.g. (1, 8, 1, 64, 2)
    
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    # x_out2 at this point is (bs, seqlen, n_heads, head_dim/2, 2), e.g. (4, 8, 32, 64, 2)
    x_out2 = x_out2.flatten(3)
    # x_out2 is now (bs, seqlen, n_heads, head_dim), e.g. (4, 8, 32, 128)

    return x_out2.type_as(x)


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
        self.w1   = nn.Linear(config.n_embd, config.intermediate_size, bias=False) # gate_proj
        self.w2 = nn.Linear(config.intermediate_size, config.n_embd, bias=False) # down proj
        self.w3 = nn.Linear(config.n_embd, config.intermediate_size, bias=False) # up proj
    
    def forward(self, x):
        return self.w2(F.gelu(self.w1(x), approximate="tanh")* self.w3(x)) 


class Block(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.norm_1 = RMSNorm(config.n_embd, config.norm_eps) # input_layernorm
        self.attn = CausalSelfAttentionGQA(config)
        self.norm_2 = RMSNorm(config.n_embd, config.norm_eps) # post_attention_layernorm
        self.mlp = MLP(config)
    
    def forward(self, x, freq):
        x = x + self.attn(self.norm_1(x), freq)
        x = x + self.mlp(self.norm_2(x))
        return x

class Aloja(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=3), #embed_tokens.weight
            h = nn.ModuleList([Block(config) for i in range(config.n_layer)]),
            norm_f =  RMSNorm(config.n_embd, config.norm_eps),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight # Linkeddddd

        self.freqs = precompute_rope(config.n_embd // config.n_head, config.max_seq_len * 2, config.rope_theta, config.use_scaled_rope)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean = 0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, start_pos:int = 0):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is smaller"

        x = self.transformer.wte(idx)
        self.freqs = self.freqs.to(x.device)
        freqs = self.freqs[:T]
        mask = torch.full((T, T), float("-inf"), device=idx.device)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.type_as(x)

        for block in self.transformer.h:
            x = block(x, freqs, mask)
        x = self.transformer.norm_f(x)
        logits = self.lm_head(x).float()
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                input=logits.transpose(1, 2),
                target=targets,
                reduction="mean",
                ignore_index=3,
            )
        return logits, loss
    
    
    def configure_optimizers(self, learning_rate, weight_decay=0.0, betas=(0.9, 0.97), device_type='cuda'):
        # start with all of the candidate parameters (that require grad)
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
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=1e-8, fused=use_fused)
        return optimizer


@dataclass
class AlojaConfig:
    block_size: int = 2048
    vocab_size: int = 65536
    n_layer: int = 32
    n_head: int = 8
    n_embd: int = 768
    intermediate_size = 2048 # 3072
    n_kv_heads: int = 4 # nombre de grups de query
    norm_eps: int = 1e-5
    rope_theta: float = 500000
    use_scaled_rope: bool = False
    max_batch_size: int = 32
    max_seq_len:int = 2048

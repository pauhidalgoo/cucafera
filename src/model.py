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


class CausalSelfAttentionGQA(nn.Module):
    def __init__(self, type, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.head_dim = config.n_embd // config.n_head


        shape = (config.n_head + 2 * config.n_kv_heads) * self.head_dim

        self.c_attn = nn.Linear(config.n_embd, shape, bias=False)
        self.c_proj = nn.Linear(self.head_dim * config.n_head, config.n_embd, bias=False)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        

        self.n_kv_heads = config.n_kv_heads # Nombre de grups de query

        assert self.n_head % self.n_kv_heads == 0, "n_head must be divisible by n_group"
        self.queries_per_kv = self.n_head // self.n_kv_heads # n_rep

        self.sliding_window_size = config.sliding_window_size
        self.max_seq_len = config.block_size

        self.attn_type = type

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
            k = torch.repeat_interleave(k, dim=2, repeats=self.queries_per_kv)
            v = torch.repeat_interleave(v, dim=2, repeats=self.queries_per_kv)

        # (B, n_h, T, hs)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        if self.attn_type == "Normal":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)

            all_ones = torch.ones((T, T), device=q.device)
            sliding_mask = torch.triu(all_ones, -self.sliding_window_size + 1) * torch.tril(all_ones, self.sliding_window_size - 1)
            sliding_mask = sliding_mask.unsqueeze(0).unsqueeze(0)
            mask = torch.where(sliding_mask == 1, torch.zeros_like(scores), torch.full_like(scores, float("-inf")))
            scores = scores + mask
            scores = F.softmax(scores.float(), dim=-1).type_as(q)
            # [B, n_h, T, hs]
            y = torch.matmul(scores, v)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).type_as(x)
        return self.scale * (x / norm)
    

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
        self.w1   = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
    
    def forward(self, x):
        return self.w2(F.gelu(self.w1(x), approximate="tanh")* self.w3(x)) 


class Block(nn.Module):

    def __init__(self, type, config):
        super().__init__()
        self.norm_1 = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = CausalSelfAttentionGQA(type, config)
        self.norm_2 = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = MLP(config)
    
    def forward(self, x, freq):
        x = x + self.attn(self.norm_1(x), freq)
        x = x + self.mlp(self.norm_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        types = ["Local", "Normal"] * (config.n_layer // 2)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(types[i], config) for i in range(config.n_layer)]),
            norm_f =  RMSNorm(config.n_embd, config.norm_eps),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight # Linke

        self.freqs = precompute_rope(config.n_embd // config.n_head, config.max_seq_len, config.rope_theta, config.use_scaled_rope)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is smaller"

        tok_embd = self.transformer.wte(idx)
        x = tok_embd
        freqs = self.freqs.to(x.device)

        for block in self.transformer.h:
            x = block(x, freqs)
        x = self.transformer.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
        

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"missmatches_keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    
    def configure_optimizers(self, learning_rate, weight_decay=0.0, betas=(0.9, 0.97), device_type='cuda'):
        train_params = []

        finetune_type = "all"
        if finetune_type == "rmsnorm":
            # let's only train the RMSNorm parameters to start
            for name, param in self.named_parameters():
                if "norm" in name:
                    train_params.append(param)
        elif finetune_type == "all":
            # let's train all parameters
            for param in self.parameters():
                train_params.append(param)
        elif finetune_type == "all_no_pos":
            # let's train all parameters except the positional embeddings and lm_head
            n, m = 0, 0
            for name, param in self.named_parameters():
                if name == "lm_head.weight":
                    # do not include
                    n += 1
                    continue
                elif name == "wte.weight":
                    # do not include and also does not require grad
                    m += 1
                    param.requires_grad = False
                else:
                    # do include
                    train_params.append(param)
            assert n == 1, "did not find lm_head.weight"
            assert m == 1, "did not find wte.weight"

        print("number of parameters: ", sum(p.numel() for p in self.parameters()))
        print("number of trainable parameters: ", sum(p.numel() for p in train_params))
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = True #'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(train_params, lr=learning_rate, betas=betas, **extra_args)
        return optimizer


@dataclass
class GPTConfig:
    block_size: int = 2048
    vocab_size: int = 50257
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

    sliding_window_size: int = 1024

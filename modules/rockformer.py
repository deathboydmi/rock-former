import torch
from torch import nn
import math

class RoPE(nn.Module):
    def __init__(self, d_head, max_context_size, base_freq=8192):
        super(RoPE, self).__init__()

        self.d_head = d_head

        inv_freq = 1.0 / (base_freq ** (torch.arange(0, d_head, 2) / d_head))
        pos = torch.arange(0, max_context_size, dtype=torch.float)
        angle = torch.outer(pos, inv_freq)

        self.register_buffer("cos", torch.cos(angle), persistent=False)
        self.register_buffer("sin", torch.sin(angle), persistent=False)

    def forward(self, x: torch.Tensor, offset=0) -> torch.Tensor:
        s = x.size(-2)
        d = self.d_head

        x_ = x.view(*x.shape[:-1], d // 2, 2)
        x1 = x_[..., 0]
        x2 = x_[..., 1]

        cos = self.cos[offset:offset+s].to(dtype=x.dtype, device=x.device)
        sin = self.sin[offset:offset+s].to(dtype=x.dtype, device=x.device)

        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos

        y = torch.stack((y1, y2), dim=-1).view_as(x)

        return y


class Attention(nn.Module):
    def __init__(self, embed_size, max_context_size, num_heads):
        super(Attention, self).__init__()

        self.num_heads = num_heads
        self.head_size = embed_size // num_heads
        self.max_context_size = max_context_size
        self.rope = RoPE(self.head_size, max_context_size, 1024)
        self.qkv_linear = nn.Linear(embed_size, 3 * embed_size, bias=False)
        self.heads_fuser = nn.Linear(embed_size, embed_size, bias=False)

        self.register_buffer("mask", torch.triu(torch.full((max_context_size, max_context_size), -torch.inf), 1), persistent=False)

        self.reset_kv_cache()


    def forward(self, x: torch.Tensor, use_kv_cache: bool):
        if x.size(1) > self.max_context_size:
            raise ValueError(f"Input sequence length {x.size(1)} exceeds maximum context size {self.max_context_size}")

        if use_kv_cache and self.kv_cache_initialised:
            return self.forward_kv_cache(x[:, -1:, ...])

        x_ = self.qkv_linear(x)

        x_ = x_.view(*x_.shape[:2], 3, self.num_heads, self.head_size).transpose(1, 3)

        q, k, v = x_.unbind(dim=2)

        q, k = self.rope(q), self.rope(k)

        if use_kv_cache:
            self.update_kv_cache(k.detach(), v.detach())
            self.kv_cache_initialised = True

        s = q @ k.transpose(-1, -2) / math.sqrt(self.head_size)

        mask = self.mask[:x.size(1), :x.size(1)].to(dtype=x.dtype, device=x.device)
        s += mask
        s = torch.softmax(s, -1)
        # DROPOUT
        a = s @ v
        a = a.transpose(1,2).reshape_as(x)

        y = self.heads_fuser(a)

        return y

    def update_kv_cache(self, k: torch.Tensor, v: torch.Tensor):
        assert k.shape == v.shape

        current_context_size = k.size(2)
        if current_context_size >= self.max_context_size:
            offset = current_context_size - self.max_context_size + 1

            self.k_cache, self.v_cache = k[:,:,offset:,...], v[:,:,offset:,...]
        else:
            self.k_cache, self.v_cache = k, v

    def reset_kv_cache(self):
        self.k_cache = None
        self.v_cache = None
        self.kv_cache_initialised = False


    def forward_kv_cache(self, x: torch.Tensor):
        assert not self.training
        qkv_last_token = self.qkv_linear(x)

        qkv_last_token = qkv_last_token.view(*qkv_last_token.shape[:2], 3, self.num_heads, self.head_size).transpose(1, 3)
        q_last_token, k_last_token, v_last_token = qkv_last_token.unbind(dim=2) # shape [batch_size, num_heads, 1, head_size]

        k_context = self.k_cache # shape [batch_size, num_heads, current_context_length, head_size]
        v_context = self.v_cache

        prev_context_size = k_context.size(-2)
        q_last_token, k_last_token = self.rope(q_last_token, prev_context_size), self.rope(k_last_token, prev_context_size)

        k = torch.cat((k_context, k_last_token), dim=2)
        v = torch.cat((v_context, v_last_token), dim=2)

        self.update_kv_cache(k, v)

        s = q_last_token @ k.transpose(-1, -2) / math.sqrt(self.head_size)

        s = torch.softmax(s, -1)
        # DROPOUT
        a = s @ v
        a = a.transpose(1,2).reshape_as(x)

        y = self.heads_fuser(a)

        return y


class SwiGLU(nn.Module):
    def __init__(self, embed_size, ff_size):
        super(SwiGLU, self).__init__()

        self.in_layer = nn.Linear(embed_size, ff_size * 2, bias=False)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(ff_size, embed_size, bias=False)

    def forward(self, x: torch.Tensor):
        ab = self.in_layer(x)
        a, b = ab.chunk(2, -1)

        y = a * self.silu(b)
        y = self.out_layer(y)

        return y


"""
                ____________
                |           x
                |           |
                |       RMSNorm
                |           |___________
                |           KQV         |
                |           |           |
                |   Multihead split     |
                |           |           |
                |       RoPE(Q,K)       |
                |           |           |   Attention
Attention Block |   Attention(K,Q,V)    |
                |           |           |
                |       Fuse Heads      |
                |           |           |
                |       x + |___________|
                |           |
                |       RMSNorm
                |           |
                |       SwiGLU FFN
                |           |
                |_______x + |
"""
class AttentionBlock(nn.Module):
    def __init__(self, embed_size, max_context_size, num_heads, ff_hidden_layer_size):
        super(AttentionBlock, self).__init__()

        self.pre_norm = nn.RMSNorm(embed_size)
        self.attention = Attention(embed_size, max_context_size, num_heads)
        self.post_norm = nn.RMSNorm(embed_size)
        self.swiglu = SwiGLU(embed_size, ff_hidden_layer_size)

    def reset_kv_cache(self):
        self.attention.reset_kv_cache()

    def forward(self, x: torch.Tensor, use_kv_cache: bool):
        y = self.attention(self.pre_norm(x), use_kv_cache)
        y = self.swiglu(self.post_norm(x + y))
        return x + y


class AttentionBlocksStack(nn.Module):
    def __init__(self, embed_size, max_context_size, num_heads, ff_hidden_layer_size, blocks_number):
        super(AttentionBlocksStack, self).__init__()

        self.atten_blocks = nn.ModuleList([AttentionBlock(embed_size, max_context_size, num_heads, ff_hidden_layer_size) for _ in range(blocks_number)])

    def reset_kv_cache(self):
        for atten_block in self.atten_blocks:
            atten_block.reset_kv_cache()

    def forward(self, x: torch.Tensor, use_kv_cache: bool = False):
        for atten_block in self.atten_blocks:
            x = atten_block(x, use_kv_cache)

        return x


class Rockformer(nn.Module):
    def __init__(self,
                    vocab_size: int,
                    max_context_size: int,
                    embed_size: int,
                    ff_hidden_layer_size: int,
                    atten_heads_number: int,
                    blocks_number: int,

                
                ):
        super(Rockformer, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)

        self.atten_blocks = AttentionBlocksStack(embed_size, max_context_size, atten_heads_number, ff_hidden_layer_size, blocks_number)
        self.post_norm = nn.RMSNorm(embed_size)
        self.reverse_embed = nn.Linear(embed_size, vocab_size, bias=False)

        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        self.reverse_embed.weight = self.embed.weight

        self.seq = nn.Sequential(
                            self.embed,
                            self.atten_blocks,
                            self.post_norm,
                            self.reverse_embed
                        )

        self.max_context_size = max_context_size
        self.__generation = False

    def reset_generation(self):
        self.atten_blocks.reset_kv_cache()

    def generation(self):
        super().eval()
        self.__generation = True
        return self

    def train(self, mode = True):
        self.__generation = False
        return super().train(mode)

    def eval(self):
        return self.train(mode=False)

    def forward(self, x):
        # legacy
        if not self.__generation:
            return self.seq(x)

        x_ = self.embed(x)
        x_ = self.atten_blocks(x_, use_kv_cache=self.__generation)
        x_ = self.post_norm(x_)
        logits = self.reverse_embed(x_)
        return logits

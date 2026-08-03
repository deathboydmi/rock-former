import torch
from torch import nn
import math

class RoPE(nn.Module):
    def __init__(self, d_head, max_context_size, base_freq=1024):
        super(RoPE, self).__init__()

        self.d_head = d_head

        inv_freq = 1.0 / (base_freq ** (torch.arange(0, d_head, 2) / d_head))
        pos = torch.arange(0, max_context_size, dtype=torch.float)
        angle = torch.outer(pos, inv_freq)

        self.register_buffer("cos", torch.cos(angle), persistent=False)
        self.register_buffer("sin", torch.sin(angle), persistent=False)

    def forward(self, x):
        s = x.size(-2) # context size
        d = self.d_head

        x_ = x.view(*x.shape[:-1], d // 2, 2)
        x1 = x_[..., 0]
        x2 = x_[..., 1]

        cos = self.cos[:s].to(dtype=x.dtype, device=x.device)
        sin = self.sin[:s].to(dtype=x.dtype, device=x.device)

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


    def forward(self, x: torch.Tensor):
        if x.size(1) > self.max_context_size:
            raise ValueError(f"Input sequence length {x.size(1)} exceeds maximum context size {self.max_context_size}")

        x_ = self.qkv_linear(x)

        x_ = x_.view(*x_.shape[:2], 3, self.num_heads, self.head_size).transpose(1, 3)

        q, k, v = x_.unbind(dim=2)

        q, k = self.rope(q), self.rope(k)

        s = q @ k.transpose(-1, -2) / math.sqrt(self.head_size)

        mask = self.mask[:x.size(1), :x.size(1)].to(dtype=x.dtype, device=x.device)
        s += mask
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


    def forward(self, x: torch.Tensor):
        y = self.attention(self.pre_norm(x))
        y = self.swiglu(self.post_norm(x + y))
        return x + y


class AttentionBlocksStack(nn.Module):
    def __init__(self, embed_size, max_context_size, num_heads, ff_hidden_layer_size, blocks_number):
        super(AttentionBlocksStack, self).__init__()

        self.atten_blocks = nn.ModuleList([AttentionBlock(embed_size, max_context_size, num_heads, ff_hidden_layer_size) for _ in range(blocks_number)])

    def forward(self, x: torch.Tensor):
        for atten_block in self.atten_blocks:
            x = atten_block(x)

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

    def forward(self, x):
        return self.seq(x)
---
hero:
  title: "RoPE Positional Encoding"
  subtitle: "Rotary Position Embeddings"
  tags:
    - "🤖 Transformers"
    - "⏱️ 10 min read"
---

RoPE (Rotary Position Embedding) is a modern way to encode position information in transformers!

## The Problem

Transformers don't know word order without position information!

```yaml
"Dog bites man" vs "Man bites dog"
→ Without positions, looks the same to transformer!

Need to add position information!
```

## How RoPE Works

```python
import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x):
        seq_len = x.size(1)
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        
        return cos_emb, sin_emb

def apply_rope(x, cos, sin):
    """Apply rotary embeddings"""
    x1, x2 = x[..., ::2], x[..., 1::2]
    rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)
    return rotated

# Use it
rope = RotaryPositionalEmbedding(dim=64)
x = torch.randn(1, 10, 64)
cos, sin = rope(x)
x_with_pos = apply_rope(x, cos, sin)
```

## Why RoPE is Better

```yaml
Old way (learned embeddings):
  - Fixed max sequence length
  - Doesn't generalize to longer sequences
  
RoPE:
  ✓ Works for any sequence length
  ✓ Relative positions encoded
  ✓ Better extrapolation
  ✓ Used in LLaMA, GPT-NeoX
```

## Key Takeaways

✓ **Rotary:** Encodes position via rotation

✓ **Relative:** Captures relative positions

✓ **Modern:** Used in latest LLMs

**Remember:** RoPE is the modern way to handle positions! 🎉

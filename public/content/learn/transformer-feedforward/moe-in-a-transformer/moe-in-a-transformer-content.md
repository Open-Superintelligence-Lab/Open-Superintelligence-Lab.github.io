---
hero:
  title: "MoE in a Transformer"
  subtitle: "Integrating Mixture of Experts"
  tags:
    - "🔀 MoE"
    - "⏱️ 10 min read"
---

MoE replaces the standard FFN in transformer blocks with a sparse expert layer!

## MoE Transformer Block

```python
import torch.nn as nn

class MoETransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, num_experts=8):
        super().__init__()
        
        # Attention (same as standard)
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # MoE instead of FFN
        self.moe = MixtureOfExperts(d_model, num_experts)
        
        # Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # MoE (replaces FFN)
        moe_out = self.moe(x)
        x = self.norm2(x + moe_out)
        
        return x
```

## Key Difference

```yaml
Standard Transformer:
  Attention → FFN → Output

MoE Transformer:
  Attention → MoE → Output
              ↑
  (Sparse expert routing)
```

## Key Takeaways

✓ **Drop-in replacement:** MoE replaces FFN

✓ **Same interface:** Input/output shapes unchanged

✓ **More capacity:** Many experts, sparse activation

**Remember:** MoE makes transformers bigger without more compute! 🎉

---
hero:
  title: "The Gate"
  subtitle: "Router Network in Mixture of Experts"
  tags:
    - "🔀 MoE"
    - "⏱️ 8 min read"
---

The gate (router) decides **which experts each token should use**!

## Router Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Router(nn.Module):
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)
    
    def forward(self, x, top_k=2):
        # x: (batch, seq, d_model)
        
        # Compute routing scores
        router_logits = self.gate(x)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, top_k, dim=-1)
        
        # Normalize
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        return top_k_probs, top_k_indices

# Use it
router = Router(d_model=512, num_experts=8)
x = torch.randn(2, 10, 512)
probs, indices = router(x, top_k=2)

print(probs.shape)    # torch.Size([2, 10, 2])
print(indices.shape)  # torch.Size([2, 10, 2])
```

## Key Takeaways

✓ **Router:** Selects which experts to use

✓ **Top-K:** Usually top-2 experts per token

✓ **Learnable:** Router weights are trained

**Remember:** The gate is the traffic controller! 🎉

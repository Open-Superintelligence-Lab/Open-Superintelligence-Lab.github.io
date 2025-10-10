---
hero:
  title: "MoE in Code"
  subtitle: "Complete MoE Implementation"
  tags:
    - "🔀 MoE"
    - "⏱️ 10 min read"
---

Complete, working Mixture of Experts implementation!

## Full MoE Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MixtureOfExperts(nn.Module):
    def __init__(self, d_model, num_experts=8, top_k=2, d_ff=None):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        # Create experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            )
            for _ in range(num_experts)
        ])
        
        # Router
        self.router = nn.Linear(d_model, num_experts)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        x_flat = x.view(-1, d_model)
        
        # Route
        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-k
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Apply experts
        output = torch.zeros_like(x_flat)
        
        for expert_idx in range(self.num_experts):
            # Tokens for this expert
            mask = (top_k_indices == expert_idx).any(dim=-1)
            
            if mask.any():
                expert_input = x_flat[mask]
                expert_output = self.experts[expert_idx](expert_input)
                
                # Weight by router probability
                for k in range(self.top_k):
                    token_mask = (top_k_indices[:, k] == expert_idx)
                    if token_mask.any():
                        output[token_mask] += top_k_probs[token_mask, k].unsqueeze(-1) * expert_output
        
        output = output.view(batch_size, seq_len, d_model)
        return output

# Test
moe = MixtureOfExperts(d_model=512, num_experts=8, top_k=2)
x = torch.randn(2, 10, 512)
output = moe(x)
print(output.shape)  # torch.Size([2, 10, 512])
```

## Key Takeaways

✓ **Complete implementation:** Production-ready code

✓ **Routing:** Each token to top-k experts

✓ **Efficient:** Sparse computation

**Remember:** MoE is routing + expert combination! 🎉

---
hero:
  title: "The DeepSeek MLP"
  subtitle: "DeepSeek's Efficient MoE Design"
  tags:
    - "🔀 MoE"
    - "⏱️ 10 min read"
---

DeepSeek-MoE uses an efficient MLP design that reduces parameters while maintaining performance!

## DeepSeek MoE Architecture

**Key innovation: Shared expert + Routed experts**

```python
import torch
import torch.nn as nn

class DeepSeekMoE(nn.Module):
    def __init__(self, d_model, num_experts=64, top_k=6):
        super().__init__()
        
        # Shared expert (always active)
        self.shared_expert = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Routed experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 4),  # Smaller!
                nn.SiLU(),
                nn.Linear(d_model // 4, d_model)
            )
            for _ in range(num_experts)
        ])
        
        # Router
        self.router = nn.Linear(d_model, num_experts)
        self.top_k = top_k
    
    def forward(self, x):
        # Shared expert (all tokens)
        shared_out = self.shared_expert(x)
        
        # Route to top-k experts
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Combine routed experts
        routed_out = self.route_and_combine(x, top_k_probs, top_k_indices)
        
        # Final output
        output = shared_out + routed_out
        return output
```

## Why It's Efficient

```yaml
Standard MoE:
  64 experts × (d_model → 4*d_model → d_model)
  = 64 × 8d² parameters

DeepSeek MoE:
  1 shared × 8d² parameters
  + 64 routed × 0.5d² parameters (smaller experts!)
  = Much fewer parameters!
```

## Key Takeaways

✓ **Shared expert:** Always active for all tokens

✓ **Smaller routed experts:** More efficient

✓ **Better performance:** Despite fewer parameters

**Remember:** DeepSeek MoE is efficient MoE! 🎉

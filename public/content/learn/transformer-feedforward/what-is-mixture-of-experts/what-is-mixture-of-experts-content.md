---
hero:
  title: "What is Mixture of Experts"
  subtitle: "Sparse Expert Models Explained"
  tags:
    - "🔀 MoE"
    - "⏱️ 10 min read"
---

Mixture of Experts (MoE) uses **multiple specialized sub-networks (experts)** and routes inputs to the most relevant ones!

![MoE Routing](/content/learn/transformer-feedforward/what-is-mixture-of-experts/moe-routing.png)

## The Core Idea

Instead of one big feedforward network:
- Have many smaller expert networks
- Route each token to top-K experts
- Combine expert outputs

```yaml
Traditional FFN:
  All tokens → Same FFN → Output

MoE:
  Token 1 → Expert 2 + Expert 5 → Output
  Token 2 → Expert 1 + Expert 3 → Output
  Token 3 → Expert 2 + Expert 7 → Output
  
Each token uses different experts!
```

## Simple Example

```python
import torch
import torch.nn as nn

class SimpleMoE(nn.Module):
    def __init__(self, d_model, num_experts=8):
        super().__init__()
        
        # Multiple expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Linear(d_model * 4, d_model)
            )
            for _ in range(num_experts)
        ])
        
        # Router (chooses which experts to use)
        self.router = nn.Linear(d_model, num_experts)
    
    def forward(self, x):
        # x: (batch, seq, d_model)
        
        # Router scores
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Get top-2 experts
        top_k_probs, top_k_indices = torch.topk(router_probs, k=2, dim=-1)
        
        # Route to experts
        output = torch.zeros_like(x)
        for i in range(len(self.experts)):
            # Find tokens routed to this expert
            mask = (top_k_indices == i).any(dim=-1)
            if mask.any():
                expert_output = self.experts[i](x[mask])
                output[mask] += expert_output * top_k_probs[mask, (top_k_indices[mask] == i).argmax(dim=-1)].unsqueeze(-1)
        
        return output
```

## Why MoE?

```yaml
Benefits:
  ✓ Huge capacity (many parameters)
  ✓ Efficient (only use few experts per token)
  ✓ Specialization (experts learn different patterns)
  
Trade-offs:
  ✗ Complex training
  ✗ Load balancing needed
  ✗ More memory
```

## Used In

- Switch Transformer
- DeepSeek-MoE
- Mixtral
- GPT-4 (rumored)

## Key Takeaways

✓ **Multiple experts:** Specialized sub-networks

✓ **Sparse routing:** Each token uses few experts

✓ **Scalable:** Add experts without much compute cost

**Remember:** MoE = specialized experts for different patterns! 🎉

---
hero:
  title: "Combining Experts"
  subtitle: "Weighted Combination of Expert Outputs"
  tags:
    - "🔀 MoE"
    - "⏱️ 8 min read"
---

After routing, we combine expert outputs using router weights!

## Combining Formula

**Output = Σ (router_weight_i × expert_i(x))**

```python
import torch

# Router selected experts 2 and 5 with weights
expert_indices = [2, 5]
expert_weights = [0.6, 0.4]

# Expert outputs
expert_2_output = torch.tensor([1.0, 2.0, 3.0])
expert_5_output = torch.tensor([4.0, 5.0, 6.0])

# Weighted combination
final_output = 0.6 * expert_2_output + 0.4 * expert_5_output
print(final_output)
# tensor([2.2000, 3.2000, 4.2000])
```

## Complete MoE Forward

```python
def moe_forward(x, experts, router):
    # Get routing decisions
    weights, indices = router(x, top_k=2)
    
    # Combine expert outputs
    output = torch.zeros_like(x)
    
    for i in range(len(experts)):
        # Mask for tokens using this expert
        expert_mask = (indices == i).any(dim=-1)
        
        if expert_mask.any():
            expert_out = experts[i](x[expert_mask])
            expert_weight = weights[expert_mask][(indices[expert_mask] == i).any(dim=-1)]
            output[expert_mask] += expert_weight.unsqueeze(-1) * expert_out
    
    return output
```

## Key Takeaways

✓ **Weighted sum:** Combine based on router weights

✓ **Sparse:** Only use selected experts

✓ **Efficient:** Skip unused experts

**Remember:** Combining is just weighted averaging! 🎉

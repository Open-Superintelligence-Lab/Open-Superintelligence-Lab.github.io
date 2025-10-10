---
hero:
  title: "The Expert"
  subtitle: "Individual Expert Networks in MoE"
  tags:
    - "🔀 MoE"
    - "⏱️ 8 min read"
---

An expert is a **specialized feedforward network** in the Mixture of Experts architecture!

## Expert Structure

```python
import torch
import torch.nn as nn

class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),  # Modern activation
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x):
        return self.net(x)

# Create expert
expert = Expert(d_model=512, d_ff=2048)
x = torch.randn(10, 512)
output = expert(x)
print(output.shape)  # torch.Size([10, 512])
```

## Multiple Experts

```python
num_experts = 8

experts = nn.ModuleList([
    Expert(d_model=512, d_ff=2048)
    for _ in range(num_experts)
])

# Each expert specializes in different patterns!
# Expert 0: Maybe handles technical text
# Expert 1: Maybe handles conversational text
# Expert 2: Maybe handles code
# etc.
```

## Expert Specialization

```yaml
During training:
  - Router learns which expert for which pattern
  - Experts specialize automatically
  - No manual assignment needed!

Result:
  - Expert 1: Good at math
  - Expert 2: Good at grammar
  - Expert 3: Good at facts
  - etc.
```

## Key Takeaways

✓ **Expert = FFN:** Same structure as standard feedforward

✓ **Specialized:** Each learns different patterns

✓ **Independent:** Trained separately via routing

**Remember:** Experts are specialized sub-networks! 🎉

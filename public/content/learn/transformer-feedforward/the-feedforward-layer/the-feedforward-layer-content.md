---
hero:
  title: "The Feedforward Layer"
  subtitle: "FFN in Transformer Blocks"
  tags:
    - "🔀 MoE"
    - "⏱️ 8 min read"
---

The feedforward network (FFN) in transformers processes each position independently!

## Structure

```python
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

# Typical: d_ff = 4 × d_model
ffn = FeedForward(d_model=512, d_ff=2048)
```

## Key Takeaways

✓ **Two layers:** Expand then compress

✓ **Position-wise:** Same FFN for each position

✓ **Standard ratio:** d_ff = 4 × d_model

**Remember:** FFN adds capacity after attention! 🎉

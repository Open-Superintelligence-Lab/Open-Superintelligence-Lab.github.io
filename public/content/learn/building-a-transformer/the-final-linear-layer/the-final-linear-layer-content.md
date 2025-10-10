---
hero:
  title: "The Final Linear Layer"
  subtitle: "From Hidden States to Predictions"
  tags:
    - "🤖 Transformers"
    - "⏱️ 8 min read"
---

The final linear layer projects transformer outputs to vocabulary logits for prediction!

## Language Model Head

```python
import torch
import torch.nn as nn

class LMHead(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, x):
        x = self.ln(x)
        logits = self.linear(x)
        return logits

# Use it
lm_head = LMHead(d_model=768, vocab_size=50000)
hidden_states = torch.randn(32, 128, 768)  # (batch, seq, dim)
logits = lm_head(hidden_states)

print(logits.shape)  # torch.Size([32, 128, 50000])
# For each position: 50000 logits (one per vocab token)
```

## Complete Forward Pass

```python
# Input tokens → Embeddings → Transformer → LM Head → Logits

input_ids = torch.randint(0, 50000, (1, 10))
embeddings = embedding_layer(input_ids)
hidden_states = transformer_blocks(embeddings)
logits = lm_head(hidden_states)

# Get next token prediction
next_token_logits = logits[:, -1, :]  # Last position
next_token = torch.argmax(next_token_logits, dim=-1)
```

## Key Takeaways

✓ **Final layer:** Hidden states → vocabulary logits

✓ **Large:** Often biggest layer (vocab_size is huge)

✓ **Shared weights:** Often tied with embedding matrix

**Remember:** Final layer converts understanding to predictions! 🎉

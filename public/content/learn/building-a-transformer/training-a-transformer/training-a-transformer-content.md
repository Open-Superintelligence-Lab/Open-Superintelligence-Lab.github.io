---
hero:
  title: "Training a Transformer"
  subtitle: "How to Train Language Models"
  tags:
    - "🤖 Transformers"
    - "⏱️ 10 min read"
---

Training transformers involves next-token prediction and lots of data!

## The Training Objective

**Goal: Predict the next token given previous tokens**

```python
import torch
import torch.nn as nn

# Training data
input_tokens = torch.tensor([[1, 2, 3, 4]])    # Input
target_tokens = torch.tensor([[2, 3, 4, 5]])   # Targets (shifted by 1)

# Model forward
logits = model(input_tokens)  # (1, 4, vocab_size)

# Loss: Cross entropy
criterion = nn.CrossEntropyLoss()
loss = criterion(
    logits.view(-1, vocab_size),  # Flatten
    target_tokens.view(-1)         # Flatten
)
```

## Complete Training Loop

```python
import torch
import torch.optim as optim

def train_step(model, batch, optimizer, criterion):
    # Get input and target (shifted)
    input_ids = batch[:, :-1]
    targets = batch[:, 1:]
    
    # Forward
    logits = model(input_ids)
    
    # Loss
    loss = criterion(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    
    # Update
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()

# Training
model = Transformer(vocab_size=50000)
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch in dataloader:
        loss = train_step(model, batch, optimizer, criterion)
```

## Key Takeaways

✓ **Next-token prediction:** Core training task

✓ **Shift targets:** Input[:-1] → Target[1:]

✓ **Cross entropy:** Standard loss for LMs

**Remember:** Training is just next-token prediction! 🎉

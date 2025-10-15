---
hero:
  title: "Training a Transformer"
  subtitle: "How to Train Language Models"
  tags:
    - "🤖 Transformers"
    - "⏱️ 10 min read"
---

Training transformers involves next-token prediction and lots of data!

## The Core Training Objective

Language models learn by predicting the next token in a sequence. This simple objective powers all of GPT, LLaMA, and modern LLMs!

### Understanding Next-Token Prediction

**The setup:**
```
Given: "The cat sat"
Predict: "on" (the next word)

Given: "The cat sat on"
Predict: "the" (the next word)

Given: "The cat sat on the"
Predict: "mat" (the next word)
```

**Mathematically:**
```
P(token_t | token_1, token_2, ..., token_{t-1})

Model learns to predict each token given all previous tokens
```

### Preparing Training Data

The key trick: use the same sequence as input AND target, just shifted by one position!

```python
import torch
import torch.nn as nn

# Original sequence: "The cat sat on the mat"
sequence = torch.tensor([[15, 234, 567, 1089, 234, 2341]])

# Split into input and target
input_tokens = sequence[:, :-1]    # All except last
target_tokens = sequence[:, 1:]    # All except first

print(f"Input:  {input_tokens}")   # [[15, 234, 567, 1089, 234]]
print(f"Target: {target_tokens}")  # [[234, 567, 1089, 234, 2341]]
```

**What we're teaching:**
```
Input: 15    → Predict: 234  (after "The", predict "cat")
Input: 234   → Predict: 567  (after "cat", predict "sat")
Input: 567   → Predict: 1089 (after "sat", predict "on")
...
```

### Computing the Loss

```python
# Model forward pass
logits = model(input_tokens)  # Shape: (1, 5, vocab_size)

# Cross entropy loss
criterion = nn.CrossEntropyLoss()
loss = criterion(
    logits.view(-1, vocab_size),    # Flatten: (5, vocab_size)
    target_tokens.view(-1)           # Flatten: (5,)
)
```

**Why flatten?**
```
CrossEntropyLoss expects:
  Input: (N, vocab_size) - N predictions
  Target: (N,) - N target indices

We have:
  logits: (batch, seq_len, vocab_size)
  targets: (batch, seq_len)

Flatten: (batch×seq_len, vocab_size) and (batch×seq_len,)
```

## The Complete Training Loop

Let's build a production-quality training function:

### Step 1: The Training Step Function

```python
import torch
import torch.optim as optim

def train_step(model, batch, optimizer, criterion):
    # batch shape: (batch_size, seq_len)
    # Example: (32, 128) - 32 sequences of 128 tokens
    
    # Split into input and target (shift by 1)
    input_ids = batch[:, :-1]   # All tokens except last
    targets = batch[:, 1:]       # All tokens except first
    # Now both are shape: (32, 127)
```

**The shift:**
```
Original: [1, 2, 3, 4, 5]
Input:    [1, 2, 3, 4]     → Predict next
Target:   [2, 3, 4, 5]     ← What comes next
```

```python
    # FORWARD PASS
    logits = model(input_ids)
    # Shape: (batch, seq_len-1, vocab_size)
    # Example: (32, 127, 50000)
```

**Logits:** Scores for each position's next token.

```python
    # COMPUTE LOSS
    loss = criterion(
        logits.reshape(-1, logits.size(-1)),  # (batch×seq, vocab)
        targets.reshape(-1)                    # (batch×seq,)
    )
```

**Flatten for CrossEntropyLoss:**
```
Before: logits (32, 127, 50000), targets (32, 127)
After:  logits (4064, 50000), targets (4064,)

4064 = 32×127 independent predictions!
```

```python
    # BACKWARD PASS
    optimizer.zero_grad()  # Clear old gradients
    loss.backward()         # Compute gradients
```

**Backpropagation through entire transformer!**

```python
    # GRADIENT CLIPPING (important!)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

**Why clip?**
```
Sometimes gradients explode (become very large)
Clipping prevents this:
  If ||grad|| > 1.0, scale down to ||grad|| = 1.0
  
This stabilizes training!
```

```python
    # UPDATE WEIGHTS
    optimizer.step()
    
    return loss.item()
```

### Step 2: Setup Model and Optimizer

```python
# Create model
model = Transformer(
    vocab_size=50000,
    d_model=768,
    n_heads=12,
    n_layers=12,
    d_ff=3072
)

# AdamW optimizer (Adam with weight decay)
optimizer = optim.AdamW(
    model.parameters(),
    lr=3e-4,              # Learning rate
    betas=(0.9, 0.999),   # Momentum parameters
    weight_decay=0.01     # L2 regularization
)

# Cross entropy loss
criterion = nn.CrossEntropyLoss()
```

**Why AdamW?**
- Adaptive learning rates per parameter
- Momentum for faster convergence
- Weight decay for regularization
- Standard for transformer training

### Step 3: The Training Loop

```python
# Training hyperparameters
num_epochs = 10
accumulation_steps = 4  # Gradient accumulation

for epoch in range(num_epochs):
    model.train()  # Set to training mode
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Train step
        loss = train_step(model, batch, optimizer, criterion)
        total_loss += loss
        
        # Log progress
        if batch_idx % 100 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {avg_loss:.4f}")
    
    # Epoch summary
    avg_epoch_loss = total_loss / len(dataloader)
    print(f"\\nEpoch {epoch} complete. Average loss: {avg_epoch_loss:.4f}")
```

**Expected output:**
```
Epoch 0, Batch 0, Loss: 10.8234
Epoch 0, Batch 100, Loss: 8.2341
Epoch 0, Batch 200, Loss: 6.7650
...
Epoch 0 complete. Average loss: 5.2341

Epoch 1, Batch 0, Loss: 4.8234
...
```

Loss decreases over time - the model is learning!

## Key Takeaways

✓ **Next-token prediction:** Core training task

✓ **Shift targets:** Input[:-1] → Target[1:]

✓ **Cross entropy:** Standard loss for LMs

**Remember:** Training is just next-token prediction! 🎉

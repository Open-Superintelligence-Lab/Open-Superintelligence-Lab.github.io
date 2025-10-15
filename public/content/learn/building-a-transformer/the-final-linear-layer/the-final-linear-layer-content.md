---
hero:
  title: "The Final Linear Layer"
  subtitle: "From Hidden States to Predictions"
  tags:
    - "🤖 Transformers"
    - "⏱️ 8 min read"
---

The final linear layer projects transformer outputs to vocabulary logits for prediction!

## The Critical Last Step

After all the transformer blocks process your input, you have rich, context-aware representations for each token. But these are still just vectors in hidden space - we need to convert them to **predictions**!

### The Problem

```yaml
Transformer output: (batch, seq_len, d_model)
  Example: (32, 100, 768)
  
Each position is a 768-dimensional vector representing:
  - The token's meaning
  - Its context from other tokens
  - High-level understanding
  
But we need: Predictions over vocabulary (50,000 words)!
```

### The Solution: Linear Projection

**Map hidden space → vocabulary space**

```
768-dimensional vector → 50,000 scores (one per vocab word)
```

## Building the Language Model Head

Let's build this step-by-step:

### Step 1: Initialize Components

```python
import torch
import torch.nn as nn

class LMHead(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        # Final layer normalization
        self.ln = nn.LayerNorm(d_model)
```

**Purpose:** Normalize hidden states before projection for stability.

```python
        # Projection to vocabulary
        self.linear = nn.Linear(d_model, vocab_size, bias=False)
```

**The projection:**
```
d_model=768 → vocab_size=50,000

Parameters: 768 × 50,000 = 38,400,000
This is often the LARGEST layer in the model!
```

**Why no bias?** Common practice in language models - saves parameters and works well.

### Step 2: Forward Pass

```python
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        
        # Normalize
        x = self.ln(x)
        # Ensures stable values before projection
        
        # Project to vocabulary
        logits = self.linear(x)
        # Shape: (batch, seq_len, vocab_size)
        
        return logits
```

**What we get:**
- For each position in each sequence
- Scores for all vocabulary words
- Higher score = more likely word

### Step 3: Using the LM Head

```python
# Create language model head
lm_head = LMHead(d_model=768, vocab_size=50000)

# Hidden states from transformer
hidden_states = torch.randn(32, 128, 768)  # 32 sequences, 128 tokens

# Project to vocabulary
logits = lm_head(hidden_states)

print(f"Hidden shape: {hidden_states.shape}")  # torch.Size([32, 128, 768])
print(f"Logits shape: {logits.shape}")         # torch.Size([32, 128, 50000])
```

**Interpreting the output:**
```
logits[0, 5, :] = scores for all 50,000 words at position 5 of sequence 0

Example values:
  logits[0, 5, 1234] = 2.5  (word "cat")
  logits[0, 5, 5678] = 4.1  (word "dog") ← Highest!
  logits[0, 5, 9012] = 0.3  (word "quantum")
  
Prediction: "dog" (index 5678, highest score 4.1)
```

## The Complete Forward Pass

Let's see the full journey from tokens to predictions:

### Step 1: Token IDs

```python
# Example input: "The cat sat"
input_ids = torch.tensor([[15, 234, 567]])
# Token IDs: 15="The", 234="cat", 567="sat"
```

### Step 2: Embedding Layer

```python
embedding_layer = nn.Embedding(50000, 768)
embeddings = embedding_layer(input_ids)
# Shape: (1, 3, 768)
# Each token → 768-dim vector
```

### Step 3: Transformer Blocks

```python
# Pass through N transformer blocks
hidden_states = embeddings
for block in transformer_blocks:
    hidden_states = block(hidden_states)
# Shape still: (1, 3, 768)
# But now context-aware!
```

### Step 4: Language Model Head

```python
lm_head = LMHead(d_model=768, vocab_size=50000)
logits = lm_head(hidden_states)
# Shape: (1, 3, 50000)
```

**What we have:**
```
Position 0 ("The"): 50,000 scores for next word
Position 1 ("cat"): 50,000 scores for next word
Position 2 ("sat"): 50,000 scores for next word
```

### Step 5: Get Predictions

```python
# For autoregressive generation, we care about the LAST position
next_token_logits = logits[:, -1, :]  # Last position only
# Shape: (1, 50000)

# Get most likely token
next_token_id = torch.argmax(next_token_logits, dim=-1)
print(f"Next token ID: {next_token_id.item()}")
# Example: 1089 (maybe "on")

# Or sample from distribution
probs = torch.softmax(next_token_logits, dim=-1)
next_token_id = torch.multinomial(probs, num_samples=1)
# Stochastic sampling for more diverse generation
```

**Complete generation loop:**
```python
# Start with prompt
tokens = [15, 234, 567]  # "The cat sat"

# Generate 5 more tokens
for _ in range(5):
    # Forward pass
    input_ids = torch.tensor([tokens])
    logits = model(input_ids)
    
    # Get last position logits
    next_logits = logits[0, -1, :]
    
    # Sample next token
    probs = torch.softmax(next_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()
    
    # Add to sequence
    tokens.append(next_token)

print("Generated:", tokens)
# [15, 234, 567, 1089, 234, 2341, 567, 890]
# "The cat sat on the mat" (example)
```

## Key Takeaways

✓ **Final layer:** Hidden states → vocabulary logits

✓ **Large:** Often biggest layer (vocab_size is huge)

✓ **Shared weights:** Often tied with embedding matrix

**Remember:** Final layer converts understanding to predictions! 🎉

---
hero:
  title: "Transformer Architecture"
  subtitle: "Understanding the Transformer Model"
  tags:
    - "🤖 Transformers"
    - "⏱️ 12 min read"
---

The Transformer is the architecture behind GPT, BERT, and modern LLMs. It's built entirely on attention!

![Transformer Diagram](/content/learn/building-a-transformer/transformer-architecture/transformer-diagram.png)

## The Paradigm Shift

In 2017, the paper "Attention Is All You Need" introduced transformers and changed AI forever. Before transformers, the dominant architectures were:
- **RNNs/LSTMs**: Sequential processing, slow
- **CNNs**: Good for local patterns, limited for sequences

**Transformers eliminated both recurrence and convolution**, relying purely on attention mechanisms. This was revolutionary!

### Why Transformers Won

**Speed:**
```
RNN: O(n) sequential steps → slow, can't parallelize
Transformer: O(1) parallel operations → fast, GPU-friendly
```

**Memory:**
```
RNN: Information compressed through hidden states → lossy
Transformer: Direct attention to all positions → no compression
```

**Scale:**
```
RNN: Hard to train with >1000 time steps
Transformer: Easily handles 100,000+ tokens (with modifications)
```

**Result:** Transformers power ALL modern large language models!

## The Two Main Architectures

Transformers come in three flavors, depending on the task:

### 1. Encoder-Only (BERT-style)

```yaml
Use case: Understanding tasks
Examples: BERT, RoBERTa
Tasks: Classification, named entity recognition, question answering

Flow:
  Input text → Encoder (bidirectional) → Task-specific head
  
Can see entire sequence at once (bidirectional attention)
```

### 2. Decoder-Only (GPT-style)

```yaml
Use case: Generation tasks  
Examples: GPT, LLaMA, Claude
Tasks: Text generation, completion, chat

Flow:
  Input text → Decoder (causal) → Next token prediction
  
Can only see previous tokens (causal masking)
```

### 3. Encoder-Decoder (Original Transformer)

```yaml
Use case: Sequence-to-sequence
Examples: T5, BART, translation models
Tasks: Translation, summarization, text-to-text

Flow:
  Source → Encoder → Cross-attention ← Decoder ← Target
  
Encoder sees full source, Decoder generates target
```

## The Big Picture: Component Overview

Let's break down the transformer architecture into its essential components:

**Transformer = Embedding → N × Blocks → Output**

```yaml
1. Input Processing:
   - Token Embedding (word → vector)
   - Positional Encoding (position info)
   
2. N × Transformer Blocks (typically N=6 to 96):
   - Multi-Head Self-Attention (context gathering)
   - Feed-Forward Network (feature transformation)
   - Layer Normalization (training stability)
   - Residual Connections (gradient flow)
   
3. Output Layer:
   - Final normalization
   - Linear projection to vocabulary
```

## Building a Single Transformer Block

Let's build the fundamental unit - one transformer block. Understanding this is key because transformers are just these blocks stacked!

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        
        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # Attention block
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)  # Residual connection
        
        # Feedforward block
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)  # Residual connection
        
        return x

# Test
block = TransformerBlock(embed_dim=512, num_heads=8, ff_dim=2048)
x = torch.randn(10, 32, 512)  # (seq, batch, embed)
output = block(x)
print(output.shape)  # torch.Size([10, 32, 512])
```

## Complete Transformer

### Step 1: Initialize Embeddings

```python
class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, 
                 num_layers=6, ff_dim=2048, max_seq_len=5000):
        super().__init__()
        
        # Token embeddings: Convert word IDs to vectors
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional embeddings: Add position information
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
```

**Why two embeddings?**
- Token embedding: "What is this word?"
- Positional embedding: "Where is this word in the sentence?"

Without positional info, "cat sat" = "sat cat" (order lost!)

### Step 2: Stack Transformer Blocks

```python
        # Create N identical transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
```

**Typical configurations:**
```
GPT-2 small: 12 layers
GPT-2 large: 36 layers
GPT-3: 96 layers
LLaMA-2: 32-80 layers
```

More layers = more capacity, but harder to train!

### Step 3: Output Projection

```python
        # Final normalization and projection
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
```

**Purpose:** Convert final hidden states back to vocabulary space for predictions.

### Step 4: Forward Pass - Embeddings

```python
    def forward(self, x):
        # x shape: (batch, seq_len)
        # Example: (32, 100) = 32 sequences of 100 token IDs
        
        batch, seq_len = x.size()
```

**Input:** Token IDs (integers), not embeddings yet!

```python
        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        # Shape: (1, seq_len) → broadcasts to (batch, seq_len)
```

**Positions:** [0, 1, 2, 3, ..., seq_len-1]

```python
        # Embed tokens and add position information
        x = self.token_embedding(x) + self.pos_embedding(positions)
        # Shape: (batch, seq_len, embed_dim)
```

**The combination:**
```
Token embedding: "This is the word 'cat'"
  + 
Position embedding: "This is at position 5"
  =
Full embedding: "The word 'cat' at position 5"
```

### Step 5: Apply All Transformer Blocks

```python
        # Pass through all transformer blocks sequentially
        for block in self.blocks:
            x = block(x)
        # Each block adds more processing depth
```

**The stacking:**
```
Input → Block₁ → Block₂ → ... → Block_N → Output

Each block:
  - Adds more context understanding
  - Extracts higher-level features
  - Refines representations
```

### Step 6: Final Output Projection

```python
        # Final layer normalization
        x = self.ln_f(x)
        
        # Project to vocabulary size
        logits = self.head(x)
        # Shape: (batch, seq_len, vocab_size)
        
        return logits
```

**Logits interpretation:**
```
For each position, we get scores for all possible next tokens:
logits[i, j, :] = scores for vocab_size words at position j of sequence i

Example: vocab_size=50000
  logits[0, 5, 1234] = score for word #1234 at position 5
```

### Creating and Using the Model

```python
# Create transformer (GPT-2 small configuration)
model = Transformer(
    vocab_size=50000,
    embed_dim=512,
    num_heads=8,
    num_layers=12,
    ff_dim=2048
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
# ~100M parameters!

# Test forward pass
input_ids = torch.randint(0, 50000, (4, 20))  # 4 sequences, 20 tokens
output = model(input_ids)

print(f"Input shape: {input_ids.shape}")  # torch.Size([4, 20])
print(f"Output shape: {output.shape}")    # torch.Size([4, 20, 50000])
# For each of 20 positions, scores for all 50000 words!
```

## Understanding the Data Flow

Let's trace a single token through the entire transformer:

### Token Journey Through Transformer

```
Input token ID: 1234 (the word "cat" at position 5)
  ↓
Embedding layer: 
  1234 → [0.1, 0.3, -0.2, ..., 0.5] (512-dim vector)
  +
  Position 5 → [0.0, 0.1, 0.0, ..., -0.1] (512-dim vector)
  =
  Combined: [0.1, 0.4, -0.2, ..., 0.4]
  ↓
Block 1 (Attention + FFN):
  Gathers context from all other tokens
  → [0.2, 0.3, -0.1, ..., 0.6]
  ↓
Block 2:
  Further refines with more context
  → [0.3, 0.2, 0.0, ..., 0.7]
  ↓
... (blocks 3-12)
  ↓
Block 12:
  Final high-level representation
  → [0.4, 0.1, 0.2, ..., 0.8]
  ↓
Output projection:
  [0.4, 0.1, 0.2, ..., 0.8] → 50,000 scores
  
  Scores for next token:
    "dog": 2.3
    "sat": 4.1  ← Highest!
    "ran": 1.8
    ...
  ↓
Prediction: "sat" (highest score)
```

## The Key Innovation: Parallel Processing

**Why transformers are fast:**

```python
# Transformer processes ALL positions at once
x = torch.randn(1, 100, 512)  # 100 tokens

# One forward pass processes everything in parallel!
output = model(x)  
# All 100 tokens processed simultaneously

# Compare to RNN:
# for i in range(100):
#     hidden = rnn_step(x[i], hidden)  # Sequential! Slow!
```

**Speed comparison:**
```
RNN: 100 sequential steps
Transformer: 1 parallel operation

GPU utilization:
RNN: ~30% (sequential bottleneck)
Transformer: ~95% (fully parallel)
```

## Essential Components Summary

```yaml
1. Embeddings (Input Processing):
   Token: Maps vocabulary to vectors
   Position: Encodes sequence order
   Combined: Word identity + position info

2. Transformer Blocks (Repeated N times):
   Multi-Head Attention: Gather context from all positions
   FFN: Transform each position independently
   LayerNorm: Stabilize training (normalize distributions)
   Residual: Help gradients flow (x + sublayer(x))

3. Output Layer:
   Final LayerNorm: Last normalization
   Linear: Project to vocabulary size
   Softmax: Convert to probabilities (done in loss function)
```

## Key Takeaways

✓ **Self-attention based:** No recurrence, no convolution

✓ **Parallel:** Processes entire sequence at once

✓ **Scalable:** Stack more blocks for more capacity

✓ **Powerful:** Powers GPT, BERT, LLaMA

**Remember:** Transformers are just stacked attention blocks! 🎉

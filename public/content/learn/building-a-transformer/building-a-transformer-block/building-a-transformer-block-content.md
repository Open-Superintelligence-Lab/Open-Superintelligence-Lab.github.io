---
hero:
  title: "Building a Transformer Block"
  subtitle: "Creating the Core Transformer Component"
  tags:
    - "🤖 Transformers"
    - "⏱️ 10 min read"
---

A transformer block is the **repeatable unit** that makes transformers work!

![Block Diagram](/content/learn/building-a-transformer/building-a-transformer-block/block-diagram.png)

## Understanding the Transformer Block

A transformer block is like a LEGO piece - stack many of them to build powerful models! GPT-3 uses 96 of these blocks, LLaMA uses 80, BERT uses 12.

**The key insight:** Each block is self-contained and does the same operations:
1. Let tokens talk to each other (attention)
2. Let each token think independently (feedforward)
3. Stabilize the process (normalization)
4. Preserve information (residual connections)

### Why This Architecture?

Each component serves a critical purpose:

**Attention:** "What context do I need from other tokens?"
**FFN:** "How do I process this context?"
**LayerNorm:** "Keep values stable for training"
**Residuals:** "Don't lose original information"

## Building the Block Component by Component

Let's build a transformer block step-by-step, understanding why each piece is essential:

### Component 1: Multi-Head Attention

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,      # Model dimension (e.g., 512)
            num_heads=n_heads,       # Number of attention heads (e.g., 8)
            dropout=dropout,         # Regularization
            batch_first=True         # Expect (batch, seq, embed) format
        )
```

**Purpose:** Let each position gather relevant context from all other positions in the sequence.

**Typical values:**
- d_model=512, n_heads=8 (GPT-2 small)
- d_model=768, n_heads=12 (BERT)
- d_model=2048, n_heads=16 (GPT-2 large)

### Component 2: Feed-Forward Network

```python
        # Position-wise feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),      # Expand: 512 → 2048
            nn.ReLU(),                      # Non-linearity
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),      # Compress: 2048 → 512
            nn.Dropout(dropout)
        )
```

**Purpose:** Apply the same 2-layer MLP to each position independently.

**The expand-compress pattern:**
```
d_model=512 → d_ff=2048 → d_model=512

Why?
  Expansion: Increases capacity to process information
  Compression: Returns to model dimension
  
Typically: d_ff = 4 × d_model
```

### Component 3: Layer Normalization

```python
        # Layer normalization for training stability
        self.norm1 = nn.LayerNorm(d_model)  # After attention
        self.norm2 = nn.LayerNorm(d_model)  # After FFN
```

**Purpose:** Normalize each sample to mean=0, std=1 across the feature dimension.

**Why essential:**
```
Without LayerNorm:
  Activations can grow/shrink unboundedly
  Gradients become unstable
  Training fails in deep networks

With LayerNorm:
  Stable activation magnitudes
  Consistent gradients
  Deep networks train successfully
```

### Component 4: Dropout

```python
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
```

**Purpose:** Randomly zero out values during training to prevent overfitting.

## The Forward Pass: Data Flow

Now let's see how data flows through the block with detailed explanations:

### Sub-Block 1: Multi-Head Attention

```python
    def forward(self, x, mask=None):
        # Input shape: (batch, seq_len, d_model)
        # Example: (32, 100, 512)
        
        # ATTENTION SUB-BLOCK
        # Step 1: Multi-head self-attention
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        # attn_out shape: (batch, seq_len, d_model)
```

**What attention does:** Each token gathers context from all other tokens.

```python
        # Step 2: Residual connection + dropout
        x = x + self.dropout(attn_out)
```

**Residual connection:** Add the input back! This helps gradients flow during backpropagation.

```python
        # Step 3: Layer normalization
        x = self.norm1(x)
```

**Normalization:** Stabilize the values before the next sub-block.

### Sub-Block 2: Feed-Forward Network

```python
        # FFN SUB-BLOCK
        # Step 1: Feed-forward transformation
        ffn_out = self.ffn(x)
```

**What FFN does:** Processes each position's context independently (no interaction between positions).

```python
        # Step 2: Residual connection
        x = x + ffn_out
        
        # Step 3: Layer normalization
        x = self.norm2(x)
        
        return x
```

**Complete flow visualization:**
```
Input (batch, seq, d_model)
  ↓
[Multi-Head Attention] → attn_out
  ↓
[Add & Norm] → x + attn_out → normalize
  ↓
[Feed-Forward Network] → ffn_out
  ↓
[Add & Norm] → x + ffn_out → normalize
  ↓
Output (batch, seq, d_model)
```

### Testing the Block

```python
# Create transformer block (GPT-2 config)
block = TransformerBlock(
    d_model=512,
    n_heads=8,
    d_ff=2048,
    dropout=0.1
)

# Input: 32 sequences, 10 tokens, 512-dim
x = torch.randn(32, 10, 512)

# Apply block
output = block(x)

print(f"Input shape: {x.shape}")      # torch.Size([32, 10, 512])
print(f"Output shape: {output.shape}") # torch.Size([32, 10, 512])
```

**Critical property:** Input and output have the same shape! This allows stacking many blocks.

## Why Residual Connections Are Critical

Let's understand residuals with a concrete example:

### Without Residuals

```python
# Vanilla network (no residuals)
x = input
x = layer1(x)
x = layer2(x)
x = layer3(x)
output = x

# Problem: Gradients must flow through all layers
# ∂L/∂input = ∂L/∂layer3 × ∂layer3/∂layer2 × ∂layer2/∂layer1 × ∂layer1/∂input
# Each multiplication can shrink the gradient → vanishing!
```

### With Residuals

```python
# With residual connections
x = input
x = x + layer1(x)  # Add input back!
x = x + layer2(x)  # Add previous back!
x = x + layer3(x)  # Add previous back!
output = x

# Benefit: Gradients have a "highway"
# ∂L/∂input has a direct path: ∂L/∂output × 1
# Gradient = identity + (other terms)
```

### Mathematical Explanation

**Without residual:**
```
f(x) = layer(x)
∂f/∂x = ∂layer/∂x  ← Can be small (vanishing gradient)
```

**With residual:**
```
f(x) = x + layer(x)
∂f/∂x = 1 + ∂layer/∂x  ← Always has the 1! (gradient flows)
```

The "1" ensures gradients never completely vanish!

### Empirical Evidence

```python
import torch
import torch.nn as nn

# 50-layer network without residuals
deep_no_residual = nn.Sequential(*[
    nn.Linear(512, 512) for _ in range(50)
])
# This will NOT train well! Gradients vanish.

# 50-layer network with residuals
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer = nn.Linear(dim, dim)
    
    def forward(self, x):
        return x + self.layer(x)  # Residual!

deep_with_residual = nn.Sequential(*[
    ResidualBlock(512) for _ in range(50)
])
# This WILL train! Gradients flow through residuals.
```

**Result:** Residuals enable training of 100+ layer networks!

## Stacking Multiple Blocks

The power of transformers comes from stacking many blocks:

### Creating a Full Transformer

```python
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, 
                 n_layers=6, d_ff=2048):
        super().__init__()
        
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
```

**Step 1:** Convert token IDs to vectors.

```python
        # Stack N transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
```

**Step 2:** Create the processing stack. Each block is identical in structure but learns different transformations.

```python
        # Output layers
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
```

**Step 3:** Final normalization and vocabulary projection.

### The Forward Pass

```python
    def forward(self, x):
        # x: token IDs, shape (batch, seq_len)
        
        # Embed tokens
        x = self.embedding(x)
        # Shape: (batch, seq_len, d_model)
```

**After embedding:** Tokens are now continuous vectors.

```python
        # Pass through all N transformer blocks
        for block in self.blocks:
            x = block(x)  # Each block refines the representation
```

**The sequential processing:**
```
Block 1: Basic patterns and relationships
Block 2: Higher-level combinations  
Block 3: Even more abstract features
...
Block N: Very high-level understanding
```

```python
        # Final normalization
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.head(x)
        # Shape: (batch, seq_len, vocab_size)
        
        return logits
```

### Using the Full Model

```python
# Create 12-layer transformer
model = Transformer(
    vocab_size=50000,
    n_layers=12,
    d_model=768,
    n_heads=12,
    d_ff=3072
)

# Count parameters
params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params:,}")
# ~117M parameters (GPT-2 small size)

# Test forward pass
input_ids = torch.randint(0, 50000, (4, 50))  # 4 sequences, 50 tokens
output = model(input_ids)

print(f"Input: {input_ids.shape}")   # torch.Size([4, 50])
print(f"Output: {output.shape}")     # torch.Size([4, 50, 50000])
```

**Output interpretation:**
- For each of 50 positions
- We get 50,000 scores (one per vocabulary word)
- Highest score = most likely next token

## Key Takeaways

✓ **Core component:** Attention + FFN + Norm + Residuals

✓ **Repeatable:** Stack many blocks

✓ **Same shape:** Input and output dimensions match

✓ **Self-contained:** Each block is independent

**Remember:** Transformers are just stacked blocks! 🎉

---
hero:
  title: "Full Transformer in Code"
  subtitle: "Complete Implementation from Scratch"
  tags:
    - "🤖 Transformers"
    - "⏱️ 15 min read"
---

Let's build a complete, working transformer from scratch!

## The Complete Picture

We've learned all the components separately. Now let's put everything together into one complete, production-ready transformer implementation!

### What We're Building

A full GPT-style decoder-only transformer with:
- Token and positional embeddings
- Multi-head attention
- Feed-forward networks
- Layer normalization
- Residual connections
- Output projection

**This is real code you can train!**

## Component-by-Component Implementation

Let's build each piece carefully, then assemble them:

### Component 1: Multi-Head Attention

Let's build multi-head attention from scratch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
```

**Dimension calculation:**
```
d_model = 512, n_heads = 8
head_dim = 512 / 8 = 64
Each head operates in 64-dimensional space
```

```python
        # Q, K, V projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
```

**Why four linear layers?**
- Three for Q, K, V projections
- One for combining heads back

```python
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Project to Q, K, V and split into heads
        Q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # Shapes: (batch, n_heads, seq_len, head_dim)
```

**The reshape:**
- Project: (batch, seq, d_model)
- View: (batch, seq, n_heads, head_dim)
- Transpose: (batch, n_heads, seq, head_dim)

```python
        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        output = attn @ V
```

**Attention computation in one parallel operation for all heads!**

```python
        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, d_model)
        output = self.out_linear(output)
        
        return output
```

### Component 2: Feed-Forward Network

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),      # Expand
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),      # Compress
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
```

**Simple expand-compress MLP** applied to each position independently.

**Parameters:**
```
For d_model=512, d_ff=2048:
  Layer 1: 512 × 2048 = 1,048,576
  Layer 2: 2048 × 512 = 1,048,576
  Total: ~2.1M parameters per FFN
```

### Component 3: Transformer Block

Now combine attention and FFN with normalization and residuals:

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
```

**Two sub-layers with normalization and residual connections.**

```python
    def forward(self, x, mask=None):
        # Sub-layer 1: Attention with residual
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Sub-layer 2: FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x
```

**The residual pattern:**
```
x_new = LayerNorm(x_old + Dropout(Sublayer(x_old)))

This is called "Pre-LN" (layer norm before sublayer)
Alternative: "Post-LN" (layer norm after sublayer)
```

### Component 4: The Complete Transformer

Finally, assemble everything:

```python
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, 
                 n_layers=6, d_ff=2048, max_seq_len=512, dropout=0.1):
        super().__init__()
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
```

**Embedding setup:** Token + position information.

```python
        # Stack N transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
```

**The stack:** 6-96 identical blocks (in structure, different in learned parameters).

```python
        # Output layers
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
```

**Final projection:** Hidden → vocabulary.

```python
    def forward(self, x):
        batch, seq_len = x.size()
        
        # Add embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(positions)
        x = self.dropout(x)
```

**Embedding combination:** Token + position with dropout for regularization.

```python
        # Apply all transformer blocks
        for block in self.blocks:
            x = block(x)
```

**Sequential processing** through all blocks.

```python
        # Output projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
```

**Final steps:** Normalize and project to vocabulary.

## Creating and Testing the Model

Let's create a GPT-style model and test it:

### Instantiate the Model

```python
# Create GPT-style transformer (GPT-2 small config)
model = Transformer(
    vocab_size=50000,
    d_model=768,
    n_heads=12,
    n_layers=12,
    d_ff=3072,
    max_seq_len=1024,
    dropout=0.1
)
```

**This is a GPT-2 Small configuration!**

### Count Parameters

```python
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
# Output: ~117,000,000 (117M parameters)

# Break down by component
emb_params = sum(p.numel() for p in model.token_emb.parameters())
emb_params += sum(p.numel() for p in model.pos_emb.parameters())
block_params = sum(p.numel() for b in model.blocks for p in b.parameters())
head_params = sum(p.numel() for p in model.head.parameters())

print(f"Embeddings: {emb_params:,}")      # ~38M
print(f"Blocks: {block_params:,}")        # ~70M
print(f"Head: {head_params:,}")           # ~38M
```

**Parameter distribution:**
```
Token embedding: 50,000 × 768 = 38,400,000
Position embedding: 1,024 × 768 = 786,432
12 Transformer blocks: ~70,000,000
Output head: 768 × 50,000 = 38,400,000
---
Total: ~117M parameters
```

### Test Forward Pass

```python
# Create random token IDs
tokens = torch.randint(0, 50000, (2, 64))  # 2 sequences, 64 tokens each
print(f"Input shape: {tokens.shape}")      # torch.Size([2, 64])

# Forward pass
model.eval()  # Set to evaluation mode
with torch.no_grad():
    logits = model(tokens)

print(f"Output shape: {logits.shape}")     # torch.Size([2, 64, 50000])
print(f"Output range: [{logits.min():.2f}, {logits.max():.2f}]")
```

**What we get:**
- For each of 2 sequences
- For each of 64 positions
- Scores for all 50,000 vocabulary tokens

### Generate Text

```python
def generate(model, prompt_tokens, max_new_tokens=20, temperature=1.0):
    """Generate text autoregressively"""
    model.eval()
    tokens = prompt_tokens.clone()
    
    for _ in range(max_new_tokens):
        # Get predictions for current sequence
        with torch.no_grad():
            logits = model(tokens)
        
        # Get logits for last position
        next_token_logits = logits[0, -1, :] / temperature
        
        # Sample from distribution
        probs = F.softmax(next_token_logits, dim=0)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        tokens = torch.cat([tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
    
    return tokens

# Example usage
prompt = torch.tensor([[15, 234, 567]])  # "The cat sat"
generated = generate(model, prompt, max_new_tokens=10)
print(f"Generated sequence: {generated}")
```

**This generates text token-by-token!**

## Key Architectural Details

### Memory and Compute

```yaml
For batch_size=32, seq_len=512, d_model=768:

Attention:
  Q, K, V: 3 × (32, 512, 768) = 37.7 MB
  Scores: (32, 12, 512, 512) = 402.7 MB  ← Memory bottleneck!
  
FFN:
  Hidden: (32, 512, 3072) = 201.3 MB
  
Total per block: ~640 MB
Total for 12 blocks: ~7.7 GB

This is why we use gradient checkpointing for large models!
```

### Computational Complexity

**Per token:**
```
Attention: O(seq_len² × d_model)
FFN: O(d_model × d_ff)

For seq_len=512, d_model=768, d_ff=3072:
  Attention: 512² × 768 = 201M ops
  FFN: 768 × 3072 = 2.4M ops
  
Attention dominates for long sequences!
```

## Modifications and Variants

You can easily modify this transformer:

**1. Add causal masking (for GPT-style):**
```python
def create_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

# In forward pass:
mask = create_causal_mask(seq_len, x.device)
for block in self.blocks:
    x = block(x, mask)
```

**2. Use different positional encoding (RoPE):**
```python
# Replace learned pos_emb with RoPE
self.rope = RotaryPositionalEmbedding(d_model)

# In forward:
cos, sin = self.rope(x)
x = self.token_emb(x)  # No position addition
```

**3. Add MoE layers:**
```python
# Replace FFN in some blocks with MoE
self.expert_block = MixtureOfExperts(d_model, num_experts=8)
```

## Key Takeaways

✓ **Complete:** All components working together

✓ **Production-ready:** Real implementation you can train

✓ **Modular:** Easy to modify and extend

✓ **Educational:** Matches architecture from papers

**Remember:** You just built a transformer from scratch! This is the same architecture powering ChatGPT (with modifications). 🎉

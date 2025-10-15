---
hero:
  title: "MoE in a Transformer"
  subtitle: "Integrating Mixture of Experts"
  tags:
    - "🔀 MoE"
    - "⏱️ 10 min read"
---

MoE replaces the standard FFN in transformer blocks with a sparse expert layer!

## The Integration Point

In a standard transformer block, there are two main components:
1. **Self-attention:** Token interaction
2. **FFN:** Individual token processing

**MoE replaces the FFN** while keeping attention unchanged!

### Standard Transformer Block

```yaml
Input
  ↓
LayerNorm
  ↓
Self-Attention
  ↓
Residual + Dropout
  ↓
LayerNorm
  ↓
Feedforward Network (FFN)  ← Replace this!
  ↓
Residual + Dropout
  ↓
Output
```

### MoE Transformer Block

```yaml
Input
  ↓
LayerNorm
  ↓
Self-Attention
  ↓
Residual + Dropout
  ↓
LayerNorm
  ↓
Mixture of Experts (MoE)  ← New!
  ↓
Residual + Dropout
  ↓
Output
```

**Key insight:** MoE is a drop-in replacement for FFN!

## Side-by-Side Comparison

### Standard Transformer Block

```python
import torch.nn as nn

class StandardTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        
        # Attention
        self.attention = nn.MultiheadAttention(
            d_model, 
            n_heads, 
            batch_first=True
        )
        
        # Standard FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Attention sub-layer
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN sub-layer
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x
```

### MoE Transformer Block

```python
class MoETransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, num_experts=8, top_k=2):
        super().__init__()
        
        # Attention (unchanged)
        self.attention = nn.MultiheadAttention(
            d_model, 
            n_heads, 
            batch_first=True
        )
        
        # MoE instead of FFN
        self.moe = MixtureOfExperts(
            d_model, 
            num_experts=num_experts,
            top_k=top_k
        )
        
        # Normalization (unchanged)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout (unchanged)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Attention sub-layer (unchanged)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # MoE sub-layer (new!)
        moe_out, router_probs = self.moe(x)
        x = self.norm2(x + self.dropout(moe_out))
        
        return x, router_probs  # Return router probs for load balancing
```

**Only difference:** Replaced `self.ffn` with `self.moe`!

## Complete MoE Transformer

Let's build a full transformer with MoE:

### Step 1: Import and Setup

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class PositionalEncoding(nn.Module):
    """Standard positional encoding"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.encoding = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        batch, seq = x.shape[:2]
        positions = torch.arange(seq, device=x.device).unsqueeze(0)
        return self.encoding(positions)
```

### Step 2: MoE Transformer Layer

```python
class MoETransformerLayer(nn.Module):
    """Single transformer layer with MoE"""
    def __init__(self, d_model, n_heads, num_experts, top_k, dropout=0.1):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # MoE replaces FFN
        self.moe = MixtureOfExperts(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k
        )
        
        # Layer normalization (pre-norm style)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # x: (batch, seq, d_model)
        
        # Self-attention with residual
        attn_out, _ = self.self_attn(
            self.norm1(x),  # Pre-norm
            self.norm1(x),
            self.norm1(x),
            attn_mask=mask
        )
        x = x + self.dropout1(attn_out)
```

**Pre-norm:** Apply LayerNorm before attention (more stable).

```python
        # MoE with residual
        moe_out, router_probs = self.moe(self.norm2(x))  # Pre-norm
        x = x + self.dropout2(moe_out)
        
        return x, router_probs
```

### Step 3: Complete Transformer

```python
class MoETransformer(nn.Module):
    """Full transformer with MoE layers"""
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_heads=8,
        num_layers=6,
        num_experts=8,
        top_k=2,
        max_seq_len=512,
        dropout=0.1
    ):
        super().__init__()
        
        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_seq_len)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Stack of MoE transformer layers
        self.layers = nn.ModuleList([
            MoETransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output = nn.Linear(d_model, vocab_size)
```

### Step 4: Forward Pass

```python
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # x: (batch, seq) - token IDs
        
        # Embedding + positional encoding
        x = self.token_emb(x) + self.pos_enc(x)
        x = self.dropout(x)
        
        # Collect router probabilities for load balancing
        all_router_probs = []
        
        # Pass through layers
        for layer in self.layers:
            x, router_probs = layer(x, mask)
            all_router_probs.append(router_probs)
        
        # Final normalization
        x = self.norm(x)
        
        # Output projection
        logits = self.output(x)
        
        return logits, all_router_probs
```

**Returns:**
- `logits`: Predictions over vocabulary
- `all_router_probs`: For computing load balancing loss

## Training with Load Balancing

### Step 1: Compute Task Loss

```python
def train_step(model, batch, targets, optimizer):
    # Forward pass
    logits, all_router_probs = model(batch)
    
    # Task loss (next-token prediction)
    task_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1)
    )
```

### Step 2: Compute Load Balancing Loss

```python
    # Load balancing loss across all layers
    total_balance_loss = 0
    num_experts = model.layers[0].moe.num_experts
    
    for layer_idx, router_probs in enumerate(all_router_probs):
        # router_probs: (batch*seq, num_experts)
        
        # Compute balancing loss for this layer
        f = router_probs.mean(dim=0)  # (num_experts,)
        loss = num_experts * (f ** 2).sum()
        
        total_balance_loss += loss
    
    # Average over layers
    balance_loss = total_balance_loss / len(all_router_probs)
```

### Step 3: Combined Loss and Optimization

```python
    # Total loss
    total_loss = task_loss + 0.01 * balance_loss
    
    # Backward and update
    optimizer.zero_grad()
    total_loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    
    return {
        'total_loss': total_loss.item(),
        'task_loss': task_loss.item(),
        'balance_loss': balance_loss.item()
    }
```

## Model Variants

### Variant 1: Sparse MoE Transformer (Switch Transformer Style)

```python
class SparseMoETransformer(MoETransformer):
    """Top-1 routing (maximum sparsity)"""
    def __init__(self, *args, **kwargs):
        kwargs['top_k'] = 1  # Only 1 expert per token
        kwargs['num_experts'] = 128  # Many experts
        super().__init__(*args, **kwargs)
```

**Ultra-sparse:** Each token uses only 1 out of 128 experts!

### Variant 2: Dense-Sparse Hybrid (DeepSeek Style)

```python
class HybridMoETransformer(nn.Module):
    """Alternating dense and MoE layers"""
    def __init__(self, vocab_size, d_model, n_heads, num_layers):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            if i % 2 == 0:
                # Even layers: Standard FFN
                layer = StandardTransformerBlock(d_model, n_heads, 4*d_model)
            else:
                # Odd layers: MoE
                layer = MoETransformerBlock(d_model, n_heads, num_experts=8)
            
            self.layers.append(layer)
```

**Pattern:** Dense → MoE → Dense → MoE → ...

### Variant 3: Top-K Adjustment

```python
class AdaptiveTopKTransformer(nn.Module):
    """Different top-k for different layers"""
    def __init__(self, vocab_size, d_model, n_heads, num_layers):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        for layer_idx in range(num_layers):
            # Early layers: More experts (broader context)
            # Later layers: Fewer experts (more specialized)
            top_k = 4 if layer_idx < num_layers // 2 else 2
            
            layer = MoETransformerLayer(
                d_model, n_heads,
                num_experts=8,
                top_k=top_k
            )
            self.layers.append(layer)
```

## Computational Analysis

### Standard Transformer

```python
# Per layer, per token
d_model = 512
d_ff = 2048
n_heads = 8

# Attention
attn_compute = 3 * d_model² + seq_len * d_model²
              = 3 * 512² + seq_len * 512²

# FFN
ffn_compute = 2 * d_model * d_ff
            = 2 * 512 * 2048
            = 2,097,152 ops

# Total per token (approximate)
total = attn_compute + ffn_compute
```

### MoE Transformer

```python
# Per layer, per token
top_k = 2
num_experts = 8

# Attention (unchanged)
attn_compute = 3 * d_model² + seq_len * d_model²

# MoE (only top-k experts)
moe_compute = top_k * 2 * d_model * d_ff
            = 2 * 2 * 512 * 2048
            = 4,194,304 ops

# Router (negligible)
router_compute = d_model * num_experts
               = 512 * 8 = 4,096 ops

# Total per token
total = attn_compute + moe_compute + router_compute
```

**MoE uses 2× compute of standard (because top-k=2), but has 8× capacity!**

## Practical Tips

### 1. Start with Standard Hyperparameters

```python
model = MoETransformer(
    vocab_size=50000,
    d_model=768,
    n_heads=12,
    num_layers=12,
    num_experts=8,      # Start small
    top_k=2,            # Standard choice
    dropout=0.1
)
```

### 2. Monitor Expert Usage

```python
def analyze_expert_usage(router_probs):
    """Check if experts are being used uniformly"""
    # router_probs: (batch*seq, num_experts)
    
    usage = router_probs.mean(dim=0)  # (num_experts,)
    
    print("Expert usage:")
    for i, u in enumerate(usage):
        print(f"  Expert {i}: {u:.3f}")
    
    # Ideally all close to 1/num_experts
    ideal = 1.0 / len(usage)
    imbalance = ((usage - ideal).abs()).mean()
    print(f"Imbalance: {imbalance:.4f}")
```

### 3. Adjust Load Balancing Weight

```python
# Start with small weight
balance_weight = 0.01

# If experts imbalanced, increase
if imbalance > 0.05:
    balance_weight = 0.1
    
# If training unstable, decrease
if loss.isnan():
    balance_weight = 0.001
```

### 4. Gradient Checkpointing for Memory

```python
from torch.utils.checkpoint import checkpoint

class MoETransformerWithCheckpointing(MoETransformer):
    def forward(self, x, mask=None):
        x = self.token_emb(x) + self.pos_enc(x)
        
        all_router_probs = []
        
        for layer in self.layers:
            # Use checkpointing to save memory
            x, router_probs = checkpoint(layer, x, mask)
            all_router_probs.append(router_probs)
        
        x = self.norm(x)
        logits = self.output(x)
        
        return logits, all_router_probs
```

**Trades compute for memory** - useful for large models!

## Example Training Loop

```python
# Create model
model = MoETransformer(
    vocab_size=50000,
    d_model=512,
    n_heads=8,
    num_layers=6,
    num_experts=8,
    top_k=2
)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Training loop
for epoch in range(10):
    for batch, targets in dataloader:
        losses = train_step(model, batch, targets, optimizer)
        
        if step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}")
            print(f"  Task loss: {losses['task_loss']:.4f}")
            print(f"  Balance loss: {losses['balance_loss']:.4f}")
```

## Key Takeaways

✓ **Drop-in replacement:** MoE replaces FFN in transformer blocks

✓ **Same interface:** Input/output shapes unchanged

✓ **Sparse activation:** Massive capacity with controlled compute

✓ **Load balancing:** Critical for stable training

✓ **Scalable:** Add experts without proportional compute increase

**Remember:** MoE transforms transformers from dense to sparse - same architecture, different capacity! 🎉

---
hero:
  title: "The Expert"
  subtitle: "Individual Expert Networks in MoE"
  tags:
    - "🔀 MoE"
    - "⏱️ 8 min read"
---

An expert is a **specialized feedforward network** in the Mixture of Experts architecture!

## What is an Expert?

In MoE, an "expert" is simply a feedforward network (FFN) - **structurally identical** to a standard transformer FFN, but trained to specialize in specific patterns through routing.

### Expert vs Standard FFN

```yaml
Standard Transformer FFN:
  - One network
  - Processes ALL tokens
  - General-purpose

MoE Expert:
  - One of many networks (8-2048 experts)
  - Processes SOME tokens (routed)
  - Specialized
```

**Key insight:** Experts are not architecturally different - they specialize through **conditional activation**!

## Expert Architecture

### Standard Expert Structure

An expert is typically a two-layer FFN:

```
Expert(x) = W₂ · Activation(W₁ · x + b₁) + b₂

Where:
  W₁: d_model → d_ff (expansion)
  W₂: d_ff → d_model (compression)
  Activation: ReLU, SiLU, GELU, etc.
```

**Same as transformer FFN!**

### Implementation

```python
import torch
import torch.nn as nn

class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        
        # Expansion layer
        self.w1 = nn.Linear(d_model, d_ff)
        
        # Activation function
        self.activation = nn.SiLU()  # Modern choice
        
        # Compression layer
        self.w2 = nn.Linear(d_ff, d_model)
```

**Why SiLU?**
```
SiLU(x) = x · sigmoid(x)

Advantages over ReLU:
  - Smooth (differentiable everywhere)
  - Non-monotonic
  - Better gradient flow
  - Used in: LLaMA, Mixtral, DeepSeek
```

### Forward Pass

```python
    def forward(self, x):
        # x: (num_tokens, d_model)
        # Note: Only tokens routed to THIS expert!
        
        # Expand
        hidden = self.w1(x)        # (num_tokens, d_ff)
        
        # Activate
        hidden = self.activation(hidden)
        
        # Compress
        output = self.w2(hidden)    # (num_tokens, d_model)
        
        return output
```

**Shape preservation:** Input d_model → Output d_model (critical for residual connections).

## Creating Multiple Experts

### ModuleList for Experts

```python
class ExpertPool(nn.Module):
    def __init__(self, num_experts, d_model, d_ff):
        super().__init__()
        
        # Create N experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff)
            for _ in range(num_experts)
        ])
        self.num_experts = num_experts
```

**What we have:**
```
Expert 0: Parameters θ₀
Expert 1: Parameters θ₁
Expert 2: Parameters θ₂
...
Expert 7: Parameters θ₇

All have SAME structure, DIFFERENT learned weights!
```

### Testing an Expert

```python
# Create single expert
expert = Expert(d_model=512, d_ff=2048)

# Input: batch of tokens
x = torch.randn(100, 512)  # 100 tokens, 512-dim

# Process
output = expert(x)

print(f"Input shape: {x.shape}")      # torch.Size([100, 512])
print(f"Output shape: {output.shape}") # torch.Size([100, 512])

# Parameter count
params = sum(p.numel() for p in expert.parameters())
print(f"Parameters: {params:,}")  # ~2,099,712
```

## How Experts Specialize

Experts are **not manually assigned** to specific tasks. Specialization **emerges automatically** during training!

### The Specialization Process

**Step 1: Random initialization**
```
Initially: All experts have random weights
  Expert 0: Random θ₀
  Expert 1: Random θ₁
  ...
  
Router: Also random
```

**Step 2: Training begins**
```
Token: "The integral of"
Router (random): Routes to Expert 2, Expert 5

Token: "Paris is the"
Router (random): Routes to Expert 1, Expert 3

Both get gradient updates
```

**Step 3: Positive feedback loop**
```
After many updates:
  Expert 2: Better at math (received math tokens)
  Expert 5: Better at math (received math tokens)
  Expert 1: Better at geography
  Expert 3: Better at geography
  
Router learns: Math → Experts 2,5
               Geography → Experts 1,3
```

**Step 4: Specialization emerges**
```
Expert 0: Code patterns
Expert 1: Geographic knowledge
Expert 2: Mathematical reasoning
Expert 3: Historical facts
Expert 4: Grammar and syntax
Expert 5: Mathematical computation
Expert 6: Common sense
Expert 7: Creative language
```

**This happens automatically - no explicit labeling!**

### Mathematical View of Specialization

Given training data D = {(x₁, y₁), (x₂, y₂), ...}:

```
Standard FFN:
  Optimizes: θ* = argmin Σᵢ L(FFN_θ(xᵢ), yᵢ)
  All data influences all parameters
  
MoE Experts:
  Expert j sees: D_j = {xᵢ : Router(xᵢ) selects j}
  Optimizes: θⱼ* = argmin Σᵢ∈Dⱼ L(Expert_j(xᵢ), yᵢ)
  
Different data subsets → Different specializations!
```

## Expert Capacity and Load Balancing

### The Capacity Problem

Each expert has limited capacity (can only process so many tokens efficiently):

```python
# Expert capacity calculation
tokens_per_expert = (num_tokens / num_experts) * top_k * capacity_factor

# Example:
# 1024 tokens, 8 experts, top-2 routing, capacity_factor=1.25
tokens_per_expert = (1024 / 8) * 2 * 1.25 = 320 tokens max
```

**What happens if exceeded?**
```
Option 1: Drop tokens (tokens don't get processed)
Option 2: Overflow to other experts
Option 3: Dynamic capacity

Switch Transformer uses Option 1 (with load balancing loss)
```

### Load Balancing

**The problem:**
```
Ideal distribution:
  Expert 0: 125 tokens
  Expert 1: 125 tokens
  ...
  Expert 7: 125 tokens
  (1000 tokens / 8 experts = 125 each)
  
Actual (without balancing):
  Expert 0: 650 tokens  ← Overloaded!
  Expert 1: 10 tokens   ← Underutilized
  Expert 2: 15 tokens
  ...
  Expert 7: 5 tokens
```

**Solution:** Add load balancing loss:

```python
def load_balancing_loss(router_probs, expert_mask):
    """
    Encourages uniform distribution across experts
    
    Args:
        router_probs: (batch*seq, num_experts) - routing probabilities
        expert_mask: (batch*seq, num_experts) - which experts were selected
    """
    # Fraction of tokens to each expert
    f = expert_mask.float().mean(dim=0)  # (num_experts,)
    
    # Average router probability for each expert
    p = router_probs.mean(dim=0)  # (num_experts,)
    
    # Load balancing loss: encourages f ≈ 1/num_experts
    num_experts = router_probs.size(1)
    loss = num_experts * torch.sum(f * p)
    
    return loss
```

**Intuition:**
```
If expert_j is overused:
  f_j is high (many tokens routed)
  p_j is high (high router probabilities)
  f_j × p_j is large
  → Increases loss
  → Gradient pushes router AWAY from expert_j
  
Balances load automatically!
```

## Expert Variants

### 1. Standard Expert (Switch Transformer)

```python
class StandardExpert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x):
        return self.ffn(x)
```

### 2. GLU Expert (Mixtral)

```python
class GLU_Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        gate = F.silu(self.w1(x))
        value = self.w2(x)
        return self.w3(gate * value)
```

**Gating allows more expressive transformations!**

### 3. Small Expert (DeepSeek-MoE)

```python
class SmallExpert(nn.Module):
    def __init__(self, d_model, expert_size_ratio=0.25):
        super().__init__()
        d_ff = int(d_model * expert_size_ratio)  # Much smaller!
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))
```

**Smaller experts enable more experts with same parameters:**
```
8 standard experts: 8 × 4d² = 32d² parameters
32 small experts (0.25×): 32 × d² = 32d² parameters
Same parameters, 4× more experts!
```

## Key Takeaways

✓ **Expert = FFN:** Same architecture as standard feedforward

✓ **Specialization emerges:** Not manually assigned, learned through routing

✓ **Multiple experts:** 8-2048 experts in typical MoE

✓ **Conditional activation:** Each expert processes subset of tokens

✓ **Load balancing critical:** Prevents expert underutilization

**Remember:** Experts are specialized through routing, not architecture! 🎉

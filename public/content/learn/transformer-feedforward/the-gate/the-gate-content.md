---
hero:
  title: "The Gate"
  subtitle: "Router Network in Mixture of Experts"
  tags:
    - "🔀 MoE"
    - "⏱️ 8 min read"
---

The gate (router) decides **which experts each token should use**!

## The Router's Critical Role

The router is the **decision-making component** of MoE - it determines which experts process which tokens. This single component enables the entire sparse computation paradigm!

### What the Router Does

```yaml
Input: Token representation (d_model dimensions)
Output: Expert selection + weights

For each token:
  1. Compute scores for all N experts
  2. Select top-K highest-scoring experts
  3. Assign weights to selected experts
  4. Route token to those experts
```

**The router is a learned function!** Not hand-crafted rules.

## Mathematical Formulation

### Router Function

```
G(x) = softmax(Wᵣ · x + bᵣ)

Where:
  x ∈ ℝ^d_model         (input token)
  Wᵣ ∈ ℝ^(num_experts × d_model)  (router weights)
  G(x) ∈ ℝ^num_experts   (expert probabilities)
  
  G(x)ᵢ = probability of routing to expert i
  Σᵢ G(x)ᵢ = 1  (probabilities sum to 1)
```

### Top-K Selection

```
Top-K(G(x)) = indices of K largest values in G(x)

Example with K=2, num_experts=8:
  G(x) = [0.05, 0.32, 0.08, 0.15, 0.03, 0.28, 0.04, 0.05]
  Top-2 indices = [1, 5]  (expert 1 and expert 5)
  Top-2 weights = [0.32, 0.28]
```

### Renormalization

After selecting top-K, renormalize weights to sum to 1:

```
G'(x)ᵢ = G(x)ᵢ / Σⱼ∈TopK G(x)ⱼ  for i ∈ TopK
       = 0                      otherwise

Example:
  Before: [0.32, 0.28] → sum = 0.60
  After:  [0.533, 0.467] → sum = 1.0
```

**Why renormalize?** Ensures weighted combination is properly scaled.

## Implementation Step-by-Step

### Step 1: Define the Router

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Router(nn.Module):
    def __init__(self, d_model, num_experts):
        super().__init__()
        
        # Linear layer: d_model → num_experts
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.num_experts = num_experts
```

**The gate is just a linear layer!**
```
Parameters: d_model × num_experts

Example:
  d_model=512, num_experts=8
  Parameters = 512 × 8 = 4,096
  
Tiny compared to expert parameters!
```

### Step 2: Compute Router Logits

```python
    def forward(self, x, top_k=2):
        # x: (batch, seq_len, d_model)
        # Example: (2, 10, 512)
        
        # Compute logits for each expert
        router_logits = self.gate(x)
        # Shape: (2, 10, 8) - 8 scores per token
```

**What are logits?**
```
Raw, unnormalized scores
Can be any real number: (-∞, +∞)

Example for one token:
  logits = [2.3, -1.5, 0.8, 3.1, -0.2, 1.9, 0.3, -0.7]
  
Higher value = model thinks expert is more relevant
```

### Step 3: Convert to Probabilities

```python
        # Softmax: convert logits → probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        # Shape: (2, 10, 8)
```

**Softmax formula:**
```
softmax(zᵢ) = exp(zᵢ) / Σⱼ exp(zⱼ)

Properties:
  - Output in (0, 1)
  - Sum to 1
  - Differentiable (gradients flow!)
  
Example:
  logits = [2.3, -1.5, 0.8]
  exp = [9.97, 0.22, 2.23]
  sum = 12.42
  probs = [0.803, 0.018, 0.179]
```

### Step 4: Select Top-K Experts

```python
        # Get top-K experts per token
        top_k_probs, top_k_indices = torch.topk(
            router_probs, 
            k=top_k, 
            dim=-1
        )
        # top_k_probs: (2, 10, 2) - weights
        # top_k_indices: (2, 10, 2) - expert IDs
```

**Detailed example for one token:**
```
router_probs = [0.05, 0.32, 0.08, 0.15, 0.03, 0.28, 0.04, 0.05]

torch.topk(probs, k=2):
  values: [0.32, 0.28]  ← Top 2 probabilities
  indices: [1, 5]       ← Experts 1 and 5
  
Token will be routed to experts 1 and 5!
```

### Step 5: Renormalize

```python
        # Renormalize top-k probabilities to sum to 1
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        return top_k_probs, top_k_indices
```

**Why necessary:**
```
Before renormalization:
  [0.32, 0.28] → sum = 0.60
  "These experts got 60% of total probability"
  
After renormalization:
  [0.533, 0.467] → sum = 1.0
  "Expert 1 gets 53.3% weight, Expert 5 gets 46.7% weight"
  
Cleaner for weighted combination!
```

## Complete Router Example

```python
# Create router
router = Router(d_model=512, num_experts=8)

# Input: 2 sequences, 10 tokens each
x = torch.randn(2, 10, 512)

# Route with top-2
probs, indices = router(x, top_k=2)

print(f"Probabilities shape: {probs.shape}")  # torch.Size([2, 10, 2])
print(f"Indices shape: {indices.shape}")      # torch.Size([2, 10, 2])

# Examine first token's routing
print(f"\\nFirst token:")
print(f"  Selected experts: {indices[0, 0]}")  # tensor([3, 7])
print(f"  Weights: {probs[0, 0]}")              # tensor([0.62, 0.38])
```

**Interpretation:**
```
Token [0, 0] routes to:
  - Expert 3 with weight 0.62 (62%)
  - Expert 7 with weight 0.38 (38%)
```

## Routing Strategies

### 1. Top-1 (Switch Transformer)

Only route to single best expert:

```python
top_k_probs, top_k_indices = torch.topk(router_probs, k=1, dim=-1)
```

**Advantages:**
- Maximum sparsity (1 expert per token)
- Fastest inference

**Disadvantages:**
- Less robust
- Higher variance

### 2. Top-2 (Mixtral, DeepSeek)

Route to two best experts:

```python
top_k_probs, top_k_indices = torch.topk(router_probs, k=2, dim=-1)
```

**Advantages:**
- More robust (backup expert)
- Better quality
- Still very sparse

**Disadvantages:**
- 2× compute vs top-1

### 3. Expert Choice (Recent Research)

Instead of token choosing experts, **experts choose tokens**!

```python
# Each expert selects top-C tokens to process
for expert in experts:
    token_scores = router_logits[:, :, expert_idx]
    top_tokens = torch.topk(token_scores, k=capacity)
    # Process these tokens
```

**Advantages:**
- Natural load balancing
- Better hardware utilization

## Router Training Dynamics

### What the Router Learns

During training, the router learns to map token features to expert specializations:

```
Math token features → High score for math experts
Code token features → High score for code experts
Language token features → High score for language experts
```

**This happens implicitly through gradient descent!**

### Training Challenges

**1. Router collapse:**
```
Problem: Router sends all tokens to same expert
  Expert 0: 90% of tokens
  Experts 1-7: 10% of tokens total
  
Solution: Load balancing loss
```

**2. Router overfitting:**
```
Problem: Router learns shortcuts
  Token position → Expert (not token content)
  
Solution: Dropout on router, noise injection
```

**3. Expert underutilization:**
```
Problem: Some experts rarely selected
  Expert 6: 0.1% of tokens
  → Doesn't learn effectively
  
Solution: Expert capacity limits, balancing
```

### Load Balancing Loss

```python
def router_z_loss(router_logits):
    """Encourages router to not produce very large logits"""
    num_experts = router_logits.size(-1)
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z.pow(2).mean()
    return z_loss

def load_balance_loss(router_probs, selected_experts):
    """Encourages uniform expert usage"""
    num_experts = router_probs.size(-1)
    
    # Fraction of probability mass per expert
    f = router_probs.mean(dim=[0, 1])  # (num_experts,)
    
    # Fraction of tokens per expert  
    P = selected_experts.float().mean(dim=[0, 1])  # (num_experts,)
    
    # Both should be ~1/num_experts
    loss = num_experts * (f * P).sum()
    return loss
```

**Total training loss:**
```
L_total = L_task + α·L_balance + β·L_z

Where:
  L_task: Main task loss (cross-entropy)
  L_balance: Load balancing loss
  L_z: Router z-loss (logit regularization)
  α, β: Hyperparameters (typically 0.01-0.1)
```

## Router Variants

### 1. Learned Router (Standard)

```python
router = nn.Linear(d_model, num_experts)
```

### 2. Hash Router (Deterministic)

```python
def hash_router(x, num_experts):
    # Hash token representation to expert
    hash_val = hash(x.argmax().item())  # Simplified
    expert_id = hash_val % num_experts
    return expert_id
```

**No learned parameters, but no adaptation!**

### 3. Random Router (Baseline)

```python
def random_router(x, num_experts, top_k):
    batch, seq = x.shape[:2]
    indices = torch.randint(0, num_experts, (batch, seq, top_k))
    probs = torch.ones_like(indices).float() / top_k
    return probs, indices
```

**Useful for ablation studies.**

## Key Takeaways

✓ **Router = Linear layer:** Simple but powerful

✓ **Top-K selection:** Sparse routing (usually K=1 or 2)

✓ **Learned mapping:** Token features → Expert selection

✓ **Training challenges:** Load balancing critical

✓ **Tiny overhead:** Router has minimal parameters

**Remember:** The router is the brain of MoE - it makes all routing decisions! 🎉

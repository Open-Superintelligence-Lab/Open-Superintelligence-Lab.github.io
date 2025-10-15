---
hero:
  title: "Combining Experts"
  subtitle: "Weighted Combination of Expert Outputs"
  tags:
    - "🔀 MoE"
    - "⏱️ 8 min read"
---

After routing, we combine expert outputs using router weights!

## The Combination Problem

After the router selects top-K experts and the experts process their tokens, we need to **merge** the outputs back into a single representation.

### What We Have

```yaml
Input: Single token x
Router: Selected experts [2, 5] with weights [0.6, 0.4]
Expert 2: Produced output y₂
Expert 5: Produced output y₅

Question: How do we combine y₂ and y₅?
```

**Answer:** Weighted average based on router weights!

## Mathematical Formulation

### The Combination Formula

```
Final output: y = Σᵢ∈TopK(x) wᵢ · yᵢ

Where:
  wᵢ = Router weight for expert i
  yᵢ = Expert i's output
  TopK(x) = Set of selected experts
```

**In our example:**
```
y = 0.6 · y₂ + 0.4 · y₅
```

### Element-wise Combination

```
If expert outputs are vectors:
  y₂ = [1.2, 0.8, -0.5, ..., 0.3]  (d_model dimensions)
  y₅ = [0.5, 1.1, 0.2, ..., -0.1]
  
Weighted combination (element-wise):
  y[0] = 0.6 × 1.2 + 0.4 × 0.5 = 0.72 + 0.20 = 0.92
  y[1] = 0.6 × 0.8 + 0.4 × 1.1 = 0.48 + 0.44 = 0.92
  y[2] = 0.6 × (-0.5) + 0.4 × 0.2 = -0.30 + 0.08 = -0.22
  ...
  y[d-1] = 0.6 × 0.3 + 0.4 × (-0.1) = 0.18 - 0.04 = 0.14
```

**Result:** Combined vector of same dimension!

### Matrix Form

For batch processing:

```
Y = W ⊙ E

Where:
  Y ∈ ℝ^(B×L×D) = Final outputs
  W ∈ ℝ^(B×L×K) = Router weights (top-K per token)
  E ∈ ℝ^(B×L×K×D) = Expert outputs (K outputs per token)
  ⊙ = Weighted sum over K dimension

Explicitly:
  Y[b,l,:] = Σₖ W[b,l,k] · E[b,l,k,:]
```

## Implementation Approaches

### Approach 1: Loop Over Experts

```python
import torch

def combine_experts_v1(x, expert_outputs, router_weights, router_indices):
    """
    Args:
        x: (batch, seq, d_model) - input tokens
        expert_outputs: List of (n_tokens_i, d_model) - output per expert
        router_weights: (batch, seq, top_k) - routing weights
        router_indices: (batch, seq, top_k) - selected expert indices
    """
    batch, seq_len, d_model = x.shape
    top_k = router_weights.size(-1)
    
    # Initialize output
    final_output = torch.zeros_like(x)
    
    # For each position
    for b in range(batch):
        for s in range(seq_len):
            # Get this token's routing
            indices = router_indices[b, s]  # (top_k,)
            weights = router_weights[b, s]  # (top_k,)
            
            # Combine outputs from selected experts
            for k in range(top_k):
                expert_id = indices[k].item()
                weight = weights[k]
                expert_out = expert_outputs[expert_id][b, s]  # (d_model,)
                
                final_output[b, s] += weight * expert_out
    
    return final_output
```

**Simple but slow!** Triple nested loop.

### Approach 2: Vectorized with Masking

```python
def combine_experts_v2(x, experts, router_weights, router_indices):
    """
    More efficient vectorized approach
    
    Args:
        x: (batch, seq, d_model)
        experts: nn.ModuleList of expert networks
        router_weights: (batch, seq, top_k)
        router_indices: (batch, seq, top_k)
    """
    batch, seq_len, d_model = x.shape
    num_experts = len(experts)
    top_k = router_weights.size(-1)
    
    # Initialize output
    final_output = torch.zeros_like(x)
    
    # Process each expert
    for expert_idx in range(num_experts):
        # Create mask: which tokens use this expert?
        expert_mask = (router_indices == expert_idx)  # (batch, seq, top_k)
```

**Explanation:**
```
Example: expert_idx = 2

router_indices:
  Token 0: [2, 5]   → expert_mask[0] = [True, False]
  Token 1: [1, 3]   → expert_mask[1] = [False, False]
  Token 2: [2, 7]   → expert_mask[2] = [True, False]
  Token 3: [2, 5]   → expert_mask[3] = [True, False]

Tokens 0, 2, 3 use expert 2!
```

```python
        # Check if any token uses this expert
        if expert_mask.any():
            # Get tokens for this expert
            token_mask = expert_mask.any(dim=-1)  # (batch, seq)
            expert_input = x[token_mask]  # (n_tokens, d_model)
            
            # Run expert
            expert_output = experts[expert_idx](expert_input)
            
            # Add weighted output back
            for k in range(top_k):
                # Tokens where expert appears at position k
                k_mask = expert_mask[:, :, k]
                if k_mask.any():
                    weights = router_weights[:, :, k][k_mask]  # (n,)
                    final_output[k_mask] += weights.unsqueeze(-1) * expert_output
    
    return final_output
```

### Approach 3: Einsum (Most Elegant)

```python
def combine_experts_v3(expert_outputs, router_weights):
    """
    Super clean using Einstein summation
    
    Args:
        expert_outputs: (batch, seq, top_k, d_model) - stacked expert outputs
        router_weights: (batch, seq, top_k) - routing weights
    Returns:
        combined: (batch, seq, d_model)
    """
    # Weighted sum over top_k dimension
    combined = torch.einsum('bskd,bsk->bsd', expert_outputs, router_weights)
    return combined
```

**Einsum notation:**
```
'bskd,bsk->bsd'

Left tensor (expert_outputs):
  b = batch
  s = sequence position
  k = top-k expert
  d = d_model

Right tensor (router_weights):
  b = batch
  s = sequence position
  k = top-k expert (matched!)

Output:
  b = batch
  s = sequence
  d = d_model
  
Operation: Sum over k dimension (weighted)
```

## Complete Example Walkthrough

Let's trace through combining 2 experts for 1 token:

### Setup

```python
# Token representation
token = torch.tensor([0.1, 0.5, -0.3, 0.8])  # d_model=4

# Router selected experts 1 and 3
selected_experts = [1, 3]
weights = torch.tensor([0.7, 0.3])

# Expert 1 output
expert_1_out = torch.tensor([1.0, 0.5, 0.2, 0.9])

# Expert 3 output
expert_3_out = torch.tensor([0.3, 1.2, -0.1, 0.4])
```

### Step-by-Step Combination

```python
# Dimension 0
combined[0] = 0.7 * 1.0 + 0.3 * 0.3 = 0.7 + 0.09 = 0.79

# Dimension 1
combined[1] = 0.7 * 0.5 + 0.3 * 1.2 = 0.35 + 0.36 = 0.71

# Dimension 2
combined[2] = 0.7 * 0.2 + 0.3 * (-0.1) = 0.14 - 0.03 = 0.11

# Dimension 3
combined[3] = 0.7 * 0.9 + 0.3 * 0.4 = 0.63 + 0.12 = 0.75

# Result
combined = [0.79, 0.71, 0.11, 0.75]
```

### In Code

```python
combined = torch.zeros(4)
for i, (expert_out, weight) in enumerate(zip([expert_1_out, expert_3_out], weights)):
    combined += weight * expert_out

print(combined)
# tensor([0.7900, 0.7100, 0.1100, 0.7500])
```

## Handling Different Top-K Values

### Top-1 (Single Expert)

```python
# No combination needed!
final_output = expert_outputs[selected_expert_idx]
# Or with weight (always 1.0):
final_output = 1.0 * expert_outputs[selected_expert_idx]
```

### Top-2 (Most Common)

```python
final_output = w₁ * expert₁_out + w₂ * expert₂_out
```

### Top-K (General)

```python
final_output = Σₖ wₖ · expertₖ_out
```

## Sparse vs Dense Combination

### Sparse (MoE - What We Want)

```python
# Only K experts contribute (K << N)
num_experts = 64
top_k = 2

# Combination involves only 2 experts
output = 0.6 * expert_2_output + 0.4 * expert_5_output
# Experts 0, 1, 3, 4, 6-63: Not computed!
```

**Compute:** O(K) where K=2

### Dense (Hypothetical - Inefficient)

```python
# All experts contribute
output = Σᵢ₌₀⁶³ wᵢ · expertᵢ_output

# Must run all 64 experts!
```

**Compute:** O(N) where N=64

**MoE wins:** 2 vs 64 expert evaluations!

## Numerical Stability Considerations

### Issue: Weight Normalization

```python
# Before normalization
weights = [0.32, 0.28]  # Sum = 0.60

# After normalization
weights = [0.533, 0.467]  # Sum = 1.00
```

**Why normalize?**
- Ensures output scale is consistent
- Prevents magnitude drift during training
- Easier to reason about

### Issue: Gradient Flow

```python
# Both expert output AND router weights get gradients!

∂L/∂expertᵢ_output = w_i · ∂L/∂combined_output
∂L/∂wᵢ = expertᵢ_output · ∂L/∂combined_output

# Router learns which experts to select
# Experts learn how to process their assigned tokens
```

## Complete MoE Forward with Combination

```python
def moe_forward(x, experts, router):
    """Complete MoE forward pass"""
    batch, seq, d_model = x.shape
    
    # Step 1: Route
    router_weights, router_indices = router(x)  # (B,S,K), (B,S,K)
    
    # Step 2: Collect expert outputs
    expert_outputs = []
    for expert_idx in range(len(experts)):
        mask = (router_indices == expert_idx).any(dim=-1)
        if mask.any():
            expert_out = experts[expert_idx](x[mask])
            expert_outputs.append(expert_out)
        else:
            expert_outputs.append(None)
    
    # Step 3: Combine
    final_output = torch.zeros_like(x)
    for b in range(batch):
        for s in range(seq):
            for k in range(router_weights.size(-1)):
                expert_idx = router_indices[b, s, k].item()
                weight = router_weights[b, s, k]
                if expert_outputs[expert_idx] is not None:
                    final_output[b, s] += weight * expert_outputs[expert_idx][...]
    
    return final_output
```

## Key Takeaways

✓ **Weighted sum:** Combine expert outputs using router weights

✓ **Element-wise:** Each dimension combined independently

✓ **Sparse:** Only K experts contribute (K << N total experts)

✓ **Normalized weights:** Ensure weights sum to 1.0

✓ **Gradient flow:** Both router and experts learn together

**Remember:** Combining is just weighted averaging - simple but powerful! 🎉

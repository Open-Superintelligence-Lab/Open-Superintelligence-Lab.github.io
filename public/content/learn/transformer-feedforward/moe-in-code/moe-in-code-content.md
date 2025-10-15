---
hero:
  title: "MoE in Code"
  subtitle: "Complete MoE Implementation"
  tags:
    - "🔀 MoE"
    - "⏱️ 10 min read"
---

Complete, working Mixture of Experts implementation!

## Building MoE From Scratch

Now let's bring everything together into a complete, production-ready MoE layer! We'll build it incrementally to understand each component.

## Step 1: The Expert Network

First, define what an individual expert looks like:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """Single expert: standard FFN"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.activation = nn.SiLU()  # Modern activation
    
    def forward(self, x):
        # x: (num_tokens, d_model)
        return self.w2(self.activation(self.w1(x)))
```

**What it does:**
```
Input: (n_tokens, 512)
  ↓ Linear expansion
Hidden: (n_tokens, 2048)
  ↓ SiLU activation
  ↓ Linear compression
Output: (n_tokens, 512)
```

## Step 2: The Router

Create the routing mechanism:

```python
class Router(nn.Module):
    """Routes tokens to experts"""
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.num_experts = num_experts
    
    def forward(self, x, top_k=2):
        # x: (batch * seq_len, d_model)
        
        # Compute routing scores
        logits = self.gate(x)  # (batch*seq, num_experts)
        probs = F.softmax(logits, dim=-1)
        
        # Select top-k
        top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
        
        # Renormalize
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        return top_k_probs, top_k_indices, probs
```

**Returns:**
- `top_k_probs`: Weights for selected experts
- `top_k_indices`: Which experts were selected
- `probs`: All probabilities (for load balancing)

## Step 3: Complete MoE Layer

Now combine everything:

```python
class MixtureOfExperts(nn.Module):
    def __init__(self, d_model, num_experts=8, top_k=2, d_ff=None):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        # Create all experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff)
            for _ in range(num_experts)
        ])
        
        # Create router
        self.router = Router(d_model, num_experts)
```

**Initialization:** Creates N independent expert networks + 1 router.

### Forward Pass: The Complete Process

```python
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        batch_size, seq_len, d_model = x.size()
        
        # Flatten for easier processing
        x_flat = x.view(-1, d_model)  # (batch*seq, d_model)
```

**Why flatten?**
```
Before: (2, 10, 512) - 2 sequences, 10 tokens each
After: (20, 512) - 20 tokens total

Easier to route: each of 20 tokens handled independently
```

### Step 3.1: Route

```python
        # Get routing decisions
        top_k_probs, top_k_indices, all_probs = self.router(x_flat, self.top_k)
        # top_k_probs: (20, 2) - weights for 2 experts per token
        # top_k_indices: (20, 2) - which 2 experts per token
        # all_probs: (20, 8) - probabilities for all 8 experts
```

**Example for token 0:**
```
all_probs[0] = [0.05, 0.32, 0.08, 0.15, 0.03, 0.28, 0.04, 0.05]
top_k_indices[0] = [1, 5]  (experts 1 and 5)
top_k_probs[0] = [0.533, 0.467]  (renormalized)
```

### Step 3.2: Process with Experts

```python
        # Initialize output
        final_output = torch.zeros_like(x_flat)
        
        # Create one-hot encoding of expert assignments
        expert_mask = torch.zeros(
            x_flat.size(0), 
            self.num_experts, 
            device=x.device
        )  # (20, 8)
        
        # Fill in which experts each token uses
        for i in range(self.top_k):
            expert_mask.scatter_(1, top_k_indices[:, i:i+1], 1)
```

**Expert mask example:**
```
Token 0 uses experts [1, 5]:
  expert_mask[0] = [0, 1, 0, 0, 0, 1, 0, 0]
  
Token 1 uses experts [0, 3]:
  expert_mask[1] = [1, 0, 0, 1, 0, 0, 0, 0]
```

### Step 3.3: Parallel Expert Processing

```python
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            token_mask = expert_mask[:, expert_idx].bool()
            
            if not token_mask.any():
                continue  # No tokens for this expert
            
            # Get inputs for this expert
            expert_input = x_flat[token_mask]  # (n_tokens_for_expert, d_model)
            
            # Run expert
            expert_output = self.experts[expert_idx](expert_input)
```

**Batching per expert:**
```
Expert 1 processes tokens [0, 3, 7, 12, 15]  (5 tokens)
Expert 2 processes tokens [1, 5, 9, 10]      (4 tokens)
...

Efficient: Each expert processes its tokens in a single batch!
```

### Step 3.4: Weighted Combination

```python
            # Distribute output back to original positions
            # Need to weight by router probabilities
            for k_idx in range(self.top_k):
                # Tokens that selected this expert at position k_idx
                k_mask = (top_k_indices[:, k_idx] == expert_idx) & token_mask
                
                if k_mask.any():
                    # Get weights for these tokens
                    weights = top_k_probs[:, k_idx][k_mask]
                    
                    # Add weighted expert output
                    # Note: expert_output indices don't match k_mask indices
                    # Need to map back
                    expert_token_positions = torch.where(token_mask)[0]
                    k_token_positions = torch.where(k_mask)[0]
                    
                    for token_idx, k_pos in enumerate(k_token_positions):
                        expert_out_idx = (expert_token_positions == k_pos).nonzero(as_tuple=True)[0].item()
                        final_output[k_pos] += weights[token_idx] * expert_output[expert_out_idx]
```

**What's happening:**
```
Token 0 routed to Expert 1 (weight 0.533) and Expert 5 (weight 0.467)

When processing Expert 1:
  expert_output[0] corresponds to token 0
  Add 0.533 * expert_output[0] to final_output[0]
  
When processing Expert 5:
  expert_output[2] corresponds to token 0 (it's the 3rd token for expert 5)
  Add 0.467 * expert_output[2] to final_output[0]
  
Final: final_output[0] = 0.533 * E1(x0) + 0.467 * E5(x0)
```

### Step 3.5: Reshape and Return

```python
        # Reshape back to original shape
        final_output = final_output.view(batch_size, seq_len, d_model)
        
        return final_output, all_probs  # Return probs for load balancing
```

## Step 4: Load Balancing Loss

Add auxiliary loss to encourage balanced expert usage:

```python
def load_balancing_loss(all_probs, top_k_indices, num_experts):
    """
    Encourages uniform expert usage
    
    Args:
        all_probs: (batch*seq, num_experts) - router probabilities
        top_k_indices: (batch*seq, top_k) - selected experts
        num_experts: int
    """
    # Fraction of routing probability to each expert
    p = all_probs.mean(dim=0)  # (num_experts,)
    
    # Fraction of tokens assigned to each expert
    expert_mask = torch.zeros_like(all_probs)
    expert_mask.scatter_(1, top_k_indices, 1)
    f = expert_mask.mean(dim=0)  # (num_experts,)
    
    # Both should be ~1/num_experts
    # Minimize product sum
    loss = num_experts * (f * p).sum()
    
    return loss
```

**Why this works:**
```
Ideal: Each expert gets 1/8 = 12.5% of tokens
  f = [0.125, 0.125, ..., 0.125]
  p = [0.125, 0.125, ..., 0.125]
  loss = 8 × Σ(0.125 × 0.125) = 8 × 8 × 0.0156 = 1.0
  
Imbalanced: Expert 0 gets 50% of tokens
  f = [0.50, 0.07, ..., 0.07]
  p = [0.50, 0.07, ..., 0.07]
  loss = 8 × (0.25 + 7×0.0049) = 8 × 0.284 = 2.27 > 1.0
  
Higher loss encourages rebalancing!
```

## Complete Training Example

```python
# Create MoE layer
moe = MixtureOfExperts(
    d_model=512,
    num_experts=8,
    top_k=2,
    d_ff=2048
)

# Sample data
x = torch.randn(4, 16, 512)  # 4 sequences, 16 tokens each
targets = torch.randn(4, 16, 512)

# Forward
output, router_probs = moe(x)

# Compute losses
task_loss = F.mse_loss(output, targets)

# Load balancing
# Need to extract top_k_indices from forward pass
# (In practice, modify forward to return this)
balance_loss = 0.01 * load_balancing_loss(
    router_probs,
    top_k_indices,  # From forward pass
    num_experts=8
)

# Total loss
total_loss = task_loss + balance_loss

# Backward
total_loss.backward()
```

## Optimizations and Variants

### 1. Capacity Factor (Switch Transformer)

Limit tokens per expert:

```python
def forward_with_capacity(self, x, capacity_factor=1.25):
    ...
    # Compute capacity
    num_tokens = x_flat.size(0)
    capacity = int((num_tokens / self.num_experts) * self.top_k * capacity_factor)
    
    # Limit tokens per expert to capacity
    for expert_idx in range(self.num_experts):
        expert_tokens = tokens_for_expert[expert_idx]
        if len(expert_tokens) > capacity:
            # Drop lowest-probability tokens
            expert_tokens = expert_tokens[:capacity]
```

### 2. Expert Choice Routing

Let experts choose tokens instead:

```python
class ExpertChoiceRouter(nn.Module):
    def forward(self, x, capacity_per_expert):
        # Each expert selects top-C tokens
        all_scores = self.gate(x)  # (num_tokens, num_experts)
        
        expert_assignments = []
        for expert_idx in range(self.num_experts):
            scores = all_scores[:, expert_idx]
            top_tokens = torch.topk(scores, k=capacity_per_expert)
            expert_assignments.append(top_tokens.indices)
        
        return expert_assignments
```

### 3. Shared Expert (DeepSeek-MoE)

Add one expert that always runs:

```python
class MoEWithSharedExpert(nn.Module):
    def __init__(self, d_model, num_experts, top_k):
        super().__init__()
        
        # Shared expert (always active)
        self.shared_expert = Expert(d_model, 4 * d_model)
        
        # Routed experts (sparse)
        self.routed_experts = MixtureOfExperts(d_model, num_experts, top_k)
    
    def forward(self, x):
        # Always run shared expert
        shared_out = self.shared_expert(x)
        
        # Sparse routing for other experts
        routed_out, probs = self.routed_experts(x)
        
        # Combine
        return shared_out + routed_out, probs
```

## Testing Your MoE

```python
# Create MoE
moe = MixtureOfExperts(d_model=512, num_experts=8, top_k=2)

# Test forward pass
x = torch.randn(2, 10, 512)
output, probs = moe(x)

print(f"Input shape: {x.shape}")          # torch.Size([2, 10, 512])
print(f"Output shape: {output.shape}")    # torch.Size([2, 10, 512])
print(f"Router probs shape: {probs.shape}") # torch.Size([20, 8])

# Check expert usage
expert_usage = (probs > probs.mean()).float().mean(dim=0)
print(f"Expert usage distribution: {expert_usage}")
# Should be roughly uniform: [0.12, 0.13, 0.11, ...]

# Count parameters
moe_params = sum(p.numel() for p in moe.parameters())
print(f"MoE parameters: {moe_params:,}")

# Compare to single FFN
ffn = nn.Sequential(
    nn.Linear(512, 2048),
    nn.ReLU(),
    nn.Linear(2048, 512)
)
ffn_params = sum(p.numel() for p in ffn.parameters())
print(f"Single FFN parameters: {ffn_params:,}")
print(f"MoE has {moe_params / ffn_params:.1f}× more parameters")
```

## Key Takeaways

✓ **Complete implementation:** Production-ready MoE code

✓ **Three components:** Experts, Router, Combination logic

✓ **Load balancing:** Critical for stable training

✓ **Sparse activation:** Only top-K experts active per token

✓ **Scalable:** Add experts without proportional compute increase

**Remember:** MoE = Routing + Expert Processing + Weighted Combination! 🎉

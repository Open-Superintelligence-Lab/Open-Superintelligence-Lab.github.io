---
hero:
  title: "What is Mixture of Experts"
  subtitle: "Sparse Expert Models Explained"
  tags:
    - "🔀 MoE"
    - "⏱️ 10 min read"
---

Mixture of Experts (MoE) uses **multiple specialized sub-networks (experts)** and routes inputs to the most relevant ones!

![MoE Routing](/content/learn/transformer-feedforward/what-is-mixture-of-experts/moe-routing.png)

## The Revolutionary Idea

Traditional neural networks use the **same** parameters for all inputs. MoE breaks this assumption!

### The Problem with Dense Networks

```yaml
Standard Transformer FFN:
  Input 1 (about math) → FFN → Output
  Input 2 (about history) → Same FFN → Output
  Input 3 (about code) → Same FFN → Output
  
Problem: Same network must handle ALL types of input!
  → Network becomes a "jack of all trades, master of none"
```

**The bottleneck:** One network can't be equally good at everything.

### The MoE Solution: Specialization

```yaml
MoE Transformer:
  Input 1 (math) → Expert 3 (math specialist) + Expert 7 → Output
  Input 2 (history) → Expert 1 (language) + Expert 5 → Output
  Input 3 (code) → Expert 2 (code specialist) + Expert 4 → Output
  
Each input uses DIFFERENT experts!
  → Specialization leads to better performance
```

**Key insight:** Different experts for different patterns!

## The Three Core Components

### 1. Multiple Expert Networks

Instead of one feedforward network, we have **N** specialized networks:

```
Expert 1: Specializes in pattern A
Expert 2: Specializes in pattern B
Expert 3: Specializes in pattern C
...
Expert N: Specializes in pattern Z
```

**Typical numbers:**
- Switch Transformer: 128-2048 experts
- DeepSeek-MoE: 64 experts
- Mixtral: 8 experts

### 2. Router (Gate) Network

A learned function that decides **which experts to use** for each input:

```
Router(token) → [expert_scores]
             → Select top-K experts
```

**Example:**
```
Token: "The derivative of x² is"
Router scores: [0.05, 0.85, 0.02, 0.03, 0.78, 0.01, 0.04, 0.02]
Top-2: Expert 1 (0.85), Expert 4 (0.78)
→ Route to math-specialized experts!
```

### 3. Sparse Activation

**Only activate top-K experts per input**, not all N!

```
If N=64 experts and K=2:
  Activated: 2 experts (3.1% of network)
  Inactive: 62 experts (96.9% of network)
  
Massive compute savings!
```

## Mathematical Formulation

### Dense FFN (Standard)

```
y = FFN(x)
  = W₂ · ReLU(W₁ · x)
  
All tokens use same W₁, W₂
Parameters: Always active
```

### Sparse MoE

```
y = Σᵢ₌₁ᴺ G(x)ᵢ · Eᵢ(x)

Where:
  G(x) = Router/Gate function (learned)
  G(x)ᵢ = Weight for expert i
  Eᵢ(x) = Expert i's output
  N = Total number of experts
```

**With top-K sparsity:**
```
y = Σᵢ∈TopK(G(x)) G(x)ᵢ · Eᵢ(x)

Only sum over top-K experts!
Most G(x)ᵢ = 0 (not used)
```

**Detailed breakdown:**
```
Step 1: Compute router logits
  logits = Wᵣ · x    (linear layer)
  
Step 2: Get probabilities
  G(x) = softmax(logits)
  G(x) = [g₁, g₂, ..., gₙ]
  
Step 3: Select top-K
  TopK = indices of K largest values in G(x)
  
Step 4: Compute expert outputs
  For i in TopK:
    yᵢ = Eᵢ(x) = Expert_i(x)
    
Step 5: Weighted combination
  y = Σᵢ∈TopK gᵢ · yᵢ
```

## Simple Example with Walkthrough

Let's build a minimal MoE and trace through it:

### Step 1: Create the Experts

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMoE(nn.Module):
    def __init__(self, d_model, num_experts=8):
        super().__init__()
        
        # Create N expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),  # Expand
                nn.ReLU(),
                nn.Linear(d_model * 4, d_model)    # Compress
            )
            for _ in range(num_experts)
        ])
```

**What we have:** 8 identical structures, different learned parameters.

### Step 2: Create the Router

```python
        # Router: decides which experts to use
        self.router = nn.Linear(d_model, num_experts)
        self.num_experts = num_experts
```

**Router output:** Score for each expert (higher = more relevant).

### Step 3: Forward Pass - Routing

```python
    def forward(self, x, top_k=2):
        # x: (batch, seq, d_model)
        # Example: (2, 10, 512) - 2 sequences, 10 tokens each
        
        # Compute router scores for each token
        router_logits = self.router(x)
        # Shape: (2, 10, 8) - 8 scores per token
```

**What happens:**
```
Token at position [0, 5]:
  Input: 512-dim vector
  Router: Linear(512 → 8)
  Output: [2.1, -0.5, 1.8, 0.2, -1.0, 3.2, 0.8, -0.3]
  → Raw scores for 8 experts
```

### Step 4: Get Probabilities

```python
        # Convert to probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        # Shape: (2, 10, 8)
        # Each token's 8 scores sum to 1.0
```

**Example:**
```
Raw logits: [2.1, -0.5, 1.8, 0.2, -1.0, 3.2, 0.8, -0.3]
After softmax: [0.19, 0.01, 0.14, 0.03, 0.01, 0.56, 0.05, 0.02]
Sum = 1.0 ✓

Expert 5 has highest probability (0.56)!
```

### Step 5: Select Top-K Experts

```python
        # Select top-2 experts per token
        top_k_probs, top_k_indices = torch.topk(router_probs, k=top_k, dim=-1)
        # top_k_probs: (2, 10, 2) - weights for top-2
        # top_k_indices: (2, 10, 2) - which 2 experts
```

**Example for one token:**
```
All probs: [0.19, 0.01, 0.14, 0.03, 0.01, 0.56, 0.05, 0.02]
Top-2 indices: [5, 0]  (expert 5 and expert 0)
Top-2 probs: [0.56, 0.19]
Renormalize: [0.75, 0.25]  (sum to 1.0 for numerical stability)
```

### Step 6: Route to Experts and Combine

```python
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # For each expert
        for expert_idx in range(self.num_experts):
            # Find tokens that selected this expert
            mask = (top_k_indices == expert_idx).any(dim=-1)
```

**Masking:**
```
Expert 5 selected by tokens: [0, 2, 5, 7, 9, 10, 15, 18]
mask[0] = True (token 0 uses expert 5)
mask[1] = False (token 1 doesn't use expert 5)
...
```

```python
            if mask.any():
                # Get tokens for this expert
                expert_input = x[mask]
                
                # Run expert
                expert_output = self.experts[expert_idx](expert_input)
                
                # Get weights for these tokens
                for k_idx in range(top_k):
                    k_mask = (top_k_indices[:, :, k_idx] == expert_idx)
                    if k_mask.any():
                        weight = top_k_probs[:, :, k_idx][k_mask]
                        output[k_mask] += weight.unsqueeze(-1) * expert_output[k_mask.flatten()]
        
        return output
```

**Combination example:**
```
Token 5:
  Routed to Expert 5 (weight 0.75) and Expert 0 (weight 0.25)
  
  expert_5_output = Expert_5(token_5) = [1.2, 0.8, ..., 0.5]
  expert_0_output = Expert_0(token_5) = [0.5, 1.1, ..., 0.3]
  
  final = 0.75 × expert_5_output + 0.25 × expert_0_output
        = [1.025, 0.875, ..., 0.45]
```

### Testing the MoE

```python
# Create MoE layer
moe = SimpleMoE(d_model=512, num_experts=8)

# Input: 2 sequences of 10 tokens
x = torch.randn(2, 10, 512)

# Forward pass
output = moe(x, top_k=2)

print(f"Input shape: {x.shape}")      # torch.Size([2, 10, 512])
print(f"Output shape: {output.shape}") # torch.Size([2, 10, 512])

# Each token potentially used different experts!
```

## Why MoE? The Scaling Breakthrough

### Benefits: Capacity Without Compute Cost

**1. Massive Parameter Count**
```
Standard Transformer (GPT-3 size):
  175B parameters
  All active for every token
  
MoE Transformer (Switch Transformer):
  1.6 TRILLION parameters
  Only ~10B active per token (< 1%)
  
100× more parameters, similar compute!
```

**2. Conditional Computation**
```
Dense model: y = f_all(x)
  All parameters used for all inputs
  
Sparse MoE: y = Σᵢ∈TopK(x) gᵢ · fᵢ(x)
  Different parameters for different inputs
  Model capacity scales with # experts!
```

**3. Automatic Specialization**
```
During training, experts naturally specialize:
  Expert 1: Mathematical reasoning
  Expert 2: Common sense knowledge
  Expert 3: Code generation
  Expert 4: Creative writing
  Expert 5: Factual recall
  ...
  
This emerges automatically from routing!
```

### Trade-offs and Challenges

**1. Load Balancing Problem**
```
What we want:
  Each expert processes ~equal tokens
  [Expert1: 1000 tokens, Expert2: 1000 tokens, ...]
  
What can happen:
  Expert 1: 5000 tokens (overloaded)
  Expert 2: 10 tokens (underutilized)
  Expert 3: 5 tokens
  ...
  
Solution: Load balancing loss
```

**2. Training Instability**
```
Router can collapse:
  → All tokens to same expert
  → Other experts never train
  → Specialization lost
  
Need careful regularization!
```

**3. Memory Requirements**
```
Must store all expert parameters:
  8 experts × 2GB each = 16GB
  Even if only 2 active at once
  
Memory scales with # experts, not compute!
```

## Real-World Impact

### Models Using MoE

**Switch Transformer (Google, 2021)**
```yaml
Architecture:
  - 1.6T parameters
  - 2048 experts per layer
  - Top-1 routing (1 expert per token)
  
Performance:
  - 4× faster training than T5
  - Better quality at lower compute
```

**Mixtral 8x7B (Mistral AI, 2023)**
```yaml
Architecture:
  - 8 experts
  - Top-2 routing
  - 46.7B total, 12.9B active per token
  
Performance:
  - Matches/exceeds Llama 2 70B
  - 6× faster inference
```

**DeepSeek-MoE (2024)**
```yaml
Architecture:
  - 64 experts per layer
  - Shared expert + fine-grained experts
  - Top-6 routing
  
Innovation:
  - Fewer parameters than standard MoE
  - Better load balancing
  - State-of-the-art efficiency
```

**GPT-4 (OpenAI, rumored)**
```yaml
Speculation:
  - Likely uses MoE architecture
  - 8 experts, each ~220B parameters
  - ~1.8T total parameters
  
Why likely:
  - Cost-effective scaling
  - Explainable via routing
  - Industry trend
```

## Key Takeaways

✓ **Multiple experts:** Specialized sub-networks

✓ **Sparse routing:** Each token uses few experts

✓ **Scalable:** Add experts without much compute cost

**Remember:** MoE = specialized experts for different patterns! 🎉

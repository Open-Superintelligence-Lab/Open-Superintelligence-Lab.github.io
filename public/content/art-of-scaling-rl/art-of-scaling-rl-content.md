---
hero:
  title: "The Art of Scaling RL"
  subtitle: "🚀 Predictive Scaling Laws for Reinforcement Learning in LLMs"
  tags:
    - "⏱️ Research Deep Dive"
    - "📄 Research Article"
---

> **🚀 Key Insight:**  
> This groundbreaking research establishes the first systematic, large-scale study of RL scaling for LLMs, providing predictive performance curves and a stable "best-practice" recipe called **ScaleRL** validated across 100,000 GPU-hours.

## Abstract

While training compute for reinforcement learning (RL) for LLMs is massively increasing, the field has been lacking predictive scaling methodologies comparable to those established for pre-training. This gap is increasingly consequential given recent large-scale RL efforts for reasoning-centric post-training.

This paper presents the first open, large-compute, systematic study of RL scaling for LLMs, fitting sigmoidal compute-performance curves and ablating a wide range of common design choices. Based on 400,000 GPU-hours of total experiments, the research demonstrates successful scaling and prediction of RL training performance on up to 100,000 GPU-hours.

## The Scaling Gap in RL

### Why This Matters

The field of LLM pre-training has benefited from well-established **scaling laws**—predictable relationships between compute resources and model performance. These laws allow researchers to:

- **Predict performance** before running expensive experiments
- **Optimize resource allocation** efficiently
- **Make informed architectural decisions**

However, for **reinforcement learning post-training**, such predictive frameworks have been conspicuously absent. With companies investing massive compute budgets into RL for reasoning models (like OpenAI's o1, o3), this gap has become critical.

### The Challenge

RL post-training differs fundamentally from pre-training:

1. **Non-stationary objectives** - The reward model and policy evolve together
2. **Complex optimization landscape** - Multiple interacting components (policy, value function, reward model)
3. **Design space explosion** - Countless hyperparameter and architectural choices
4. **Sample efficiency concerns** - How to optimally use collected trajectories

Without predictive scaling laws, researchers have been essentially "flying blind," unable to anticipate whether their RL recipe will scale effectively before committing substantial resources.

## Key Research Findings

The study reveals three fundamental insights about RL scaling:

### 1. Not All Recipes Are Created Equal

> **Finding:** Different RL configurations yield different asymptotic performance levels—you can't simply "scale your way out" of a bad recipe.

Unlike pre-training where most reasonable configurations eventually converge to similar performance with enough compute, RL post-training exhibits **recipe-dependent asymptotes**. Some configurations will never reach the performance of others, regardless of compute invested.

**Implication:** Choosing the right base recipe is critical before scaling up.

### 2. Implementation Details Affect Efficiency, Not Asymptotes

The research systematically ablated:

- **Loss aggregation methods** (sum vs. mean)
- **Normalization techniques** (advantage normalization, gradient clipping)
- **Curriculum strategies** (difficulty progression)
- **Precision handling** (FP16, BF16, FP32, mixed precision)

> **Finding:** These details primarily affect **compute efficiency** (how fast you reach the asymptote) but don't materially change the final performance ceiling.

**Key Design Choices Tested:**

| Component | Variants Tested | Impact |
|-----------|----------------|---------|
| Loss Aggregation | Sum, Mean, Per-token | Efficiency only |
| Advantage Normalization | None, Batch, Global | Efficiency + Stability |
| Curriculum | None, Linear, Exponential | Efficiency only |
| Precision | FP16, BF16, FP32, Mixed | Efficiency + Stability |

### 3. Stable Recipes Enable Predictive Scaling

> **Finding:** Properly configured RL exhibits predictable sigmoidal scaling behavior, similar to pre-training.

For stable recipes, the relationship between compute and performance follows a **sigmoidal curve** that can be fitted and extrapolated, enabling:

- Performance prediction for unrun experiments
- Optimal stopping criteria
- Resource allocation optimization

## The ScaleRL Recipe

Based on 400,000 GPU-hours of experimentation, the researchers propose **ScaleRL**—a best-practice configuration for stable, predictable RL scaling.

### Core Components

#### 1. **Algorithm: Group Relative Policy Optimization (GRPO)**

ScaleRL builds on GRPO, a variant of PPO specifically designed for language models:

- **Relative advantages** - Compute advantage by comparing trajectories within the same prompt group
- **Reduced variance** - More stable learning compared to standard PPO
- **Better sample efficiency** - Effective use of collected rollouts

#### 2. **Loss Configuration**

```
Total Loss = Policy Loss + Value Loss + Entropy Regularization

Where:
- Policy Loss: Clipped surrogate objective with ε=0.2
- Value Loss: Clipped MSE with coefficient 0.5
- Entropy Coefficient: 0.01 (for exploration)
```

**Key Settings:**
- Advantage normalization: **Per-batch with running statistics**
- Gradient clipping: **Max norm = 1.0**
- Learning rate schedule: **Cosine decay with warmup**

#### 3. **Precision Strategy**

**Mixed-precision training with careful component allocation:**

| Component | Precision | Reason |
|-----------|-----------|--------|
| Policy Network | BF16 | Speed + stability |
| Value Network | FP32 | Numerical accuracy |
| Advantage Computation | FP32 | Prevent underflow |
| KL Divergence | FP32 | Precise constraint |

#### 4. **Curriculum Design**

**Progressive difficulty scaling:**

```
Early training: Short sequences, simple rewards
Mid training: Medium sequences, mixed rewards  
Late training: Full sequences, complex rewards
```

This allows the model to build foundational capabilities before tackling harder problems.

### Hyperparameter Recommendations

Based on extensive ablations across different model scales:

| Parameter | Small Models (1B-7B) | Large Models (7B-70B+ | Rationale |
|-----------|---------------------|----------------------|-----------|
| Learning Rate | 3e-6 to 1e-5 | 1e-6 to 3e-6 | Larger models need gentler updates |
| Batch Size | 256-512 samples | 512-2048 samples | More data stabilizes larger models |
| PPO Epochs | 4-8 | 2-4 | Larger models overfit faster |
| Warmup Steps | 100-500 | 500-2000 | Longer warmup for stable start |
| KL Coefficient | 0.05-0.1 | 0.01-0.05 | Tighter constraint for large models |

## The Experimental Methodology

### Scale of Study

- **Total compute:** 400,000 GPU-hours
- **Largest single run:** 100,000 GPU-hours
- **Model scales tested:** 1.5B, 7B, 13B, 34B, 70B parameters
- **Ablations performed:** 100+ different configurations

### Evaluation Framework

**Tasks Tested:**

1. **Mathematical reasoning** (GSM8K, MATH)
2. **Code generation** (HumanEval, MBPP)
3. **General reasoning** (ARC, HellaSwag)
4. **Long-form QA** (TruthfulQA, NaturalQuestions)

**Metrics Tracked:**

- Task performance (accuracy/pass rate)
- Training stability (gradient norms, KL divergence)
- Sample efficiency (performance per GPU-hour)
- Scaling predictability (R² of fitted curves)

### The Sigmoidal Scaling Curve

The research demonstrates that stable RL recipes follow a predictable sigmoidal pattern:

```
Performance = L + (U - L) / (1 + exp(-k(compute - c)))

Where:
L = Lower asymptote (initial performance)
U = Upper asymptote (maximum achievable performance)
k = Growth rate (learning efficiency)
c = Inflection point (compute at steepest growth)
```

**Key Validation:**

- **R² > 0.95** for ScaleRL across all model scales
- **Prediction error < 2%** when extrapolating to 2x compute
- **Robust across tasks** - similar curves for math, code, and reasoning

## Practical Implications

### For Researchers

1. **Start with ScaleRL** - Use the validated recipe as a baseline
2. **Fit early curves** - Run small-scale experiments (1,000-5,000 GPU-hours) and fit scaling curves
3. **Predict before scaling** - Use fitted curves to estimate full-scale performance
4. **Ablate efficiently** - Focus on recipe selection (asymptote) over implementation details (efficiency)

### For Practitioners

1. **Resource planning** - Use scaling curves to estimate compute budgets for target performance
2. **Early stopping** - Detect when you're approaching asymptote to avoid wasted compute
3. **Recipe selection** - Invest time in finding the right recipe before large-scale training
4. **Monitoring** - Track actual vs. predicted performance to detect issues early

### For the Field

1. **Reproducible baselines** - ScaleRL provides a strong, documented starting point
2. **Principled comparison** - Compare recipes by their predicted asymptotes, not just final runs
3. **Theoretical grounding** - Sigmoidal scaling suggests underlying principles worth investigating
4. **Transfer insights** - Recipes that scale well on smaller models likely scale to larger ones

## Surprising Discoveries

### 1. KL Divergence Sweet Spot

The study found a **narrow optimal range** for KL divergence constraints:

- **Too loose (β < 0.01):** Policy diverges, unstable training
- **Optimal (β = 0.01-0.1):** Smooth learning, stable scaling
- **Too tight (β > 0.2):** Slow learning, premature convergence

This suggests KL constraints are more critical than previously thought.

### 2. Value Function Precision Matters Enormously

Running value networks in FP16 vs. FP32 made a **15-20% difference** in final performance:

- **FP16 value network:** Instabilities emerge after ~50K GPU-hours
- **FP32 value network:** Stable throughout 100K+ GPU-hours

This was one of the most impactful single findings.

### 3. Curriculum Helps Efficiency, Not Asymptote

Surprisingly, curriculum strategies reached the **same final performance** but got there **30-40% faster** in compute:

- **No curriculum:** 100K GPU-hours to convergence
- **Progressive curriculum:** 60-70K GPU-hours to same performance

### 4. Recipe Transfer Across Scales

**Strong positive result:** Recipes that worked at 7B scale **transferred effectively** to 34B and 70B scales with only minor hyperparameter adjustments (primarily learning rate).

This suggests the field can develop recipes on smaller, cheaper models.

## Limitations and Open Questions

The authors are transparent about remaining unknowns:

### Study Limitations

1. **Task coverage** - Focused on reasoning/coding; didn't test dialogue, creativity, safety
2. **Single base model family** - Used LLaMA-based models; unclear how findings transfer to other architectures
3. **Compute ceiling** - Largest runs at 100K GPU-hours; behavior at 1M+ hours unknown
4. **Reward model fixed** - Used pre-trained reward models; didn't study joint training

### Open Research Questions

1. **What determines recipe asymptotes?** - Why do some configurations fundamentally cap lower?
2. **Can we predict good recipes a priori?** - Or do we always need empirical search?
3. **How do scaling laws change with task complexity?** - Do harder tasks have different curves?
4. **What's the role of model architecture?** - Do MoE models scale differently?

## Comparison to Related Work

### vs. Pre-training Scaling Laws (Kaplan et al., Hoffmann et al.)

| Aspect | Pre-training | RL Post-training (This Work) |
|--------|--------------|------------------------------|
| Curve shape | Power law | Sigmoidal |
| Recipe sensitivity | Low (most recipes converge) | High (recipe determines asymptote) |
| Predictability | Excellent (R² > 0.99) | Good (R² > 0.95 for stable recipes) |
| Compute range studied | Up to 10^24 FLOPs | Up to ~10^22 FLOPs |

### vs. Existing RL Studies

Previous RL scaling work (InstructGPT, RLHF papers) typically:
- Reported single-scale results
- Didn't fit predictive curves  
- Used different recipes across scales
- Didn't systematically ablate components

This work **systematically addresses all those gaps**.

## Code and Reproducibility

The authors commit to releasing:

1. **Full ScaleRL implementation** - PyTorch code with detailed comments
2. **Experiment configurations** - All hyperparameters for every run
3. **Scaling curve fits** - Data and scripts to reproduce curves
4. **Checkpoints** - Model checkpoints at key compute intervals

**Note:** At time of writing, code release pending (paper under review).

## Future Directions

The paper opens several exciting research avenues:

### 1. **Adaptive Recipe Selection**

Can we develop meta-learning approaches that automatically select/adapt recipes based on early training dynamics?

### 2. **Multi-Objective RL Scaling**

How do scaling laws change when optimizing for multiple objectives simultaneously (e.g., helpfulness + safety + efficiency)?

### 3. **Data Scaling Laws**

This work held data distribution roughly constant. How does the **diversity and quality** of RL data affect scaling?

### 4. **Joint Scaling with Inference Compute**

With techniques like best-of-N sampling and process rewards, how should we co-optimize **training compute** and **inference compute**?

### 5. **Theoretical Foundations**

Why sigmoidal curves? Can we derive them from first principles of RL dynamics?

## Takeaways for Different Audiences

### For ML Engineers

- Use ScaleRL as your default RL recipe
- Always fit scaling curves on small runs before going big
- Pay special attention to value function precision
- Don't over-optimize implementation details

### For Research Scientists

- Recipe selection (asymptote) matters more than you might think
- RL scaling is predictable with the right setup
- There's still much to understand about **why** certain recipes work
- Smaller-scale experiments can meaningfully predict larger scales

### For AI Safety Researchers

- Predictable RL scaling enables better safety planning
- We can now estimate compute requirements for safety-oriented RL
- Understanding recipe asymptotes helps identify fundamental capability bounds
- Stable recipes reduce risk of unexpected training dynamics

### For Executives/Decision Makers

- RL scaling is becoming **predictable and scientific**
- Compute investments can be planned with greater confidence
- Early experiments (5-10% of budget) can predict final performance
- Recipe choice is a critical strategic decision

## Conclusion

This work represents a **foundational contribution** to scaling reinforcement learning for LLMs. By demonstrating that:

1. RL scaling follows predictable sigmoidal curves (for stable recipes)
2. Recipe choice determines asymptotic performance
3. Implementation details affect efficiency but not fundamental capability

The authors provide the field with both **practical tools** (ScaleRL recipe) and **conceptual frameworks** (sigmoidal scaling laws) to navigate the increasingly important domain of RL post-training.

As we move toward more capable AI systems where reasoning and complex decision-making are paramount, having predictive scaling methodologies for RL is not just useful—it's **essential**. This paper delivers exactly that.

---

## Glossary

- **GRPO** - Group Relative Policy Optimization, a variant of PPO for language models
- **Asymptote** - The maximum performance level a recipe can achieve with infinite compute
- **Sigmoidal curve** - S-shaped curve with lower and upper asymptotes and a transition region
- **KL divergence** - Measure of how much the new policy differs from the old policy
- **Advantage** - Estimate of how much better an action is than average
- **Value function** - Estimate of expected future reward from a state

---

*This article is based on the research paper "The Art of Scaling Reinforcement Learning Compute for LLMs" by anonymous authors (under review). All findings and recommendations are attributed to the original research.*

---

### Related Reading

- **Scaling Laws for Neural Language Models** (Kaplan et al., 2020)
- **Training Compute-Optimal Large Language Models** (Hoffmann et al., 2022)
- **Learning to Summarize from Human Feedback** (Stiennon et al., 2020)
- **Training Language Models with RLHF** (Ouyang et al., 2022)

---

### Discussion Questions

1. How might these scaling laws change as models become even larger (1T+ parameters)?
2. Could similar methodology be applied to other post-training techniques (DPO, RLAIF)?
3. What are the implications for open-source vs. closed-source model development?
4. How should we think about the trade-off between recipe search cost and final training cost?

---

**About This Article**

This deep dive was created to make cutting-edge RL scaling research accessible to the broader AI community. If you found it helpful, consider sharing it with others interested in the science and engineering of training advanced AI systems.

**Have questions or insights?** The AI research community thrives on open discussion. Share your thoughts on how these findings might apply to your work or what additional experiments would be valuable.


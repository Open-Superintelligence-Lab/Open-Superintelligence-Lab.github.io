---
hero:
  title: "The Art of Scaling RL"
  subtitle: "🚀 Predictive Scaling Laws for Reinforcement Learning in LLMs"
  tags:
    - "⏱️ Research Deep Dive"
    - "📄 Research Article"
---

### **Hero Section**

**Title:** The Art of Scaling RL Becomes a Science  
**Subtitle:** 🚀 Predictive Scaling Laws for Reinforcement Learning in LLMs  
**Tags:**  
*   "⏱️ Research Deep Dive"  
*   "📄 Research Article"  

> **🚀 Key Insight:**  
> This groundbreaking research transforms Reinforcement Learning (RL) for LLMs from a dark art into a predictable science. It establishes the first-ever systematic scaling laws for RL, providing a stable "best-practice" recipe called **ScaleRL** that has been validated across an immense 400,000 GPU-hours of experiments.

### **Introduction: The Wild West of RL**

For years, scaling Large Language Models (LLMs) during pre-training has followed predictable rules, or "scaling laws." Researchers could reliably forecast a model's performance based on the amount of compute they invested. However, the crucial next step—fine-tuning with Reinforcement Learning (RL) to improve reasoning and instruction following—has remained a chaotic "wild west." Without predictive laws, teams have been forced to commit massive, expensive compute budgets while flying blind, hoping their chosen methods would scale.

This paper changes that. It presents the first open, large-scale, systematic study of RL scaling for LLMs. By fitting performance to compute, it demonstrates that RL *can* be scaled predictably and introduces a robust recipe to do so, turning uncertainty into a strategic advantage.

### **Predictably Scaling RL Compute**

A core discovery of this research is that with a stable training recipe, RL performance follows a predictable **sigmoidal curve**. This curve has three phases:
1.  **Initial rapid improvement** where the model learns quickly.
2.  A **predictable power-law phase** where performance scales consistently with compute.
3.  An **asymptotic plateau** where returns diminish.

Understanding this curve allows researchers to forecast performance, optimize resource allocation, and know when to stop training to avoid wasted compute, bringing much-needed engineering discipline to the field.

![Predictably Scaling RL Compute to 100k GPU Hours](/content/art-of-scaling-rl/images/predictably-scaling-rl-compute.png)

### **The ScaleRL Recipe: A Blueprint for Success**

After 400,000 GPU-hours of rigorous experimentation and ablation, the researchers engineered **ScaleRL**, a best-practice configuration for stable and predictable RL scaling. When tested against other common approaches, ScaleRL consistently demonstrates superior performance and stability.

![Comparing ScaleRL with prevalent recipes](/content/art-of-scaling-rl/images/comparing-scalerl-with-prevalent-recipes.png)

#### **Core Components of ScaleRL**

ScaleRL's success isn't based on a single trick, but on a series of carefully validated design choices:

1.  **Loss Function: CISPO for Stability**
    While the research began with popular algorithms like GRPO and DAPO, they were found to be highly sensitive to hyperparameter settings. The final ScaleRL recipe favors **CISPO (Clipped Importance Sampling Policy Optimization)**. Ablation studies (Figures 10 & 11) show CISPO is incredibly robust, delivering strong performance across a wide range of settings, making it a reliable choice for large-scale training where stability is paramount.

2.  **Off-Policy Training: PipelineRL for Efficiency**
    To maximize hardware utilization, RL often uses an "off-policy" approach where data generation and model training happen in parallel. The paper found that **PipelineRL** consistently outperforms standard methods. Its streaming-based approach creates a tighter feedback loop between the data generators and the trainer, keeping the training process closer to the ideal "on-policy" regime and boosting final performance.

3.  **Precision Strategy: A Calculated Mix**
    Training in full 32-bit precision (FP32) is stable but slow, while 16-bit (BF16) is fast but can be unstable. ScaleRL uses a smart mixed-precision strategy:
    *   **Policy Network (BF16):** For speed and memory efficiency.
    *   **Value Network & Advantage Calculation (FP32):** For higher numerical accuracy to prevent errors in the crucial reward signal.

4.  **Generation Control: Interruptions over Penalties**
    A common issue in reasoning tasks is that models can generate excessively long "chain-of-thought" responses, which harms efficiency. The paper compared two methods: adding a length penalty to the reward and forcibly interrupting the generation. **Interruptions** proved to be a more effective and stable way to control response length.

### **Key Design Choices: A Deep Dive into the Ablations**

The paper's true strength lies in its exhaustive experiments, which isolate the impact of each design choice.

#### **Figure 7: Fine-Tuning the Core Algorithm**
These experiments tested foundational components of the RL algorithm. The key takeaway was that while many choices seem small, they can have a significant impact on stability. For example, the method of advantage normalization (7b) and filtering out samples with zero variance (7c) both contribute to a smoother training process.

![Loss Aggregation Techniques](/content/art-of-scaling-rl/images/loss-aggregation-techniques.png)
*Figure 7a: Loss Aggregation Techniques*

![Advantage Normalization Techniques](/content/art-of-scaling-rl/images/advantage-normalization-techniques.png)
*Figure 7b: Advantage Normalization Techniques*

![0-Variance Filtering](/content/art-of-scaling-rl/images/zero-variance-filtering.png)
*Figure 7c: 0-Variance Filtering*

![No Positive Resampling vs Uniform Sampling](/content/art-of-scaling-rl/images/no-positive-resampling-vs-uniform-sampling.png)
*Figure 7d: No Positive Resampling vs Uniform Sampling*

#### **Figure 8 & 9: The Most Consequential Choices**
These "leave-one-out" experiments highlight the most critical decisions. Replacing CISPO with a less stable loss function (8a) or removing the FP32 precision fix (8b) both hurt performance. Most importantly, the choice of the off-policy algorithm (PipelineRL) was found to be one of the most consequential decisions, affecting the ultimate performance ceiling of the model.

![Comparing loss functions](/content/art-of-scaling-rl/images/comparing-loss-functions.png)
*Figure 8a: The right loss function is crucial for stability.*

![FP32 Precision Fix at the LM Head](/content/art-of-scaling-rl/images/fp32-precision-fix-at-lm-head.png)
*Figure 8b: Mixed precision provides speed without sacrificing stability.*

![Comparing max off-policy](/content/art-of-scaling-rl/images/comparing-max-off-policy.png)
*Figure 8c: Comparing off-policy strategies.*

![Comparing off-policy algorithms](/content/art-of-scaling-rl/images/comparing-off-policy-algorithms.png)
*Figure 9a: PipelineRL offers a clear advantage.*

#### **Figure 10 & 11: The Robustness of CISPO and GSPO**
These plots show how different loss functions react to changes in their clipping ratio, a key hyperparameter. DAPO/GRPO-style losses are very sensitive, meaning a small change can destabilize training. In contrast, **CISPO** (Figure 11a) is remarkably robust, performing well across a wide range of values. This makes it a much safer and more reliable choice for large, expensive training runs.

![GSPO Clipping Ratio Scale](/content/art-of-scaling-rl/images/gspo-clipping-ratio-scale.png)
*Figure 10a: GSPO Clipping Ratio Scale*

![CISPO Upper Clipping Ratio](/content/art-of-scaling-rl/images/cispo-upper-clipping-ratio.png)
*Figure 11a: CISPO shows impressive robustness to hyperparameter changes.*

#### **Figure 12: The Entropy Misconception**
Entropy is often used as a proxy for exploration in RL. A common belief is that maintaining high entropy is good for performance. This research shows that isn't necessarily true. The larger batch size run achieved much better performance despite following an almost identical entropy curve as the smaller batch run. This suggests that **batch size is a far more decisive factor for performance than entropy**.

![Entropy across training run](/content/art-of-scaling-rl/images/entropy-across-training-run.png)
*Figure 12a: Entropy is not a reliable predictor of performance; batch size is key.*

### **Downstream Performance: Does it Generalize?**
Great performance during training is meaningless if it doesn't translate to real-world tasks. The evaluation on the AIME-25 benchmark reveals a critical pattern:
**Larger batch sizes are essential for downstream generalization.**

Runs with smaller batches showed early stagnation on the benchmark, even while their training metrics continued to improve. In contrast, larger-batch runs avoided this stagnation, with their downstream performance mirroring the predictable power-law scaling seen during training. This is a vital lesson: for RL to produce models that are actually more capable, scaling up the batch size is non-negotiable.

![AIME-25 Pass Rate](/content/art-of-scaling-rl/images/aime-25-pass-rate.png)
*AIME-25 Pass Rate vs. Compute*

![AIME-25, Scaling Batch Size](/content/art-of-scaling-rl/images/aime-25-scaling-batch-size.png)
*Scaling Batch Size on AIME-25*

![AIME-25, Scaling Generation Length](/content/art-of-scaling-rl/images/aime-25-scaling-generation-length.png)
*Scaling Generation Length on AIME-25*

![AIME-25, Scaling Model Size](/content/art-of-scaling-rl/images/aime-25-scaling-model-size.png)
*Scaling Model Size on AIME-25*

### **Practical Implications**

#### **For Researchers**
1.  **A Solid Foundation:** Use ScaleRL as a strong, validated baseline for future RL research.
2.  **Predict Before You Scale:** Run smaller experiments (1k-5k GPU-hours) to fit initial curves and forecast performance before committing to massive runs.
3.  **Focus on the Recipe:** The choice of algorithm and core components (which affects the performance ceiling) is more critical than minor implementation tweaks.

#### **For Practitioners**
1.  **Budget with Confidence:** Use scaling curves to estimate the compute required to reach a target performance level.
2.  **Avoid Wasted Compute:** Identify when your training run is approaching its asymptote and stop before returns diminish.
3.  **Prioritize Stability:** Choose robust components like the CISPO loss function to de-risk large-scale training.

### **Conclusion**

This work is a landmark achievement in the effort to engineer powerful AI systems. By systematically charting the unexplored territory of RL scaling, the authors have provided a map and a compass for the entire field. The introduction of predictive scaling laws and the open, robust ScaleRL recipe moves RL post-training from an unpredictable art to a reliable and scalable science, paving the way for the next generation of advanced reasoning models.
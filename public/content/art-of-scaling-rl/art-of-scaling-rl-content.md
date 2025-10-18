---
hero:
  title: "The Art of Scaling RL"
  subtitle: "🚀 Predictive Scaling Laws for Reinforcement Learning in LLMs"
  tags:
    - "⏱️ Research Deep Dive"
    - "📄 Research Article"
---

Of course. Here is a detailed explanation of the research paper "The Art of Scaling Reinforcement Learning Compute for LLMs".

### **1. High-Level Summary**

This paper addresses a critical gap in the development of Large Language Models (LLMs): the lack of a scientific, predictive methodology for scaling Reinforcement Learning (RL) training. While pre-training LLMs follows well-understood "scaling laws," RL fine-tuning has been more of an unpredictable "art."

The authors conduct a massive empirical study (over 400,000 GPU-hours) to establish a scientific framework for analyzing and predicting the performance of RL training as compute increases. Their main contributions are:

*   **A Predictive Framework:** They propose using a sigmoidal (S-shaped) curve to model the relationship between compute and performance, allowing them to predict a method's ultimate performance ceiling and efficiency from smaller-scale experiments.
*   **A State-of-the-Art Recipe (ScaleRL):** Based on extensive experiments, they combine the best-performing techniques into a single, stable, and highly scalable recipe called `ScaleRL`.
*   **Large-Scale Validation:** They successfully demonstrate the predictability of their framework by scaling a single `ScaleRL` training run up to 100,000 GPU-hours, showing that the final performance closely matched the curve extrapolated from the initial stages of training (Figure 1).

Essentially, the paper provides both a scientific method to study RL scaling and a practical recipe that makes large-scale RL training as predictable as pre-training.

### **2. The Core Problem: RL Scaling as an "Art"**

While RL is crucial for unlocking advanced capabilities in LLMs like complex reasoning and agentic behavior, the compute budgets for this phase are skyrocketing. The paper notes that RL compute has increased more than 10x for frontier models.

However, unlike pre-training where performance predictably improves with more data and compute, RL training has lacked a principled methodology. Practitioners face a huge number of design choices (e.g., which loss function to use, how to normalize rewards, how to set up the training pipeline) with no reliable way to know which choices will perform best at a massive scale without running costly experiments to completion. This makes progress slow and expensive.

### **3. The Proposed Scientific Framework**

To move from "art" to "science," the authors propose a mathematical framework to model performance.

#### The Sigmoidal Scaling Curve
They model the performance (Reward) as a function of training Compute (`C`) using a sigmoidal curve:

`Reward Gain = Asymptotic Reward Gain × Compute Efficiency`

This allows them to analyze two key properties of any RL recipe (see Figure 3 in the paper):

1.  **Asymptotic Performance (A):** The maximum possible performance or "ceiling" that a method can achieve, no matter how much compute is used. This is the most important parameter.
2.  **Compute Efficiency (B and Cmid):** How quickly the method approaches its performance ceiling. A higher `B` value means a steeper, more efficient curve.

This framework is powerful because it allows researchers to fit the curve using data from the early stages of a training run and then **extrapolate** to predict how it will perform at a much larger compute budget.

### **4. Key Findings from the Empirical Study**

The authors used their framework to systematically test numerous design choices on an 8B parameter model. Their investigation yielded three key principles:

*   **Performance Ceilings Are Not Universal:** Different RL methods hit different performance ceilings. For instance, the choice of loss function (`CISPO` vs. `DAPO`) and using high-precision math (`FP32`) for the final model layer were found to significantly raise the asymptotic performance `A` (Figure 5).
*   **The "Bitter Lesson" of Scaling:** A method that looks superior at a small compute budget may be worse when scaled up. Figure 2 shows that while other methods might perform well initially, `ScaleRL` has a higher performance ceiling, eventually surpassing them. The scaling framework helps identify these more promising long-term candidates early on.
*   **Re-evaluating Common Wisdom:** Many popular techniques, such as data curriculum, advantage normalization, and loss aggregation, were found to primarily affect *compute efficiency* (`B`) without changing the ultimate performance ceiling (`A`).

### **5. ScaleRL: A Predictable and Scalable Recipe**

By combining the best-performing components identified in their study, the authors created the `ScaleRL` recipe. It is not a single new algorithm but a carefully validated combination of existing techniques.

### **5. ScaleRL: A Predictable and Scalable Recipe**

By combining the best-performing components identified in their study, the authors created the `ScaleRL` recipe. It is not a single new algorithm but a carefully validated combination of existing techniques designed for stability and performance at scale.

**Key Components of ScaleRL Explained:**

*   **Asynchronous Setup (`PipelineRL`):** Instead of a classic setup where training and data generation happen in separate, alternating phases, `ScaleRL` uses `PipelineRL`. This creates a streaming data pipeline between the model generating the text (the generator) and the model being updated (the trainer). This tight feedback loop keeps the training data fresher and closer to the current model policy, which was found to be a critical factor in raising the ultimate performance ceiling.

*   **Loss Function (`CISPO`):** The paper chose `CISPO` (Clipped Importance Sampling Policy Optimization) because it provided the best balance of high performance and stability. Unlike `DAPO`, its performance is not highly sensitive to the exact choice of hyperparameters, making it more robust for large-scale runs. While another alternative, `GSPO`, was also robust, it was prone to sudden training instability. `CISPO` reliably avoids these issues.

*   **Precision (`FP32` Logits):** While most of the model can be trained with lower precision (like BFloat16) to save memory and increase speed, the paper found that casting the model's final output layer (the logits) to high precision (`FP32`) was crucial. This small change dramatically improved the model's asymptotic performance, preventing a premature plateau in its learning curve.

*   **Normalization & Aggregation:**
    *   **Batch-Level Advantage Normalization:** The "advantage" (a measure of how much better a given action is than the average) is normalized across the entire batch of generated text. This helps stabilize the learning signal.
    *   **Prompt-Level Loss Averaging:** The loss is averaged for all the text generated from a single prompt. This ensures that each prompt contributes equally to the training signal, regardless of how many tokens are in the generated responses.

*   **Length Control (Interruptions):** To prevent the model from generating overly long and inefficient reasoning chains, `ScaleRL` uses **interruptions**. If a generation exceeds a certain length (e.g., 12,000 tokens), it is forcibly stopped with a message that signals the model to wrap up its thoughts and provide a final answer. This was found to be more effective than penalizing the reward for long sequences and was critical for maintaining training stability by keeping truncation rates low.

*   **Data Filtering (Curriculum Learning):**
    *   **Zero-Variance Filtering:** If the model generates multiple responses to a prompt and all of them receive the same reward (e.g., all are correct or all are incorrect), that prompt is temporarily ignored. This avoids wasting compute on problems that are either too easy or currently too hard.
    *   **No-Positive-Resampling:** Once the model has reliably solved a specific prompt (i.e., consistently generates a correct answer), that prompt is removed from the training set for that run. This is a curriculum strategy that focuses the model's training time on the problems it still needs to master.

To validate these choices, they conducted **Leave-One-Out (LOO)** experiments, where they started with the full `ScaleRL` recipe and reverted one component at a time. In all cases, the full `ScaleRL` recipe was either the best or among the best in both performance and efficiency (Figure 7).

### **6. Scaling Predictably Across Different Axes**



The paper demonstrates that the predictive power of `ScaleRL` and the sigmoidal framework holds true not just for increasing training time, but also when scaling other important factors:



*   **Model Size:** The framework accurately predicted the performance of a much larger 17B×16 Mixture-of-Experts (MoE) model (Figure 1). As expected, the larger model achieved a significantly higher performance ceiling.

*   **Generation Length:** Training with a longer context (32k tokens vs. 14k) was less efficient initially but ultimately led to a higher performance ceiling (Figure 9).

*   **Batch Size:** Larger batch sizes also led to a higher asymptotic performance, though they were slower to show gains initially (Figure 10).

*   **Multi-Task Learning:** When trained jointly on math and code, `ScaleRL` showed predictable, parallel scaling trends for both domains (Figure 11).



### **7. Deeper Dive: The "Why" Behind the Choices**



The paper's appendix provides crucial insights into *why* certain components of `ScaleRL` were chosen.



*   **PipelineRL vs. PPO-off-policy:** `PipelineRL` consistently outperforms classic PPO-off-policy because it operates in a streaming fashion. This creates a tighter feedback loop between the generator and the trainer, keeping the training closer to an on-policy regime and reducing the mismatch between their data distributions. This choice was found to be one of the most consequential design decisions, affecting the asymptotic performance ceiling, not just efficiency.



*   **Loss Function Robustness (CISPO vs. GSPO vs. DAPO):** The choice of `CISPO` was driven by its stability and robustness. While `DAPO`-style losses are highly sensitive to the exact value of their clipping hyperparameters, `CISPO` and `GSPO` are much more forgiving. However, the researchers encountered stability issues with `GSPO` during large-scale runs, where the model would diverge mid-training. `CISPO` offered the best balance of performance and stability, making it the recommended choice.



*   **Controlling Generation Length:** The paper investigated two methods to prevent "exploding" generation lengths: **interruptions** (forcibly stopping generation) and **length penalties** (modifying the reward). `ScaleRL` uses interruptions, as replacing them with length penalties did not improve performance. The researchers also found that high truncation rates (10-15%) were a reliable warning sign of training instability. `ScaleRL`'s stability is partly attributed to keeping truncation rates low (below 5% for the 8B model).



*   **The Importance of Batch Size for Generalization:** While larger batch sizes were slower to show gains, they were crucial for avoiding **early stagnation on downstream benchmarks**. Smaller-batch runs improved on the training data but failed to generalize, whereas larger-batch runs exhibited scaling curves on downstream tasks that mirrored their in-distribution performance.



### **8. Conclusion and Impact**



This work establishes a rigorous, scientific methodology for developing and evaluating RL algorithms for LLMs at scale. By introducing a predictive framework, it allows the research community to identify promising methods cost-effectively, without needing to run every experiment to its computational limit. The proposed `ScaleRL` recipe serves as a robust, state-of-the-art baseline that scales predictably to over 100,000 GPU-hours, bringing the field of RL training for LLMs closer to the predictability long achieved in pre-training.
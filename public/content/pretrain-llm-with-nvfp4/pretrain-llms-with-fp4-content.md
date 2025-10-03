
# A Technical Guide to LLM Pretraining with NVFP4

*An overview of NVIDIA's 4-bit floating point format for efficient and accurate model training, based on the technical report "Pretraining Large Language Models with NVFP4".*

The growing scale of Large Language Models (LLMs) necessitates more efficient training methods. While 8-bit floating point (FP8) training is widely adopted, 4-bit floating point (FP4) formats offer further improvements in computational speed and memory usage. This guide provides a technical summary of **NVFP4**, a 4-bit format from NVIDIA, and the methodology required for its successful implementation in LLM pretraining.

## Background: Key Concepts in Numerical Precision

Before diving into NVFP4, it's essential to understand a few foundational concepts.

You can copy the content below into an AI chatbot for personalized lessons.

*   **Numerical Precision:** In deep learning, numbers are typically stored in floating-point formats (e.g., FP32, FP16, BF16, FP8, FP4). The number in the format name indicates the number of bits used to represent a single number. More bits (like in FP32) allow for a wider range of numbers and higher precision (more detail), but consume more memory and are computationally slower. Fewer bits (like in FP4) are faster and more memory-efficient but have lower precision.

*   **Quantization:** This is the process of converting a tensor from a higher-precision format (e.g., FP32) to a lower-precision one (e.g., FP4). This is the core technique for accelerating model training and inference. However, this process can lead to a loss of information, which, if not managed correctly, can degrade the model's accuracy.

*   **Dynamic Range:** This refers to the range of values that can be represented by a numerical format, from the smallest non-zero number to the largest. When quantizing, we often scale the values in a tensor to fit within the limited dynamic range of the target format (e.g., FP4). A key challenge is that a single very large value (an "outlier") can dominate the entire range, forcing all other smaller values to be quantized to zero or near-zero, effectively erasing their information.

## The Outlier Problem in Scaling (An Example)

The presence of outliers - ex. `50` in `[0.5, -0.2, 1.1, -0.8, 50.0]` - is a major challenge in quantization. Because the scaling factor for a block of numbers is determined by the single largest value, one outlier can ruin the precision for all other numbers in its block.
    *   **Scenario:** Imagine we have a small block of numbers: `[0.5, -0.2, 1.1, -0.8, 50.0]`
    *   **The Outlier:** The value `50.0` is a significant outlier.
    *   **Scaling:** To quantize this block into FP4, which has a maximum representable value of `6.0`, we must scale every number down with the same scaling factor. `Scale Factor = 6.0 / 50.0 = 0.12`. The scaling factor is based on the largest number (taken absolute value)
    *   **Result:** After scaling (multiplying), our block becomes `[0.06, -0.024, 0.132, -0.096, 6.0]`.
    Here, the values that can be represented in FP4 (E2M1) are ±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, and ±6. This means that after scaling, any value in the block will be rounded to the nearest of these discrete numbers. The representable range is thus from -6 to +6, with only these specific values available.
    *   **Information Loss:** When these new values are converted to the closest representable FP4 number (e.g., `±0`, `±0.5`, `±1.0`...), the first four values are so small that they will likely all be rounded to zero. The original information they contained is lost. Only the outlier retains its significance. NVFP4's techniques are designed to mitigate exactly this problem.

## Technical Advantages of NVFP4

Transitioning from FP8 to FP4 can yield a 2-3x increase in **arithmetic performance**—primarily the throughput of General Matrix Multiplication (GEMM) operations, which are the computational core of transformers—and a 50% reduction in memory usage. However, the lower precision introduces challenges. NVFP4 is designed to address these issues through several key features:

*   **Two-Level Scaling for High-Precision Representation:** In short, there are two scaling factors: one that applies to an entire tensor (like weights or activations, often millions of values), and a second that applies to each 16-element block within that tensor.

![NVFP4 Matrix Storage Format](/images/NVFP4_matrix_storage_format.png)
*Figure 1: A 16x32 matrix stored in NVFP4 format. Each block contains sixteen contiguous FP4 elements (gray and green) along with a single FP8 scale factor (yellow). The element with the largest magnitude in each block (green) is scaled to the FP4 maximum representable value and can be recovered using the block scale factor. A per-tensor FP32 scale factor (not shown) is also applied.*


NVFP4 uses two distinct scaling factors, which is its most critical feature. To understand this, let's define two terms:
    *   A **Tensor** is a large, multi-dimensional array of numbers that holds the model's weights or activations. These are the core data structures in a neural network, and their scale can be immense. Here are some concrete examples based on the models described in the paper:
        *   **Weight Tensor:** In the 12B model, a single FFN weight tensor can have over 104 million values (shape: [5120, 20480]).
        *   **Activation Tensor:** In the 8B model, activations (layer outputs) between layers can form a 2D tensor of shape [6.3M, 4096] (from batch size 768 × sequence length 8192 × model dim 4096).
    *   A **Block** is a small, fixed-size chunk of a tensor. In NVFP4, a block is a group of just 16 contiguous numbers. So, the enormous weight and activation tensors above would be partitioned into thousands or millions of these tiny blocks for quantization.
    
    The two-level scaling works as follows:
    1.  **Coarse, Per-Tensor Scaling (FP32):** First, a single scaling factor is calculated for the *entire tensor* based on its absolute maximum value (max value(abs(tensor))). This factor, stored in high-precision **FP32**, performs a rough, global adjustment, bringing all the values in the tensor into a more manageable intermediate range without the scale factor itself becoming a source of error. Using a high-precision format FP32 is crucial because an imprecise scale factor would inaccurately shrink every value in the tensor, adding error on top of the final quantization step.
    2.  **Fine-Grained, Per-Block Scaling (FP8):** After the first scaling is applied, the tensor is divided into thousands of small 16-element blocks. For *each* of these blocks, a second, more precise scaling factor is calculated. This local factor, stored in **FP8**, makes a fine-tuned adjustment, perfectly mapping the 16 values to the extremely limited range of FP4. Using FP8 for the block-level scale provides enough precision for local adjustments while remaining efficient for the hardware to process. The FP4 format used here (E2M1) can only represent the values ±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, and ±6.
    
    This dual approach is powerful because it allows for highly localized adaptation. A large outlier in one part of the tensor will only affect the scaling of its tiny 16-element block, leaving the quantization of all other blocks completely unaffected. This preserves significantly more information compared to a single scaling factor.
*   **Reduced Block Size for Better Dynamic Range:** NVFP4 uses a smaller micro-block size of 16 elements. This is crucial for **capturing the local dynamic range**. In simpler terms, if a block of numbers contains one large outlier, only the other 15 numbers in that small block are affected during scaling. In a larger block (e.g., 32 elements), that same outlier would force a less precise scaling for all 31 other numbers, potentially causing more information loss. The smaller block size isolates the impact of outliers.
*   **Native Hardware Support:** The NVIDIA Blackwell GPU architecture includes Tensor Cores with native support for NVFP4, enabling significant hardware acceleration for GEMM operations.

![NVIDIA Blackwell Tensor Cores](/images/NVIDIA_Blackwell_Tensor_Cores.png)
*Table 1: NVIDIA Blackwell Tensor Cores. This table shows the speedup of NVFP4 and other formats compared to BF16 on GB200 and GB300 GPUs.*


These design choices allow NVFP4 to provide the efficiency of 4-bit precision while **mitigating the typical trade-offs**, namely the loss of model accuracy and potential for training instability that can arise from aggressive quantization.

## Core Methodology for NVFP4 Training

Achieving training outcomes comparable to FP8 requires a specific set of techniques. The following methodology is recommended for stable and accurate pretraining with NVFP4.

![Combining NVFP4 Training Techniques](/images/combining_NVFP4_training_techniques.png)
*Figure 8: Combining NVFP4 training techniques: linear layers in last four blocks in BF16, 2D weight scaling, Random Hadamard transforms on Wgrad, and stochastic rounding on gradients. Plot shows relative difference in validation loss for a 1.2B model trained on 1T tokens.*


### 1. Mixed-Precision Strategy

Quantizing the entire model to FP4 can lead to divergence (model stops learning). A mixed-precision approach is crucial for stability.

![NVFP4 Quantized Linear Layer Compute Flow](/images/NVFP4_quantized_linear_layer_compute_flow.png)
*Figure 5: Illustration of compute flow for a NVFP4 quantized linear layer. All GEMM operations quantize their inputs to NVFP4.*


**Implementation:**
*   Use NVFP4 for the majority of GEMM operations within the linear (fully-connected) layers.
*   Maintain a small percentage of numerically sensitive linear layers (approx. 15%) in a higher precision format like BF16. The paper found that the **final layers** of the network are the most sensitive, as they require a greater dynamic range and more precision than FP4 can provide. Keeping the first and last few blocks of the model in a higher format is often sufficient to ensure stable training.

![NVFP4 BF16 Precision Switching](/images/NVFP4_BF16_precision_switching.png)
*Figure 7: Switching to higher precision towards end of training. Plot shows relative difference in validation loss for a 12B model trained on 10T tokens. NVFP4 uses the method specified in Section 4 during all of the training period (Green). The precision for tensors in forward and backward pass (Blue), tensors only in the forward pass (Orange), and tensors only in the backward pass (Purple) are switched from NVFP4 to BF16 at 8.2T tokens until remainder of training. A run where the switch to high precision occurs around 10T tokens is also shown (Red). 1D weight scaling is used when switching precision for the backward pass, since doing so is marginally better than 2D weight scaling in such a setup.*

*   Keep other critical components in their original precision (BF16 or FP32) to ensure numerical stability. This includes embeddings, the output projection head, normalization layers, non-linearities, and most parts of the attention mechanism (softmax, etc.). Only the large GEMM operations in the transformer blocks are targeted for FP4 quantization.

![Linear Layer Sensitivity to Quantization](/images/linear_layer_sensitivity_to_quantization.png)
*Figure 9: Sensitivity of linear layers to quantization. NVFP4 for all linear layers except in a few of the first and last blocks in the model. Plot shows validation loss for a 1.2B model trained on 1T tokens.*

  
  ### 2. Random Hadamard Transforms (RHT) for Outlier Management

  This is a cool trick. If you want to quantize both `Activations × Weights` to FP4, you can have a matrix `H`, such that `H × H = I (Identity Matrix)` -> `H` is orthogonal.

  Then you can do `(Activations × H) × (H × Weights)`

  This will have the same result as `Activations × Weights` but it will reduce issues with outliers (precision loss) during quantization of `Activations` and `Weights`.
  
  **An Example in Action:**
  *   **Original Block:** Consider a small block of four numbers: `[1.0, -2.0, 1.5, 30.0]`.
  *   **The Problem:** The outlier is `30.0`. To quantize this, everything must be scaled down based on this large value, causing the first three numbers to lose their precision.
  *   **The RHT Transform:** After applying the Hadamard transform, the block becomes: `[15.25, -12.75, -16.25, 15.75]`.
  *   **The Result:** The large outlier `30.0` is gone. The energy has been redistributed, and the new maximum absolute value is only `16.25`.
  *   **The Benefit:** When this new block is scaled to fit into FP4's range, the scaling factor is almost twice as large. This means the other values are scaled to larger, more distinct numbers, preserving significantly more of their original information before the final rounding to FP4.
  
  **The Math Behind the Example:**
  
  The transform is a matrix-vector multiplication. For the 4-number block in the example, the calculation uses a normalized 4x4 Hadamard matrix (`H`):
  ```
            [ 0.5,  0.5,  0.5,  0.5 ]
  H =       [ 0.5, -0.5,  0.5, -0.5 ]
            [ 0.5,  0.5, -0.5, -0.5 ]
            [ 0.5, -0.5, -0.5,  0.5 ]
  ```
  Multiplying the original block `[1.0, -2.0, 1.5, 30.0]` by this matrix (`[1.0, -2.0, 1.5, 30.0] × H`) gives the new values.
  
  **What is this matrix?** The matrix `H` is a normalized **Hadamard matrix**, a fixed, constant matrix chosen for its key property: **orthogonality**. This property guarantees the transform is perfectly reversible (`H × H^T = I`). The practical implication for researchers is that this isn't a learned parameter but a standard mathematical tool that allows for a temporary, lossless transformation of the data into a more quantization-friendly format.

  **Implementation:**
  
  *   **Target the Right Operation:** RHT is not applied everywhere. The paper found it was most critical for stability when applied to the **weight gradient (`Wgrad`) calculation**. This is the part of the backward pass where the model calculates the updates for its weights. Applying it elsewhere (like the forward pass) provided no benefit and could even hurt performance.

![Hadamard Transform Impact on Validation Loss](/images/hadamard_transform_impact_on_validation_loss.png)
*Figure 11: Impact of applying Random Hadamard Transforms (RHT) to different GEMMs (Fprop, Dgrad and Wgrad) during training, compared to no RHT. For RHT runs, each transform uses a fixed random seed across the entire training. NVFP4 quantization is applied to all linear layers except in the last four blocks. The plot shows the relative change in validation loss compared to the BF16 baseline for a 1.2B-parameter model trained on 1T tokens.*

  *   **Choose an Effective Matrix Size:** The transform is performed by multiplying the data with a "Hadamard Matrix." A larger matrix spreads outliers more effectively but is more computationally expensive. The paper found that a **16x16 matrix** provides the best trade-off for large models, offering strong outlier mitigation without too much compute overhead.

![Hadamard Matrix Size Effect](/images/hadamard_matrix_size_effect.png)
*Figure 12: Effect of varying Hadamard Matrix Size. Wgrad tensors use 16 × 16 transforms for the first 3.4T tokens, then switch to 4 × 4 or 128 × 128 for the remainder of training. Plot shows relative difference in training loss for the 12B model trained on 4T tokens. NVFP4 is applied on linear layers using the methodology specified in Section 4.*

  *   **Use Randomization to Fix "Structural Alignment":** The "Random" in RHT is a simple fix for a rare but critical failure case. The issue, called **structural alignment**, occurs when a block of data, by pure chance, has a sign pattern that perfectly mirrors a pattern in the fixed Hadamard matrix. This alignment causes the transform to fail and *create* a new outlier instead of removing one.
      *   **The Problem in Action:** Imagine a block of data is `[10, 8, -12, -9]`, which has a sign pattern of `[+, +, -, -]`. A row in the Hadamard matrix has the same `[+, +, -, -]` pattern. When the transform is applied, the matching signs cause all the numbers to add up constructively (`10 + 8 + 12 + 9 = 39`), creating a new, massive outlier.
      *   **The Fix in Action:** Randomization fixes this by randomly flipping the signs of the transform's rows, changing the pattern to something like `[+, -, +, -]`. When this new, misaligned pattern is applied to the same data, the values now cancel each other out (`10 - 8 - 12 + 9 = -1`), preventing the creation of a new outlier.
      *   **Practical Takeaway:** To prevent this, the Hadamard matrix itself is randomized. The paper found that creating a single random matrix once and reusing it for the entire training run was sufficient.

![Hadamard Transform Randomization Effect](/images/hadamard_transform_randomization_effect.png)
*Figure 13: Effect of randomization for the Hadamard transform. A single fixed seed is used for all transforms during the first 3.4 tokens and switched to one of the following randomization options for the remainder of training: a single fixed seed for all layers, a unique seed for every transform, and not using a random sign vector. Plot shows relative difference in training loss from the FP8 baseline for a 12B model trained on 4T tokens. NVFP4 training uses the training methodology specified in Section 4.*

  
  ### 3. Two-Dimensional (2D) Weight Scaling for Consistent Quantization
  
  To understand this technique, imagine a tiny 2x2 block of weights from a larger matrix: `[[W₁₁, W₁₂], [W₂₁, W₂₂]]`.
  
  **The Problem (Inconsistent 1D Scaling):**
  During training, this block is processed differently in the two main passes:
  *   **Forward Pass (Row-wise):** The weight `W₁₁` is grouped with its row-mate `W₁₂`. They are scaled together based on `max(abs(W₁₁), abs(W₁₂))`.
  *   **Backward Pass (Column-wise):** Because the weight matrix is transposed for the backward pass, `W₁₁` is now grouped with its column-mate `W₂₁`. They are scaled together based on `max(abs(W₁₁), abs(W₂₁))`.
  For example, if `W₁₁ = 2.0`, `W₁₂ = 10.0`, and `W₂₁ = 0.5`, then in the forward pass (row-wise scaling), `W₁₁` is quantized using the max of its row (`max(2.0, 10.0) = 10.0`), but in the backward pass (column-wise scaling), it's quantized using the max of its column (`max(2.0, 0.5) = 2.0`). So `W₁₁` ends up with two different quantized values, breaking the chain rule.
  
  **The Solution (Consistent 2D Scaling):**
  Instead of scaling row-by-row, 2D scaling treats the entire 2x2 block as a single unit.
  *   **How it works:** A *single* scaling factor is calculated for the whole block, based on the maximum absolute value of all four weights: `max(abs(W₁₁), abs(W₁₂), abs(W₂₁), abs(W₂₂))`.
  *   **The Result:** Because the same scaling factor is used for the entire square, it doesn't matter if it's processed row-wise or column-wise. The quantized value for `W₁₁` is now guaranteed to be the same in both the forward and backward passes, preserving consistency.
  
  **Practical Takeaway:** The paper applies this principle using larger 16x16 blocks. Use 16x16 2D block scaling for weight tensors to ensure consistency. For activations and gradients, standard 1D scaling is sufficient, as training is less sensitive to inconsistencies in those tensors.

![Tensor Consistency Effect](/images/tensor_consistency_effect.png)
*Figure 14: Effect of consistency in tensors. Relative difference in validation loss from the BF16 baseline for a 1.2B model trained on 1T tokens. NVFP4 is applied on either weights or activations. Different choices of scaling factors are applied: 1 × 16 block scales along the same dimension, 1 × 16 block scales along different dimensions, and 16 × 16 block scales, along with a global FP32 per-tensor scale.*

  
  > **Why are weights more sensitive than activations?**
  > The core difference is their role and lifespan during training:
  > *   **Weights are the model's "long-term memory."** They are persistent parameters that are learned and updated over the entire training process. An inconsistency in a weight's value has a lasting, cascading impact because it's used in every single forward and backward pass, corrupting the learning signal over time.
  > *   **Activations are "fleeting thoughts."** They are transient values, calculated in a forward pass, used once in the corresponding backward pass, and then discarded. An inconsistency here has a much more localized and temporary effect.
  > Therefore, the extra effort of 2D scaling is a crucial investment for the persistent weight tensors but offers diminishing returns for the transient activation tensors.
  
  ### 4. Stochastic Rounding for Unbiased Gradients
  
  When quantizing, many values will fall between the few representable points of FP4. Standard rounding (e.g., always rounding to the nearest value) can introduce a systematic bias. If slightly more numbers round down than up, for instance, there's a consistent downward drift in the values, which can harm learning.
  
  **The Problem: Systematic Bias in Standard Rounding**
  
  Imagine your only representable FP4 values are `0` and `1`. Now, consider a block of four high-precision gradient values: `[0.6, 0.7, 0.8, 0.9]`.
  *   **Standard Rounding:** Using the "round-to-nearest" rule, all four of these values are rounded up to `1`. The new block is `[1, 1, 1, 1]`.
  *   **The Bias:** The average of the original numbers was `0.75`. The average of the rounded numbers is `1.0`. The rounding has systematically pushed the average value up, introducing an upward bias into the gradient signal. Over thousands of training steps, this small, consistent error can accumulate and lead the model astray.
  
  **The Solution: Unbiased Stochastic Rounding**
  
  Stochastic rounding is a probabilistic method that eliminates this bias on average.
  *   **How it works:** Instead of rounding deterministically, it rounds up or down with a probability proportional to the number's distance from the two nearest representable values.
  *   **In Action:** For the value `0.7`, it has a 70% chance of being rounded up to `1` and a 30% chance of being rounded down to `0`.
  *   **The Result:** Over a large number of values, this method is statistically unbiased. The *expected* value of rounding `0.7` is `(0.7 * 1) + (0.3 * 0) = 0.7`, which is exactly the original number. It introduces a bit of randomness (noise) to each individual operation, but it ensures that the overall gradient signal remains true over time.
  
  **Practical Takeaway:** The paper found it was essential to apply **stochastic rounding** when quantizing gradient tensors. However, for weights and activations in the forward pass, standard **round-to-nearest-even** is better, as the noise from stochastic rounding can be harmful there.

![Ablations on 12B model](/images/ablations_on_12B_model.png)
*Figure 4: Ablations on the 12B model trained for 10T tokens. Ablation studies start from the model trained up to 3.43T tokens using NVFP4 except in the first two and last eight blocks, and systematically remove one methodology component at a time: stochastic rounding (SR), Random Hadamard Transforms (RHT), two-dimensional scaling (2D), and fewer blocks in BF16. Relative difference is defined as (FP8 - experiment) / FP8, where a negative difference means the experiment is worse.*

![Stochastic Rounding on Different Tensors](/images/stochastic_rounding_on_different_tensors.png)
*Figure 10: Stochastic rounding applied to different tensors: gradients, activations, weights, and backward-pass tensors. NVFP4 is applied on all linear layers except in the last four blocks. Plot shows validation loss for a 1.2B model trained on 1T tokens.*

  
  ## Empirical Validation: 12B Model on 10T Tokens
  
  The paper validates this four-part methodology by pretraining a 12-billion-parameter model on an unprecedented 10 trillion tokens.

**Results:**
*   **Training Loss:** The validation loss for the NVFP4-trained model closely matched the FP8 baseline throughout the 10T token run.

![NVFP4 vs FP8](/images/NVFP4_vs_FP8.png)
*Figure 2: Validation loss of NVFP4 and FP8 pretraining for the 12B model using 10T tokens.*

*   **Downstream Task Accuracy:** The NVFP4 model achieved accuracies comparable to the FP8 baseline across a diverse set of downstream tasks, including reasoning, mathematics, and code generation. For example, the NVFP4 model achieved an MMLU-pro accuracy of 62.58%, nearly identical to the FP8 model's 62.62%.

![Task Accuracy NVFP4 vs FP8](/images/task_accuracy_nvfp4_vs_fp8.png)
*Figure 3: Task accuracy of NVFP4 versus FP8 measured throughout 10T tokens of pretraining.*


This result constitutes the longest publicly documented 4-bit training run and demonstrates the viability of NVFP4 for large-scale pretraining.

## Format Comparison: NVFP4 vs. MXFP4

In a direct comparison using an 8B parameter model, NVFP4 demonstrated superior convergence over the MXFP4 format.

![NVFP4 vs MXFP4 Comparisons](/images/NVFP4_vs_MXFP4_comparisons.png)
*Figure 6: NVFP4 vs MXFP4 comparisons: (a) training-loss difference; (b) validation perplexity across token budgets.*


To achieve the same final training loss as the model trained with NVFP4, the model using MXFP4 required **36% more training tokens**. This suggests that the design of NVFP4 leads to greater sample efficiency.

## Conclusion

NVFP4, when combined with the specified training methodology, enables stable and accurate pretraining of large-scale language models in 4-bit precision. This approach offers significant efficiency gains in terms of computational throughput and memory usage without compromising model performance. Full support for NVFP4 is available in NVIDIA's Transformer Engine.

---

***Source:*** *This guide is a summary of the technical report "[Pretraining Large Language Models with NVFP4](https://arxiv.org/pdf/2509.25149v1)". For complete details, please refer to the original publication.*

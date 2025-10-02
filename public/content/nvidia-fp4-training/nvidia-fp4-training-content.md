---
hero:
  title: "NVIDIA Details 4-Bit LLM Training at Scale with NVFP4"
  subtitle: "Technical Breakdown of Pretraining a 12B Model on 10 Trillion Tokens"
  tags:
    - "⏱️ Technical Deep Dive"
    - "📄 Research Article"
    - "⚙️ LLM Training"
---

[Research Paper](https://arxiv.org/abs/2509.25149) • [Code in Transformer Engine](https://github.com/NVIDIA/TransformerEngine) • [Join open research on FP4 Training](https://github.com/Open-Superintelligence-Lab/fp4-training-research)

> **🚦 Prerequisite:**  
> To get the most out of this article, you should have a basic understanding of **quantization** and **neural network training**.  

As Large Language Models (LLMs) scale, the computational cost of training increases significantly. While 8-bit floating-point (FP8) training is standard for improving efficiency, 4-bit floating-point (FP4) presents the next step in reducing computational overhead.

FP4 training offers substantial gains in computational speed and memory savings, but quantizing to such a low precision introduces challenges in training stability, convergence, and implementation.

In a new technical report, NVIDIA introduces a method for effective FP4 training. They trained a 12-billion-parameter model on 10 trillion tokens, achieving results comparable to a standard FP8 baseline.

![Training Memory Comparison](/content/nvidia-fp4-training/memory-comparison.png)

This article explains NVIDIA's methodology, the NVFP4 format, and the key techniques that enable stable and accurate 4-bit training at scale.

### The NVFP4 Format: A More Precise 4-bit Representation

The core of this method is **NVFP4**, an enhanced 4-bit format that offers improved numerical properties over existing formats like MXFP4. It uses "microscaling," where blocks of numbers share a common scaling factor, but with several key improvements:

1.  **Smaller Block Size**: NVFP4 uses a block size of 16 elements, compared to 32 in MXFP4. This narrows the dynamic range within each block for a more accurate representation.
2.  **More Precise Scaling Factors**: It stores block scale factors in E4M3 format, which provides more mantissa bits for precision than the power-of-two UE8M0 format in MXFP4.
3.  **Two-Level Scaling**: NVFP4 uses a two-level scaling strategy: a fine-grained FP8 scale factor for each block, and a per-tensor FP32 scale factor. This allows NVFP4 to represent the largest value in each block with near-FP8 precision.

These design choices give NVFP4 an advantage in minimizing quantization errors and improving training performance, particularly on NVIDIA's Blackwell GPUs which provide native hardware support for NVFP4.

![FP4 Training Architecture](/content/nvidia-fp4-training/fp4-architecture.png)
*A conceptual overview of the NVFP4 block structure, combining FP4 elements with a more precise FP8 scaling factor.*

### The Methodology: 4 Keys to Stable FP4 Training

The data format alone is not sufficient. NVIDIA developed a training methodology to address the instabilities of low-precision training. Ablation studies showed that each of these components was necessary for the model's convergence over the 10T token horizon.

#### 1. Strategic Mixed Precision
The researchers found that some layers, particularly the final few layers of the network, are more sensitive to quantization and can cause training to diverge if converted to FP4.

**Solution**: Keep a small, strategic portion of the model in higher precision. For the 12B model, the first two and final eight blocks (~16% of linear layers) were kept in BF16. Other sensitive components like embeddings, normalization layers, and optimizer states were also kept in FP32 or BF16.

#### 2. Random Hadamard Transforms (RHT) for Outliers
Outliers—values with large magnitudes—can cause issues in low-bit formats. RHT is a technique that uses an orthogonal rotation to redistribute these outliers into a more manageable distribution.

**Solution**: Apply 16x16 Random Hadamard Transforms to the inputs of the weight gradient (Wgrad) computation. This helps manage outlier values during the backward pass without adding overhead where it's not needed.

#### 3. 2D Scaling for Consistency
A critical issue in training is that tensors are transposed during the backward pass. This can cause scaling to be applied differently in the forward and backward passes, leading to two different quantized representations of the same weight tensor, which violates the chain rule and degrades model accuracy.

**Solution**: Use **two-dimensional (2D) block scaling** for weights. By grouping and scaling weights in 16x16 blocks, the quantized representation remains consistent between the forward and backward passes.

#### 4. Stochastic Rounding for Unbiased Gradients
Standard rounding methods (like "round-to-nearest-even") can introduce systematic bias during quantization, especially in gradients, which can impede convergence.

**Solution**: Use **stochastic rounding** for gradient tensors. This method rounds a value probabilistically to one of its two nearest representable numbers. This ensures that, on average, the quantization is unbiased, which was essential for the 12B model's convergence.

### Results: FP4 Matches FP8 at 10 Trillion Tokens

This methodology was validated by training a **12-billion-parameter hybrid Mamba-Transformer model on 10 trillion tokens**.

The results show:
-   **Training Loss**: The validation loss of the NVFP4 model closely tracked the FP8 baseline throughout the entire 10T token run.
-   **Downstream Performance**: On a wide range of downstream tasks—including MMLU-pro, MATH, and GSM8k—the NVFP4 model achieved accuracy nearly identical to the FP8 model. For instance, it scored 62.58% on MMLU-pro, compared to the FP8 model's 62.62%.

This demonstrates that with this methodology, 4-bit training does not compromise model quality at scale.

### NVFP4 vs. MXFP4
In a direct comparison using an 8B parameter model, NVFP4 was more efficient. The model trained with NVFP4 converged to a better loss than the one trained with MXFP4. To match the final loss of the NVFP4 model, the MXFP4 model required **36% more training tokens**.

### Conclusion and Future Directions
NVIDIA's work provides the first public evidence of successful, sustained 4-bit pretraining of a large language model at a multi-trillion-token scale. The combination of the NVFP4 format and the described training methodology makes large-scale training more efficient without sacrificing accuracy.

This is a significant improvement in making the training of large models more accessible. NVFP4 training is supported in Transformer Engine for Blackwell GPUs.

Future work will focus on quantizing more parts of the model, such as attention and communication paths, and extending the methodology to other architectures like Mixture-of-Experts.

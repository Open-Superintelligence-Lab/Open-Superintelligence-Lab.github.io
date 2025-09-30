# Localhost Development Todos

## Dream-Coder 7B Exploration
- [ ] Explore Dream-Coder 7B - diffusion LLM for code generation
  - [ ] Review GitHub repository: https://github.com/DreamLM/Dream-Coder
  - [ ] Understand diffusion LLM architecture for code
  - [ ] Test code generation capabilities (21.4% pass@1 on LiveCodeBench)
  - [ ] Experiment with flexible generation patterns (sketch-first, left-to-right, iterative)
  - [ ] Try variable-length code infilling with DreamOn-7B variant
  - [ ] Install dependencies: `transformers==4.46.2` and `torch==2.5.1`
  - [ ] Test quickstart example with quicksort algorithm

## NVFP4 LLM Pretraining Research
- [ ] Research NVIDIA's NVFP4 (4-bit floating point) training methodology
  - [ ] Study the paper: https://arxiv.org/pdf/2509.25149
  - [ ] Understand NVFP4 format vs MXFP4 differences
  - [ ] Learn Random Hadamard Transforms (RHT) for outlier bounding
  - [ ] Study two-dimensional quantization scheme
  - [ ] Understand stochastic rounding for unbiased gradients
  - [ ] Research selective high-precision layers approach
  - [ ] Analyze 12B model training on 10T tokens results
  - [ ] Compare MMLU-pro accuracy: 62.58% (NVFP4) vs 62.62% (FP8)
  - [ ] Set up experimental environment for FP4 training
  - [ ] Test NVFP4 implementation with smaller models first
  - [ ] Study TransformerEngine NVFP4 implementation: https://github.com/NVIDIA/TransformerEngine/pull/2177
  - [ ] Explore NVFP4 recipe and PyTorch integration
  - [ ] Test NVFP4 support with fusible operations
  - [ ] Study Random Hadamard Transform (RHT) cast fusion
  - [ ] Understand NVFP4 quantization and dequantization kernels
  - [ ] Test distributed training with NVFP4

## MobileLLM-R1 Sub-Billion Reasoning Research
- [ ] Research Meta's MobileLLM-R1 sub-billion parameter reasoning models
  - [ ] Study the paper: https://arxiv.org/pdf/2509.24945
  - [ ] Understand data curation and resampling techniques (~2T high-quality tokens)
  - [ ] Learn benchmark-free, self-evolving data optimization approach
  - [ ] Study data-model co-evolution strategy for mid-training adaptation
  - [ ] Analyze training recipe: 4.2T tokens from resampled ~2T tokens
  - [ ] Compare performance: AIME 15.5 vs OLMo-2-1.48B (0.6) and SmolLM-2-1.7B (0.3)
  - [ ] Study how 11.7% of Qwen3's 36T tokens achieves comparable performance
  - [ ] Explore complete training recipe and data sources (fully open-sourced)
  - [ ] Test MobileLLM-R1-950M model capabilities
  - [ ] Set up experimental environment for sub-billion reasoning model training

## Notes
- Dream-Coder 7B outperforms other open-source diffusion LLMs
- Features emergent any-order generation that adapts to coding tasks
- Trained exclusively on open-source data
- Supports both base and instruct variants
- NVFP4 enables 2-3x arithmetic performance boost and 50% memory reduction vs FP8
- First successful 4-bit training of billion-parameter models on multi-trillion tokens
- MobileLLM-R1 challenges assumptions about reasoning requiring large models and massive datasets
- Achieves strong reasoning with only ~2T high-quality tokens vs 36T+ for larger models
- Complete open-source training recipe and model checkpoints available

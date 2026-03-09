---
title: "Research Roadmap"
date: "2026-03-03"
description: ""
---

## Table of Contents

- [Our roadmap to AGI / superintelligence.](#our-roadmap-to-agi--superintelligence)
  - [Nature of data](#nature-of-data)
    - [Text / Language](#text--language)
    - [Some questions we are thinking about](#some-questions-we-are-thinking-about)
  - [How to pick research question:](#how-to-pick-research-question)
- [Handpicked questions:](#handpicked-questions)
  - [LLM Research](#llm-research)
  - [Practical Model Projects](#practical-model-projects)
  - [LLM & Foundation Models](#llm--foundation-models)
    - [1. Next-Token Prediction & Objective Functions](#1-next-token-prediction--objective-functions)
    - [2. Architecture & Expressivity (Transformers & Attention)](#2-architecture--expressivity-transformers--attention)
    - [3. Scaling Laws & Emergence](#3-scaling-laws--emergence)
    - [4. In-Context Learning & Working Memory](#4-in-context-learning--working-memory)
    - [5. Representation, Semantics & Grounding](#5-representation-semantics--grounding)
    - [6. Optimization and Quantization](#6-optimization-and-quantization)
  - [JEPA](#jepa)
  - [Diffusion Models](#diffusion-models)
    - [1. Theoretical Foundations](#1-theoretical-foundations)
    - [2. Sampling Efficiency & Fast Generation](#2-sampling-efficiency--fast-generation)
    - [3. Architecture Design](#3-architecture-design)
    - [4. Latent Diffusion & Representation Learning](#4-latent-diffusion--representation-learning)
    - [5. Conditioning & Controllability](#5-conditioning--controllability)
    - [6. Image Editing & Inversion](#6-image-editing--inversion)
    - [7. Video Generation & Temporal Modeling](#7-video-generation--temporal-modeling)
    - [8. 3D Generation](#8-3d-generation)
    - [9. Audio, Music & Speech](#9-audio-music--speech)
    - [10. Scientific & Domain-Specific Applications](#10-scientific--domain-specific-applications)
    - [11. Discrete & Non-Euclidean Diffusion](#11-discrete--non-euclidean-diffusion)
    - [12. Flow Matching & Continuous Normalizing Flows](#12-flow-matching--continuous-normalizing-flows)
    - [13. Training Improvements](#13-training-improvements)
    - [14. Evaluation Metrics & Benchmarking](#14-evaluation-metrics--benchmarking)
    - [15. Personalization & Customization](#15-personalization--customization)
    - [16. Efficiency, Compression & Deployment](#16-efficiency-compression--deployment)
    - [17. Safety & Robustness](#17-safety--robustness)
    - [18. Multi-Modal & Cross-Modal Diffusion](#18-multi-modal--cross-modal-diffusion)
    - [19. Diffusion Models vs. Other Generative Paradigms](#19-diffusion-models-vs-other-generative-paradigms)
    - [20. Diffusion for Decision-Making & Robotics](#20-diffusion-for-decision-making--robotics)
    - [21. Diffusion for Inverse Problems & Restoration](#21-diffusion-for-inverse-problems--restoration)
    - [22. Interpretability & Understanding](#22-interpretability--understanding)
    - [23. Data Efficiency & Few-Shot Generation](#23-data-efficiency--few-shot-generation)
    - [24. Continual & Lifelong Learning](#24-continual--lifelong-learning)
    - [25. Connections to Physics & Thermodynamics](#25-connections-to-physics--thermodynamics)
- [Reinforcement Learning](#reinforcement-learning)
- [Research Questions for Reinforcement Learning](#research-questions-for-reinforcement-learning)
  - [1. Theoretical Foundations](#1-theoretical-foundations-1)
  - [2. Exploration](#2-exploration)
  - [3. Sample Efficiency](#3-sample-efficiency)
  - [4. Offline & Batch Reinforcement Learning](#4-offline--batch-reinforcement-learning)
  - [5. Model-Based Reinforcement Learning](#5-model-based-reinforcement-learning)
  - [6. Hierarchical Reinforcement Learning (HRL)](#6-hierarchical-reinforcement-learning-hrl)
  - [7. Multi-Agent Reinforcement Learning (MARL)](#7-multi-agent-reinforcement-learning-marl)
  - [8. Reward Design & Reward Learning](#8-reward-design--reward-learning)
  - [9. RLHF & Alignment](#9-rlhf--alignment)
  - [10. Policy Optimization & Algorithms](#10-policy-optimization--algorithms)
  - [11. Representation Learning for RL](#11-representation-learning-for-rl)
  - [12. Generalization & Transfer](#12-generalization--transfer)
  - [13. Continual & Lifelong Learning](#13-continual--lifelong-learning)
  - [14. Goal-Conditioned & Multi-Task RL](#14-goal-conditioned--multi-task-rl)
  - [15. Language & Foundation Models for RL](#15-language--foundation-models-for-rl)
  - [16. Safe & Constrained RL](#16-safe--constrained-rl)
  - [17. Robustness & Adversarial RL](#17-robustness--adversarial-rl)
  - [18. Imitation Learning & Learning from Demonstrations](#18-imitation-learning--learning-from-demonstrations)
  - [19. Temporal Abstraction & Options Framework](#19-temporal-abstraction--options-framework)
  - [20. Credit Assignment](#20-credit-assignment)
  - [21. Curriculum Learning & Automatic Environment Design](#21-curriculum-learning--automatic-environment-design)
  - [22. RL for Robotics & Embodied AI](#22-rl-for-robotics--embodied-ai)
  - [23. RL for Games & Complex Decision-Making](#23-rl-for-games--complex-decision-making)
  - [24. RL for Combinatorial Optimization](#24-rl-for-combinatorial-optimization)
  - [25. RL for Science & Engineering](#25-rl-for-science--engineering)
  - [26. RL for NLP & Code Generation](#26-rl-for-nlp--code-generation)
  - [27. Distributional RL](#27-distributional-rl)
  - [28. Inverse RL & Preference Learning](#28-inverse-rl--preference-learning)
  - [29. Offline-to-Online & Hybrid RL](#29-offline-to-online--hybrid-rl)
  - [30. Interpretability & Explainability](#30-interpretability--explainability)
  - [31. Memory & Recurrence in RL](#31-memory--recurrence-in-rl)
  - [32. Causal RL](#32-causal-rl)
  - [33. Evaluation & Reproducibility](#33-evaluation--reproducibility)
  - [34. Real-World RL Challenges](#34-real-world-rl-challenges)
  - [35. Open-Ended & Emergent Learning](#35-open-ended--emergent-learning)
  - [36. Neuroscience-Inspired RL](#36-neuroscience-inspired-rl)


# Our roadmap to AGI / superintelligence.

Main research tracks:

1. What is the structure of data? - You cannot build intelligence on data you do not understand
2. What do models actually learn? - Representations show internal understanding

Auxiliary research tracks:

3. How does understanding emerge? - Scaling and emergence need to be researched a lot deeper
4. What can limits of architectures teach us? - Every architecture has a ceiling - what can we learn from it
5. What is the difference between world models (JEPA, LLMs, Emboddied RL, etc.) - this will help us understand what a world model is.


If you would like to be the first author on any of these topics (a few hours to 6 months, depending on the task you choose and your time availability), please contact me.[X: @VukRosic99] It's important that you approach it seriously and responsibly (with our support) and write a quality blog post / paper at the end.

## Nature of data

### Text / Language

**Core question:** What is the limit of intelligence we can create with text only, and how do we move towards it?

Can we examine the underlying data properties of language by training LLMs on the same data in different ways, and studying the internal representations:

1. Autoregressive LLM
2. Diffusion LLM
3. SSM (Mamba, GDN)

Is there structure in left-to-right generation that doesn't get captured by diffusion, and what can we learn from it?

Do the emergent capabilities come from the data, or from the autoregressive processing?

**Experiments to run:**

1. **The Embedding Matrix** - Every architecture learns a token embedding matrix.
   - Nearest-neighbor structure: for every token, compute its k-nearest neighbors in embedding space. Measure overlap across architectures. High overlap = the data forces these tokens to cluster, regardless of training objective.
   - Analogical structure: do all three learn the same linear directions? (king − man + woman ≈ queen). Test systematically across semantic relations.
   - Embedding isotropy / anisotropy: do all three develop the same geometric pathologies (e.g., the narrow cone problem)?
   - Frequency-geometry correlation: plot token frequency vs. embedding norm. All three will likely show a relationship - but is it the same one?

2. **Per-token loss comparison** - loss for each token for each architecture. Which tokens are equally difficult for different architectures, which ones are not, and what can we learn?

3. **Representation collapse analysis** - token vectors: how many dimensions are near 0, are representations collapsed?

**Related question:** Can we benchmark or improve how quickly a model learns a new skill?

### Some questions we are thinking about

Which questions are best for long term research and how should we answer them and what are the goals of the research?

How do we evaluate which questions are best for long term progress?

Help us review, choose, delete, add, and define questions better.

1 main author, they can invite or interview others to join. 6 months max to solve the question, if it can't be solved in 6 month maybe it's too difficult.

There shouldn't be more than 6 people can working on 1 topic / question, as it might become difficult to manage them.

We will setup baselines and experiments that cover many questions at once to help authors.

## How to pick research question:
1. What are you passionate about and have a deep understanding of?
2. Have you done a literature review in this industry recently?
3. Are you ready to spend months on incremental research? Or years for a fundamental change? (is this how it is? i'm not sure)
- quote by @krik_exe on X

---

## LLM Research

Can we examine underlying data distribution of language by training autoregressive LLM vs diffusion LLM, is there structure in left-to-right generation that doesn't get captured by diffusion, what can we learn from it?

Research applying per-parameter momentum of AdamW to Muon optimizer.

How to make a 1-bit quantization model as close as possible to the performance of a full precision model? 
- BiLLM, issues: only works for LLMs, not for VLMs

## Practical Model Projects

Building some practical models, here are examples below
- We should probably distill bigger models into very small specialized models, possibly include reinforcement learning? Here are ideas below, give your feedback.

**NLP & Information Extraction (Small LLMs)**

1. **Email/Contact Extractor LLM** - Fine-tune a small LLM (e.g., 10M–125M parameters, SmolLM or TinyLlama) to extract structured data (emails, phone numbers, names, addresses) from raw unstructured text. Benchmark against regex, spaCy NER, and GPT-4 API on accuracy, speed, latency, and cost per 1M documents. Deliverable: trained model + benchmark report.

2. **Document Summarizer for Research Papers** - Fine-tune a small model to take arXiv abstracts + introductions and produce structured summaries (problem, method, result, limitation). Compare 125M param fine-tuned model vs. 7B general model vs. API calls. Study what the small model learns vs. what it fails at.

3. **Intent Classifier From Scratch** - Train a small transformer from scratch on customer support / chatbot intent data. Study how quickly the model learns to separate intents vs. how much data is needed. Connects to our core question: *how quickly can a model learn a new skill?*

4. **Structured JSON Generator** - Fine-tune a small LLM to convert natural language descriptions into structured JSON (e.g., "3 bedroom apartment in NYC under $2000" → `{"bedrooms": 3, "city": "NYC", "max_price": 2000}`). Study failure modes - when does structured output break and why?

**Architecture & Representation Experiments**

6. **Attention vs. SSM on Identical Data** - Train a small transformer and a small Mamba model on identical data (e.g., TinyStories or a curated Wikipedia subset). Compare internal representations: CKA similarity, probing accuracy on syntactic/semantic tasks, per-token loss distributions. This directly feeds our core research on data structure.

7. **Quantization Cliff Finder** - Take a pre-trained small LLM and progressively quantize it (FP16 → INT8 → INT4 → INT2 → 1-bit). For each level, measure not just perplexity but *which types of knowledge degrade first* (factual recall? syntax? reasoning? rare tokens?). Produces a "knowledge fragility map."

8. **Embedding Geometry Comparator** - Train 3–5 small models with different architectures (transformer, SSM, RWKV) on the same data. Analyze embedding spaces: isotropy, nearest-neighbor overlap, frequency-norm correlation, analogical structure. Directly tests whether data structure is architecture-invariant.

9. Reward hacking detector - Is there reward hacking in reasoning LLM training, how to measure and detect it?

**Diffusion & Generation**

9. **Tiny Diffusion LLM** - Implement a small diffusion language model (using MDLM or similar) and train on a small corpus. Compare token-level metrics against an autoregressive model of the same size on the same data. Which tokens does diffusion handle better/worse?

10. **1-Step vs. N-Step Diffusion Image Quality** - Train a small latent diffusion model and distill it to 1-step, 2-step, 4-step, 8-step variants. Systematically measure what visual features degrade at each step reduction (textures? global structure? fine details? faces?).

## LLM & Foundation Models

A collection of fundamental, first-principles questions regarding Large Language Models and their underlying architectures.

### 1. Next-Token Prediction & Objective Functions
- Is the next-token prediction objective sufficient to learn an accurate causal world model, or does it merely approximate statistical correlations on the surface of the data?
- What are the fundamental information-theoretic limits of learning complex reasoning processes via discrete, autoregressive token prediction?
- Under what formal conditions does exact next-token prediction mathematically guarantee the learning of the underlying generative process (the "true" data manifold)?
- Are LLMs optimizing for a Minimum Description Length (MDL) representation of the training data, and how does this relate to their generalization capabilities versus pure memorization?
- Can alternative objectives (e.g., energy-based models, continuous latent prediction, contrastive learning) bypass the theoretical limits of autoregression in capturing long-tail knowledge?

### 2. Architecture & Expressivity (Transformers & Attention)
- What is the precise mathematical role of the multi-layer perceptron (MLP) layers versus self-attention layers concerning knowledge retrieval, storage, and information routing?
- Are there fundamental limitations to self-attention's ability to model hierarchical or tree-structured data mathematically, and do current mechanisms simply approximate these structures?
- From a dynamical systems perspective, can we map the forward pass of a deep LLM to a specific class of differential equations, and what does this imply about their reasoning limits?
- What are the strict computational complexity bounds of transformer layers, and which formal languages or logic systems are fundamentally unlearnable by standard attention mechanisms?
- Does replacing standard softmax attention with alternative mechanisms (e.g., linear attention, state-space models like Mamba) structurally change the model's hypothesis space or just make it more computationally efficient?

### 3. Scaling Laws & Emergence
- Are "emergent capabilities" truly discontinuous phase transitions governed by specific parameter/compute thresholds, or are they an artifact of the evaluation metrics used (e.g., discrete accuracy vs. cross-entropy)?
- What physical or information-theoretic principles dictate the precise slope and intercept of empirical neural scaling laws?
- Is there a fundamental threshold beyond which adding more data injects contradictory noise rather than useful signal, representing a strict asymptote on scaling?
- Can we theoretically predict the exact point at which an LLM shifts from "memorizing" individual data points to "generalizing" structural patterns based on model capacity and dataset entropy?

### 4. In-Context Learning & Working Memory
- Is in-context learning fundamentally equivalent to implicit gradient descent on the prompt (meta-learning), or does it rely on a separate mechanism of dynamic circuitry and information routing within the weights?
- What are the theoretical bounds on the working memory capacity of self-attention (context window), and how does attention entropy degrade as context length scales to infinity?
- How do transformer heads construct momentary variable bindings or "pointers" in activation space, and are there fundamental computational limitations to this compared to classical von Neumann memory architectures?

### 5. Representation, Semantics & Grounding
- How is abstract semantic meaning topologically structured in high-dimensional embedding spaces, and is this geometry universal across different models, initializations, and modalities?
- Can language models ever achieve true symbol grounding purely from ungrounded text (the "text-only" hypothesis), or does theoretical physics and cognitive science suggest a formal barrier necessitating embodied interaction?
- Are the concepts learned by LLMs compositional by default? How can we mathematically measure the degree of compositionality and disentanglement in their internal representations?
- Do LLMs learn a single, coherent representation of truth, or do they construct superpositioned simulation models that branch depending on the syntactic framing of the prompt?

### 6. Optimization and Quantization
- Research applying per-parameter momentum of AdamW to Muon optimizer.
- How to make a 1-bit quantization model as close as possible to the performance of a full precision model? 
- For BiLLM, what are the theoretical obstacles that limit its effectiveness to LLMs and prevent generalization to VLMs?
- What is the theoretical limit of weight and activation quantization before structural knowledge is irreversibly destroyed, and does this limit depend on the entropy of the training data?

## JEPA

JEPA Research Questions & Topics

1. Theoretical Foundations

- What is the precise theoretical relationship between JEPA's energy-based framework and other self-supervised paradigms (contrastive learning, masked autoencoders, variational autoencoders)? Can JEPA be formalized as a special case of a broader family?
- What are the information-theoretic properties of JEPA's latent space? How much task-relevant information is preserved vs. discarded compared to pixel-level reconstruction objectives?
- Why does predicting in representation space rather than pixel/token space lead to more semantically meaningful representations? Can this be formalized via rate-distortion theory or the information bottleneck principle?
- What are the conditions under which JEPA's energy function provably avoids representation collapse without contrastive negatives? How do different collapse prevention mechanisms (VICReg, variance-covariance regularization, architectural asymmetry, EMA targets) compare theoretically?
- Can we derive generalization bounds for downstream tasks as a function of the quality of JEPA's learned energy landscape?
- How does JEPA relate to predictive coding theories in neuroscience, and can neuroscience insights (e.g., hierarchical predictive coding, precision weighting) improve JEPA architectures?
- What is the formal relationship between JEPA and optimal transport / Wasserstein distances in representation space?

2. Architecture Design

- What is the optimal encoder architecture for JEPA (ViT, CNN, hybrid, state-space models)? How does this vary across modalities (vision, audio, video, text, multimodal)?
- How should the predictor network be designed? What capacity, depth, and architectural form (MLP, Transformer, hypernetwork) is optimal, and how does predictor capacity affect the quality of learned representations?
- Is there an optimal ratio of predictor capacity to encoder capacity? What happens when the predictor is too powerful (bypasses encoder) or too weak (underfits)?
- How should the target encoder (EMA teacher) be designed? Is EMA the best choice, or can alternatives (stop-gradient, periodic sync, distillation, Polyak averaging with adaptive rates) perform better?
- Can hierarchical JEPA (H-JEPA) be effectively implemented with multiple levels of abstraction, and what is the best way to structure inter-level predictions?
- How should masking strategies be designed - what mask ratio, mask shape (block, random, semantic-aware, attention-guided), and mask scheduling work best?
- Can mixture-of-experts (MoE) predictors specialize in different types of predictions (spatial, temporal, semantic) and improve representation quality?
- What is the role of the context encoder vs. target encoder asymmetry, and can symmetric architectures work with the right training objective?

3. Training Objectives & Collapse Prevention

- What is the most effective training objective for JEPA - L2 in latent space, cosine similarity, contrastive in latent space, or energy-based objectives? How do these compare at scale?
- How do different collapse prevention strategies (VICReg regularization, Barlow Twins-style decorrelation, batch normalization, predictor architecture constraints, EMA targets, feature whitening) compare in terms of representation quality and training stability?
- Can learned energy functions replace hand-designed objectives? Can the energy landscape be shaped via meta-learning?
- Does multi-scale prediction (predicting representations at multiple spatial/temporal resolutions simultaneously) improve representation quality?
- Can curriculum strategies for masking difficulty (easy → hard predictions) improve training efficiency and final representation quality?
- What loss landscape properties (smoothness, number of minima, saddle points) characterize well-trained vs. collapsed JEPA models?
- Can adversarial or game-theoretic formulations prevent collapse more robustly than current regularization-based methods?

4. Masking & Prediction Strategies

- What is the optimal masking strategy for vision (random patches, block masking, semantic masking, object-aware masking) and how does it differ from MAE-style masking?
- How does masking ratio affect the trade-off between learning low-level features vs. high-level semantic features?
- For video JEPA (V-JEPA), what temporal masking strategies work best - masking future frames, random temporal segments, or causal masking?
- How does the spatial/temporal distance between context and target affect what the model learns (local texture vs. global semantics vs. temporal dynamics)?
- Can JEPA be extended to predict across modalities (e.g., predict audio representation from video context)?

5. Video & Temporal Understanding (V-JEPA)

- How should temporal abstraction be handled in V-JEPA - frame-level prediction, clip-level prediction, or hierarchical multi-scale temporal prediction?
- Can V-JEPA learn meaningful temporal abstractions (actions, events, causal sequences) without explicit supervision?
- How does V-JEPA compare to video MAE, VideoGPT, and contrastive video methods (e.g., VideoBYOL) on temporal reasoning benchmarks?
- Can V-JEPA learn predictive models of physical dynamics (intuitive physics) purely from video observation?
- What is the optimal temporal context window - how much past context is needed to predict future representations accurately?
- Can V-JEPA be extended to learn action-conditioned world models for robotics and embodied AI?
- How should V-JEPA handle variable frame rates, long videos, and temporally sparse events?
- Can V-JEPA representations support zero-shot or few-shot action recognition, temporal localization, and video question answering?

6. World Models & Planning

- Can JEPA serve as the backbone for a learned world model, as proposed in LeCun's autonomous machine intelligence framework? What are the practical bottlenecks?
- How can JEPA's latent predictions be used for planning (e.g., model-predictive control in latent space)? What planning algorithms work best with JEPA-learned dynamics?
- Can JEPA learn to predict the effects of actions in latent space (action-conditioned JEPA), and how does this compare to model-based RL world models (Dreamer, IRIS, TD-MPC)?
- How should uncertainty be represented in JEPA's predictions - deterministic prediction, stochastic latent variables, energy-based multiple hypotheses, or ensemble disagreement?
- Can hierarchical JEPA (H-JEPA) learn multi-level world models where higher levels predict at longer time horizons with more abstract representations?
- How can JEPA-based world models handle multimodal futures (multiple plausible outcomes from the same context)?
- Can JEPA world models generalize to out-of-distribution states and novel compositions of known elements?
- What is the relationship between JEPA world models and the free energy principle / active inference framework?

7. Multimodal & Cross-Modal JEPA

- Can JEPA be extended to jointly learn representations across vision, language, audio, and other modalities? What is the best architecture for multimodal JEPA?
- How should cross-modal prediction be structured - predict one modality's representation from another, or predict a shared representation?
- Can JEPA provide a more principled alternative to CLIP-style contrastive multimodal learning by predicting in latent space rather than contrasting?
- How does multimodal JEPA handle missing modalities at inference time?
- Can JEPA learn a shared semantic space where representations from different modalities are naturally aligned without explicit alignment objectives?
- How should the predictor be designed for cross-modal prediction (e.g., audio from video) where the mapping is inherently uncertain/many-to-many?

8. Representation Quality, Scaling & Evaluation

- How do JEPA representations compare to contrastive (SimCLR, DINO, DINOv2), reconstructive (MAE, BEiT), and generative (diffusion, VAE) representations on diverse downstream tasks?
- Do JEPA representations capture compositional structure (objects, attributes, relations), and how can this be measured?
- What do JEPA representations encode that other methods miss, and vice versa? Can representation probing (e.g., probing for depth, surface normals, object parts) reveal qualitative differences?
- How does representation quality scale with model size, data size, and compute - are there scaling laws for JEPA?
- Can JEPA be made more data-efficient - how much unlabeled data is needed to learn good representations for a given domain?
- Can progressive training (gradually increasing resolution, model size, or masking difficulty) improve JEPA's efficiency?

10. Applications & Downstream Impact

- Can JEPA representations improve performance in robotics and embodied AI (manipulation, navigation, locomotion) compared to current self-supervised methods?
- How effective are JEPA representations for medical imaging, satellite imagery, scientific data, and other domains with limited labeled data?
- Can JEPA-based world models be used for safe RL by enabling planning and look-ahead in latent space before acting in the real world?
- Can JEPA representations serve as a foundation for continual/lifelong learning where the model must adapt to new data without forgetting?
- How can JEPA be combined with LLMs - can JEPA provide the perceptual backbone for multimodal large language models?
- Can JEPA representations improve retrieval, search, and recommendation systems?
- Can JEPA be applied to scientific domains (molecular dynamics, climate modeling, protein structure) where prediction in latent space may be more natural than in observation space?

11. Comparison with Alternative Paradigms

- JEPA vs. MAE: When does predicting in representation space outperform predicting in pixel space, and vice versa? Can hybrid approaches combine the best of both?
- JEPA vs. Contrastive Learning (SimCLR, DINO): How do invariance properties differ? Does JEPA preserve more information by avoiding explicit invariance enforcement?
- JEPA vs. Generative Models (VAE, Diffusion): Can JEPA learn representations as good as or better than generative models without the cost of learning to generate?
- JEPA vs. Autoregressive Models (GPT-style): Is the JEPA prediction paradigm fundamentally more efficient than next-token prediction for learning world models?
- Can JEPA and contrastive objectives be combined (e.g., JEPA loss + contrastive loss) to get the benefits of both?
- How does JEPA compare to BYOL/SimSiam (which also use prediction + EMA but in augmentation space rather than masking space)?

12. Open Fundamental Questions

- Is representation-space prediction sufficient for general intelligence, or are there fundamental limitations to the JEPA framework?
- Can JEPA learn abstract reasoning, causal understanding, and common sense, or are additional mechanisms needed?
- How should JEPA handle inherent stochasticity in the world - is a single deterministic prediction in latent space sufficient, or must the model represent distributions over latent predictions?
- Can JEPA be extended to an active learning / active inference setting where the agent chooses what to observe and predict?
- What is the minimal set of inductive biases needed for JEPA to learn useful representations from raw sensory data?
- Can JEPA eventually replace autoregressive language modeling for text, or is it fundamentally better suited for continuous modalities (vision, audio)?
- How close is the current JEPA paradigm to LeCun's full vision of autonomous machine intelligence, and what are the missing pieces?

## Diffusion Models

## 1. Theoretical Foundations

- What is the precise relationship between score-based generative models, stochastic differential equations (SDEs), and denoising diffusion probabilistic models (DDPMs), and can a more unified theoretical framework be developed?
- Can tighter variational bounds or alternative training objectives (beyond ELBO) improve diffusion model learning and generation quality?
- What are the theoretical convergence guarantees of diffusion models under realistic assumptions (finite data, finite network capacity, limited diffusion steps)?
- How does the choice of noise schedule (linear, cosine, sigmoid, learned) affect the geometry of the learned probability path, and is there a provably optimal schedule?
- Can information-theoretic perspectives (e.g., mutual information, rate-distortion) provide new insights into the forward and reverse diffusion processes?
- What is the relationship between the manifold hypothesis and diffusion model performance - how do diffusion models handle data lying on low-dimensional manifolds embedded in high-dimensional space?
- Can optimal transport theory provide better formulations of the diffusion process (e.g., flow matching, rectified flows) that are theoretically superior to standard Gaussian diffusion?
- How does the dimensionality of the data space affect the sample complexity and convergence rate of diffusion models?

---

## 2. Sampling Efficiency & Fast Generation

- How can the number of sampling steps be reduced to single-digit or even one-step generation without significant quality degradation?
- What is the theoretical minimum number of function evaluations (NFE) required for high-fidelity generation, and how close are current methods?
- How do distillation-based acceleration methods (progressive distillation, consistency models, consistency training) compare to solver-based acceleration (DDIM, DPM-Solver, exponential integrators)?
- Can adaptive step-size ODE/SDE solvers be designed specifically for the structure of diffusion model reverse processes?
- How does the trade-off between stochastic (SDE) and deterministic (ODE/probability flow) sampling affect quality, diversity, and speed?
- Can caching, token merging, or feature reuse across denoising steps meaningfully reduce per-step computational cost?
- How effective are parallel/asynchronous denoising strategies (e.g., Picard iterations, ParaDiGMS) for reducing wall-clock generation time?
- Can early stopping criteria be developed to terminate sampling when generation quality is "good enough"?

---

## 3. Architecture Design

- What are the optimal architectural choices (U-Net vs. Transformer/DiT vs. hybrid) for diffusion model backbones at different scales and modalities?
- How does scaling the denoising network (depth, width, attention heads) affect generation quality, and are there scaling laws for diffusion models?
- Can state-space models (Mamba, S4) or linear attention variants serve as efficient alternatives to full attention in diffusion architectures?
- How should the timestep conditioning be injected into the network (AdaLN, cross-attention, FiLM, concatenation), and does it matter at scale?
- What role does the architecture play in enabling few-step generation vs. many-step generation?
- Can mixture-of-experts (MoE) architectures be effective for diffusion models, with different experts specializing in different noise levels?
- How do architectural choices affect the model's ability to learn fine-grained details vs. global structure?
- What is the impact of normalization strategies (LayerNorm, GroupNorm, RMSNorm) on diffusion model training stability and generation quality?

---

## 4. Latent Diffusion & Representation Learning

- What properties should an ideal latent space have for latent diffusion models (LDMs), and how should autoencoders be designed/trained for this purpose?
- How does the choice of autoencoder (VAE, VQ-VAE, VQ-GAN, AE) and its bottleneck dimensionality affect downstream diffusion quality?
- Can diffusion models be used as representation learners, and how do the learned representations compare to those from contrastive learning, MAE, or DINO?
- Is there an optimal latent space dimensionality as a function of data complexity, and how does compression ratio affect generation fidelity?
- Can hierarchical latent spaces (multi-scale latent diffusion) improve both quality and efficiency?
- How do tokenized/discrete latent spaces (as in masked diffusion or discrete diffusion) compare to continuous latent spaces?
- Can the autoencoder and diffusion model be trained jointly end-to-end, and does this improve performance?

---

## 5. Conditioning & Controllability

- How can text-to-image alignment be improved beyond CLIP-based conditioning (e.g., T5, LLM-based encoders, multimodal embeddings)?
- What are the most effective methods for spatial control (layout, keypoints, depth, segmentation maps, edges) in conditional generation?
- How can classifier-free guidance (CFG) be improved or replaced with more principled alternatives that avoid the quality-diversity trade-off?
- Can negative prompting and guidance be formalized and improved theoretically?
- How can fine-grained attribute control (e.g., changing a single attribute while preserving others) be achieved without retraining?
- What are the best approaches for compositional generation (multiple objects with specified relationships, counts, and attributes)?
- How can diffusion models better follow complex, multi-constraint prompts (e.g., "a red cube to the left of a blue sphere on a wooden table")?
- Can training-free control mechanisms (e.g., attention manipulation, energy-based guidance, universal guidance) match the quality of trained adapters (ControlNet, T2I-Adapter, IP-Adapter)?
- How effective are reward/feedback-based fine-tuning methods (RLHF, DPO, ReFL) for aligning diffusion models with human preferences?

---

## 6. Image Editing & Inversion

- How can diffusion model inversion (mapping real images to latent/noise space) be made more accurate and consistent?
- What are the theoretical limits of editing fidelity when using DDIM inversion, null-text inversion, or encoder-based inversion?
- How can text-guided image editing be performed while preserving unedited regions with pixel-level fidelity?
- Can attention-based editing methods (Prompt-to-Prompt, MasaCtrl, PnP) be made more robust and generalizable?
- How should editing operations be disentangled (style vs. content vs. structure vs. color)?
- Can diffusion models support interactive, iterative editing workflows with undo/redo capabilities?
- How can instruction-based editing (e.g., InstructPix2Pix) be improved to handle ambiguous or complex instructions?

---

## 7. Video Generation & Temporal Modeling

- How should temporal consistency be enforced in video diffusion models - through 3D attention, temporal layers, autoregressive conditioning, or other mechanisms?
- What are the best strategies for long-video generation beyond the training window (sliding window, hierarchical generation, autoregressive extension)?
- How can motion quality and physical plausibility be improved in video diffusion models?
- Can video diffusion models learn meaningful temporal dynamics that generalize to unseen motions or physics?
- How should video diffusion models handle variable frame rates, resolutions, and aspect ratios?
- What is the role of optical flow, depth, or other structural priors in improving video generation?
- How can video diffusion models be efficiently fine-tuned for specific domains (e.g., driving, robotics, medical imaging)?
- Can video diffusion models serve as world models for planning and decision-making in embodied AI?

---

## 8. 3D Generation

- How can diffusion models generate high-quality 3D assets (meshes, NeRFs, Gaussian splats, point clouds) directly or via 2D lifting?
- What are the trade-offs between multi-view diffusion, SDS-based optimization (DreamFusion), and native 3D diffusion?
- How can multi-view consistency be guaranteed in 3D generation pipelines?
- Can diffusion models generate 3D objects with physically accurate materials, lighting, and texture?
- How should diffusion models handle the diversity of 3D representations (implicit fields, meshes, voxels, point clouds, Gaussian splats)?
- Can 4D generation (dynamic 3D scenes) be achieved with diffusion models, and what are the key challenges?
- How can 3D diffusion models be conditioned on partial observations (single image, text, sketch, partial point cloud)?

---

## 9. Audio, Music & Speech

- How do diffusion models for audio/speech compare to autoregressive and flow-based alternatives in quality, speed, and controllability?
- What are the best spectrogram representations and vocoders for diffusion-based speech synthesis?
- Can diffusion models generate long-form, structurally coherent music (minutes-long compositions)?
- How can diffusion models handle multi-speaker, multi-language, and emotional speech synthesis?
- What are the best approaches for audio inpainting, source separation, and enhancement using diffusion models?
- Can text-to-audio diffusion models capture fine-grained temporal and spectral details of complex soundscapes?

---

## 10. Scientific & Domain-Specific Applications

- How can diffusion models be applied to molecular generation (drug design, materials science) with domain-specific constraints (valency, stability, synthesizability)?
- Can diffusion models improve protein structure prediction, protein design, or antibody generation?
- How effective are diffusion models for weather/climate forecasting compared to traditional numerical methods?
- Can diffusion models be used for medical image synthesis, reconstruction (MRI, CT), and augmentation while maintaining clinical validity?
- How should equivariant diffusion models be designed for data with known symmetries (SE(3), SO(3), permutation)?
- Can diffusion models solve inverse problems in physics (deblurring, super-resolution, compressed sensing, inpainting) with theoretical guarantees?
- How can diffusion models be applied to time-series generation, forecasting, and imputation?
- Can diffusion models generate realistic and useful synthetic tabular data?

---

## 11. Discrete & Non-Euclidean Diffusion

- How should the forward corruption process be defined for discrete data (text, graphs, code, categorical data)?
- Can discrete diffusion models (D3PM, Masked Diffusion, MDLM) match or surpass autoregressive models for text generation?
- How do absorbing-state, uniform, and structured transition matrices compare for discrete diffusion?
- Can diffusion models be effectively defined on non-Euclidean domains (graphs, manifolds, Riemannian spaces, hyperbolic spaces)?
- What are the challenges and solutions for diffusion on combinatorial structures (permutations, trees, sets)?
- How does discrete diffusion scale with vocabulary size and sequence length?

---

## 12. Flow Matching & Continuous Normalizing Flows

- How do flow matching methods (conditional flow matching, rectified flows, stochastic interpolants) compare to score-based diffusion in theory and practice?
- Can straight-line probability paths (as in rectified flows) provide inherently faster sampling without distillation?
- What is the optimal transport cost of different probability path choices, and how does it relate to generation quality?
- Can iterative reflow/rectification procedures provably converge to optimal transport maps?
- How do flow matching methods scale to high-resolution image and video generation?

---

## 13. Training Improvements

- How do different loss weighting strategies (uniform, SNR-based, min-SNR, P2 weighting) affect generation quality across different noise levels?
- What is the effect of training data quality, diversity, and scale on diffusion model performance, and are there scaling laws?
- Can curriculum learning (e.g., training on progressively harder noise levels or resolutions) improve diffusion model training?
- How does the choice of noise prediction target (ε-prediction, x₀-prediction, v-prediction, score prediction) affect training dynamics and generation quality?
- What are the best practices for mixed-precision training, gradient accumulation, and EMA scheduling for large-scale diffusion models?
- Can self-supervised or unsupervised pre-training improve diffusion model performance in low-data regimes?
- How does data augmentation interact with diffusion model training - does it help or can it introduce artifacts?
- What is the role of batch size, learning rate scheduling, and optimizer choice (Adam, AdamW, Adafactor, Lion) in diffusion model training?

---

## 14. Evaluation Metrics & Benchmarking

- Are current metrics (FID, IS, CLIP score, FVD) adequate for evaluating diffusion models, or do we need new metrics?
- How can compositional correctness, text-image alignment, and semantic accuracy be reliably measured?
- How can evaluation protocols account for memorization vs. genuine generalization?
- How should video and 3D generation quality be evaluated systematically?

---

## 15. Personalization & Customization

- How can diffusion models be efficiently personalized to generate specific subjects/styles with minimal examples (few-shot personalization)?
- What are the trade-offs between fine-tuning-based (DreamBooth, Textual Inversion), adapter-based (LoRA, IP-Adapter), and encoder-based personalization?
- Can personalization be achieved without fine-tuning at all (zero-shot personalization via in-context learning)?
- How can overfitting and mode collapse be prevented during few-shot personalization?

---

## 16. Efficiency, Compression & Deployment

- How can diffusion models be compressed (quantization, pruning, distillation, NAS) for edge deployment?
- What is the impact of weight quantization (INT8, INT4, mixed precision) on diffusion model generation quality?
- Can token pruning/merging in DiT architectures reduce computation without affecting quality?
- How do knowledge distillation methods (progressive, consistency, adversarial) compare for creating small, fast student diffusion models?
- What are the best serving strategies for diffusion models in production (batching, caching, streaming)?
- Can diffusion models be made efficient enough for real-time interactive applications on consumer hardware?

---

## 17. Safety & Robustness

- What are effective methods for concept erasure (removing specific concepts from a trained model without retraining from scratch)?
- How vulnerable are diffusion models to adversarial attacks (adversarial prompts, backdoor attacks, membership inference)?
- Can watermarking or provenance tracking be embedded in diffusion model outputs reliably and imperceptibly?
- How do diffusion models memorize and potentially reproduce training data, and how can this be detected and mitigated?
- Can differential privacy be effectively applied to diffusion model training without destroying generation quality?

---

## 18. Multi-Modal & Cross-Modal Diffusion

- How should diffusion models be designed for joint generation across modalities (text + image + audio + video)?
- Can a single unified diffusion model handle multiple modalities, or are modality-specific components necessary?
- How can cross-modal consistency be enforced (e.g., generated audio matches generated video)?
- What are the best conditioning strategies for cross-modal generation (e.g., image-to-audio, audio-to-video)?
- Can diffusion models perform cross-modal retrieval or translation as effectively as discriminative models?
- How should heterogeneous data types (continuous images, discrete text, sequential audio) coexist in a single diffusion framework?

---

## 19. Diffusion Models vs. Other Generative Paradigms

- How do diffusion models compare to GANs, VAEs, normalizing flows, and autoregressive models in terms of quality, diversity, speed, and training stability?
- Can the strengths of different paradigms be combined (e.g., GAN-diffusion hybrids, autoregressive + diffusion, VAE + diffusion)?
- Are there generation tasks where diffusion models are fundamentally inferior to other approaches?
- How do masked generative models (MaskGIT, MUSE) compare to continuous diffusion for image generation?
- Can diffusion models match the speed of GANs while maintaining their diversity advantage?
- Will autoregressive visual generation (e.g., next-token prediction over image tokens) surpass diffusion models?

---

## 20. Diffusion for Decision-Making & Robotics

- How effective are diffusion models as policy representations in offline reinforcement learning (Diffuser, Decision Diffuser)?
- Can diffusion-based planners generate diverse, multi-modal action trajectories for robot manipulation?
- How should diffusion policies handle real-time control requirements given slow sampling?
- Can diffusion models serve as world models for model-based RL?
- How can diffusion models incorporate constraints (safety, feasibility, physical limits) in planning?
- Can goal-conditioned and language-conditioned diffusion policies generalize to unseen tasks?
- How do diffusion policies compare to autoregressive, flow-based, and energy-based alternatives for action generation?

---

## 21. Diffusion for Inverse Problems & Restoration

- Can diffusion models provide principled posterior sampling for linear and nonlinear inverse problems?
- How do diffusion-based methods (DPS, DDRM, ΠGDM, RED-diff) compare to classical optimization for image restoration (deblurring, inpainting, super-resolution, compressed sensing)?
- Can theoretical guarantees (e.g., posterior consistency) be provided for diffusion-based inverse problem solvers?
- How can measurement-conditioned generation be performed without retraining the diffusion model?
- Can diffusion-based inverse solvers handle unknown or partially known forward operators?
- How do these methods scale to high-dimensional inverse problems in medical imaging, remote sensing, or scientific computing?

---

## 22. Interpretability & Understanding

- What do diffusion models learn at different noise levels - is there a coarse-to-fine hierarchy, and how does it relate to human perception?
- Can the internal representations of diffusion models (attention maps, intermediate features) be interpreted meaningfully?
- How do diffusion models represent and compose concepts internally?
- Can mechanistic interpretability tools be applied to understand failure modes (e.g., wrong counting, incorrect spatial relationships)?
- What is the role of memorization vs. generalization in diffusion model generation?
- Can the score function or denoising function be analyzed to understand the learned data manifold?

---

## 23. Data Efficiency & Few-Shot Generation

- How can diffusion models be trained effectively with limited data (hundreds or thousands of images)?
- What regularization techniques, augmentation strategies, or transfer learning approaches are most effective in low-data regimes?
- Can pre-trained diffusion models be adapted to novel domains with minimal data while avoiding overfitting?
- How do synthetic data augmentation and data-efficient training interact in diffusion models?
- Can meta-learning frameworks improve few-shot adaptation of diffusion models to new classes or domains?

---

## 24. Continual & Lifelong Learning

- How can diffusion models be updated with new data or concepts without catastrophic forgetting?
- Can modular or compositional approaches (e.g., adding new LoRA modules) enable continual learning for diffusion models?
- How should new concepts be integrated while preserving existing generation capabilities?
- Can diffusion models perform open-world generation, adapting to novel concepts encountered after training?

---

## 25. Connections to Physics & Thermodynamics

- How does the diffusion process relate to non-equilibrium thermodynamics, and can this connection inspire better algorithms?
- Can insights from statistical mechanics (Langevin dynamics, Fokker-Planck equations) improve sampling or training?
- Is there a meaningful connection between the entropy production in the forward process and generation quality?
- Can physical simulation techniques (molecular dynamics, Monte Carlo methods) be adapted to improve diffusion model sampling?
- How do diffusion models relate to renormalization group flows in physics?

---

This collection spans foundational theory, algorithmic innovations, architectural design, applications across domains, and societal implications - providing a broad landscape for impactful diffusion model research.

# Reinforcement Learning



# Research Questions for Reinforcement Learning

A comprehensive collection of research directions organized by thematic area.

---

## 1. Theoretical Foundations

- What are the fundamental sample complexity lower bounds for RL in various settings (tabular, linear, general function approximation), and how close are current algorithms?
- Can we develop a unified theoretical framework that connects value-based, policy-gradient, and model-based RL under common assumptions?
- What are the necessary and sufficient conditions for the convergence of deep RL algorithms (DQN, PPO, SAC) in the function approximation setting?
- How does the deadly triad (function approximation + bootstrapping + off-policy learning) manifest in practice, and can it be provably resolved?
- What is the fundamental relationship between exploration complexity and the structure of the MDP (e.g., diameter, effective horizon, branching factor)?
- Can information-theoretic tools provide tighter regret bounds for RL in structured environments?
- How does partial observability (POMDPs) fundamentally change the hardness of learning compared to fully observable MDPs?
- What are the theoretical foundations of reward shaping - when does it provably help, and when can it mislead learning?
- Can the PAC-MDP and regret frameworks be unified into a single performance measure for RL?
- How does non-stationarity in the environment affect the theoretical guarantees of RL algorithms?

---

## 2. Exploration

- How can exploration be efficiently conducted in environments with sparse, deceptive, or delayed rewards?
- What are the relative merits of count-based exploration, curiosity-driven exploration (ICM, RND), information-gain exploration, and optimism-based exploration?
- Can intrinsic motivation methods (curiosity, empowerment, surprise) scale to complex, high-dimensional environments without the "noisy TV" problem?
- How should exploration and exploitation be balanced in non-stationary or continually changing environments?
- Can hierarchical exploration strategies (exploring at multiple levels of temporal abstraction) provably improve exploration efficiency?
- How effective are ensemble-based uncertainty estimation methods (bootstrapped DQN, ensemble disagreement) for driving exploration?
- Can directed exploration methods (e.g., Go-Explore, first-return-then-explore) be generalized beyond their original domains?
- How should exploration strategies differ in single-agent vs. multi-agent settings?
- Can exploration bonuses be automatically calibrated rather than requiring manual hyperparameter tuning?
- What is the role of state abstraction and representation learning in making exploration tractable in large state spaces?

---

## 3. Sample Efficiency

- What are the most effective methods for improving sample efficiency in model-free deep RL?
- How do replay ratio, update-to-data (UTD) ratio, and data reuse strategies affect learning efficiency and stability?
- Can data augmentation techniques (image transformations, state perturbations) provably improve sample efficiency, and when do they fail?
- How do model-based methods (Dreamer, MBPO, TD-MPC) compare to model-free methods in sample efficiency across different domains?
- Can pre-training on diverse offline data followed by online fine-tuning achieve the best of both worlds in sample efficiency?
- How does the choice of network architecture (CNN, Transformer, MLP, state-space model) affect sample efficiency?
- What is the role of auxiliary tasks and self-supervised objectives in improving representation quality and sample efficiency?
- Can foundation models or large pre-trained representations eliminate the need for extensive environment interaction?
- How does distributional RL (C51, QR-DQN, IQN) improve sample efficiency, and why?
- What is the minimum amount of interaction data needed to learn a near-optimal policy as a function of environment complexity?

---

## 4. Offline & Batch Reinforcement Learning

- How can offline RL algorithms avoid distributional shift and extrapolation error without being overly conservative?
- What is the optimal trade-off between policy constraint strength and policy improvement in offline RL (CQL, IQL, TD3+BC, AWAC)?
- Can offline RL be made to work reliably with heterogeneous, multi-source, multi-quality datasets?
- How should the quality and coverage of offline datasets be characterized and measured?
- Can pessimism-based methods (lower confidence bound) be made practical and scalable?
- How do diffusion-based and sequence-modeling approaches (Decision Transformer, Trajectory Transformer) compare to traditional Bellman-based offline RL?
- What are the best strategies for offline-to-online fine-tuning, and how can catastrophic forgetting of offline knowledge be prevented?
- Can offline RL benefit from data augmentation, and what augmentations preserve the validity of the behavioral distribution?
- How should offline RL handle non-Markovian or partially observable data?
- Can offline RL methods be applied reliably in safety-critical domains (healthcare, autonomous driving) with guarantees?

---

## 5. Model-Based Reinforcement Learning

- How accurate must a learned world model be for model-based RL to outperform model-free methods?
- What are the best architectures for world models (deterministic, stochastic, autoregressive, diffusion-based, transformer-based)?
- How can model uncertainty and epistemic uncertainty be properly quantified and used for decision-making?
- Can world models generalize beyond the training distribution, and how should out-of-distribution model predictions be handled?
- How should planning be performed with learned models - Dyna-style, shooting methods, model-predictive control (MPC), or tree search?
- What is the optimal balance between model learning and policy learning in terms of computational and data budget allocation?
- Can world models learn abstract, task-relevant representations rather than pixel-level reconstruction?
- How do latent-space world models (Dreamer, IRIS, TD-MPC) compare to observation-space models?
- Can object-centric or structured world models improve generalization and sample efficiency?
- How should world models handle long-horizon prediction without compounding errors?
- Can diffusion models serve as effective world models, and what advantages do they offer over autoregressive alternatives?

---

## 6. Hierarchical Reinforcement Learning (HRL)

- How can meaningful temporal abstractions (options, skills, subgoals) be discovered automatically without supervision?
- What are the best objective functions for learning reusable, composable skills (mutual information, empowerment, coverage, task reward)?
- How should the manager-worker (high-level/low-level) communication interface be designed in feudal/hierarchical architectures?
- Can hierarchical RL provably improve exploration and sample efficiency in long-horizon tasks?
- How should credit assignment work across hierarchy levels, and how can non-stationarity at lower levels be addressed?
- Can language serve as an effective abstraction layer for hierarchical decision-making?
- How do option discovery methods (eigenoption, diversity-based, bottleneck-based) compare in transfer and generalization?
- Can hierarchical RL scale to real-world robotic manipulation and navigation tasks?
- How should hierarchical policies be evaluated - on compositional generalization, transfer, or raw task performance?
- What is the relationship between hierarchical RL and planning with learned abstract models?

---

## 7. Multi-Agent Reinforcement Learning (MARL)

- How can credit assignment be performed effectively in cooperative multi-agent systems (beyond QMIX, MAPPO, COMA)?
- What are the best paradigms for MARL: centralized training with decentralized execution (CTDE), fully decentralized, or communication-based?
- How can emergent communication protocols between agents be encouraged and interpreted?
- What equilibrium concepts (Nash, correlated, coarse correlated) are appropriate learning targets, and how can they be found efficiently?
- How can MARL algorithms handle large numbers of agents (hundreds or thousands) without combinatorial explosion?
- Can mean-field approximations or population-based methods scale MARL to realistic multi-agent environments?
- How should agents handle non-stationary environments caused by co-adapting agents?
- What are the best approaches for opponent modeling, theory of mind, and recursive reasoning in competitive settings?
- Can cooperative MARL benefit from shared representations, shared experience, or parameter sharing?
- How can fairness, social welfare, and equity be incorporated into MARL objectives?
- What role does communication (learned or structured) play in improving coordination?
- Can MARL methods handle mixed cooperative-competitive (general-sum) settings effectively?

---

## 8. Reward Design & Reward Learning

- How can reward functions be specified for complex, real-world tasks where manual reward design is impractical?
- What are the relative merits of reward shaping, potential-based shaping, intrinsic rewards, and learned rewards?
- How can inverse reinforcement learning (IRL) and reward learning from demonstrations scale to high-dimensional, complex environments?
- Can reward models learned from human preferences (as in RLHF) generalize to out-of-distribution states and behaviors?
- How can reward hacking and reward misspecification be detected and prevented?
- What is the relationship between reward model overoptimization and Goodhart's Law in RL?
- Can reward functions be inferred from natural language instructions, videos, or other weak supervision?
- How should multi-objective rewards be specified and optimized (Pareto optimality, scalarization, constrained optimization)?
- Can curiosity/intrinsic motivation replace extrinsic rewards entirely for open-ended learning?
- How do learned reward models degrade over the course of RL training, and how can this be mitigated (e.g., iterative RLHF, online reward learning)?

---

## 9. RLHF & Alignment

- How can RLHF be made more sample-efficient in terms of the number of human comparisons required?
- What are the fundamental limitations of learning from pairwise preferences vs. cardinal ratings vs. rankings?
- Can direct alignment methods (DPO, IPO, KTO, ORPO) match or surpass PPO-based RLHF, and under what conditions?
- How should reward model capacity, training data diversity, and evaluation be managed to avoid reward model collapse?
- Can constitutional AI (RLAIF) - using AI feedback instead of human feedback - achieve comparable alignment quality?
- How can RLHF handle diverse, conflicting, or culturally-varying human preferences?
- What are the best strategies for preventing reward hacking in RLHF-trained models?
- How can alignment be maintained as models are fine-tuned for downstream tasks (alignment tax)?
- Can online/iterative RLHF (where the reward model is updated with new data from the evolving policy) improve stability?
- How should RLHF be extended to multi-turn, agentic, and tool-use settings?
- What are the theoretical guarantees of preference-based RL under misspecified preference models?

---

## 10. Policy Optimization & Algorithms

- What are the fundamental differences between on-policy (PPO, A2C) and off-policy (SAC, TD3) methods in terms of stability, sample efficiency, and asymptotic performance?
- Can trust-region methods (TRPO, PPO) be improved with better constraint mechanisms or natural gradient approximations?
- How do maximum entropy RL methods (SAC, SQL) compare to standard RL in exploration, robustness, and transfer?
- What are the best variance reduction techniques for policy gradient methods (baselines, control variates, GAE, V-trace)?
- Can evolutionary strategies (ES), population-based training (PBT), and quality-diversity methods complement gradient-based RL?
- How should continuous and discrete action spaces be handled differently in policy optimization?
- Can policy optimization be made more robust to hyperparameter choices (learning rate, discount factor, clipping, etc.)?
- What is the role of entropy regularization, KL constraints, and other regularizers in policy optimization?
- How do implicit policies (energy-based, diffusion-based) compare to explicit parameterized policies for multi-modal action distributions?
- Can zeroth-order or derivative-free optimization methods be competitive with backpropagation-based RL?

---

## 11. Representation Learning for RL

- What properties make a good state representation for RL (bisimulation, predictive, contrastive, reconstructive)?
- How do self-supervised representation learning methods (CURL, SPR, ATC, BYOL-Explore) improve RL performance?
- Can pre-trained visual encoders (CLIP, DINOv2, ViT) be effectively used as frozen feature extractors for RL?
- How should representation learning and policy learning interact - should they be joint, alternating, or decoupled?
- Can object-centric representations improve generalization and sample efficiency in RL?
- How do bisimulation-based metrics (DeepMDP, DBC) compare to other approaches for learning task-relevant representations?
- Can language-grounded representations improve generalization across tasks and environments?
- What is the impact of representation dimensionality, architecture, and pre-training data on downstream RL performance?
- How can representations be made invariant to task-irrelevant factors while preserving task-relevant information?
- Can world-model representations serve as effective state representations for policy learning?

---

## 12. Generalization & Transfer

- How can RL agents generalize to unseen environments, levels, or task variations (procedural generation, domain randomization)?
- What are the most effective methods for sim-to-real transfer (domain randomization, domain adaptation, system identification)?
- Can meta-RL algorithms (MAML, RL², PEARL) achieve fast adaptation to new tasks with minimal interaction?
- How should training environment distributions be designed to maximize generalization (automatic curriculum, UED, PAIRED)?
- Can in-context learning (as in Decision Transformers or Algorithm Distillation) replace explicit meta-learning?
- How do pre-trained foundation models (LLMs, VLMs) serve as priors for RL generalization?
- What benchmarks and metrics best capture meaningful generalization in RL?
- Can skill libraries or task decompositions enable compositional generalization to novel tasks?
- How does the diversity of training environments affect generalization, and can diversity be optimized?
- Can zero-shot or few-shot transfer be achieved across significantly different environments or embodiments?

---

## 13. Continual & Lifelong Learning

- How can RL agents learn continuously across a sequence of tasks without catastrophic forgetting?
- What are the best methods for balancing plasticity (learning new tasks) and stability (retaining old knowledge)?
- Can modular architectures (progressive networks, PackNet, attention-based routing) enable effective continual RL?
- How should experience replay be designed for continual learning (reservoir sampling, prioritized replay, task-aware replay)?
- Can RL agents develop increasingly general skills over a lifetime of diverse experience?
- What is the relationship between continual RL and open-ended learning?
- How should performance be evaluated in continual RL (forward transfer, backward transfer, total regret)?
- Can world models or successor features mitigate forgetting in continual RL?
- How does non-stationarity from continual learning interact with non-stationarity from environment changes?

---

## 14. Goal-Conditioned & Multi-Task RL

- How can goal-conditioned policies generalize to unseen goals effectively?
- What are the best goal representation spaces (state, image, language, latent embedding)?
- How should hindsight experience replay (HER) and its variants be designed for maximum effectiveness?
- Can automatic goal generation and curriculum learning improve goal-conditioned RL (MEGA, CURIOUS, GoExplore)?
- How should multi-task RL handle task interference, gradient conflicts, and negative transfer?
- Can universal value functions and successor features scale to complex, high-dimensional environments?
- How should task specification work - reward functions, goal states, language instructions, demonstrations, or preferences?
- Can multi-task RL benefit from shared representations, modular policies, or mixture-of-experts architectures?
- What is the optimal training distribution over tasks for multi-task RL?

---

## 15. Language & Foundation Models for RL

- How can LLMs serve as planners, reward models, world models, or policy priors for RL agents?
- Can LLMs provide useful reward signals (language-based reward shaping) for RL training?
- How should vision-language models (VLMs) be integrated with RL for visually-grounded decision-making?
- Can RL agents use language for reasoning, planning, and self-reflection during decision-making?
- How do language-conditioned policies compare to goal-conditioned or reward-conditioned alternatives?
- Can pre-trained LLMs be fine-tuned with RL for agentic tasks (web browsing, coding, tool use)?
- What are the best methods for grounding LLM knowledge in embodied environments?
- Can chain-of-thought or tree-of-thought reasoning improve RL planning and decision-making?
- How can LLM-based agents learn from environmental feedback rather than static datasets?
- What is the role of in-context RL (providing RL-like adaptation through the LLM context window)?

---

## 16. Safe & Constrained RL

- How can RL agents satisfy safety constraints during both training and deployment?
- What are the best formulations for safe RL: constrained MDPs (CMDPs), risk-sensitive objectives, or constraint satisfaction?
- Can Lagrangian-based methods (CPO, RCPO, Lagrangian PPO) reliably satisfy constraints without excessive conservatism?
- How should safety constraints be specified - hard constraints, probabilistic constraints, chance constraints, or CVaR?
- Can formal verification or shielding methods guarantee safety while allowing RL to optimize performance?
- How can safe exploration be performed (safe set expansion, Lyapunov-based methods, reachability analysis)?
- What is the role of human oversight and intervention in safe RL training?
- Can uncertainty quantification enable safer decision-making in safety-critical applications?
- How should safety be maintained during transfer from simulation to real-world deployment?
- Can robust MDPs and distributionally robust optimization provide meaningful safety guarantees?
- How do risk measures (VaR, CVaR, entropic risk) affect policy behavior and constraint satisfaction?

---

## 17. Robustness & Adversarial RL

- How can RL policies be made robust to perturbations in observations, actions, transitions, or rewards?
- Can adversarial training (RARL, robust adversarial RL) produce meaningfully more robust policies?
- How should distributional robustness be incorporated into RL (distributionally robust MDPs)?
- What are the best approaches for handling sim-to-real gaps in dynamics, perception, and control?
- Can certified robustness (provable bounds on worst-case performance) be achieved for RL policies?
- How vulnerable are RL policies to adversarial observations, and how can this be mitigated?
- What is the relationship between robustness to adversarial perturbations and generalization to natural distribution shifts?
- Can domain randomization be optimized to improve robustness (automatic domain randomization, ADR)?

---

## 18. Imitation Learning & Learning from Demonstrations

- What are the relative advantages of behavioral cloning (BC), DAgger, IRL, and GAIL for different settings?
- How can compounding errors in behavioral cloning be mitigated without interactive expert access?
- Can demonstrations from suboptimal or heterogeneous demonstrators be effectively leveraged?
- How many demonstrations are needed for effective imitation, and how does this scale with task complexity?
- Can visual imitation learning (learning from videos without action labels) match action-labeled imitation?
- How should imitation learning handle distributional shift between expert and agent-visited states?
- Can one-shot or few-shot imitation learning generalize to novel tasks from a single demonstration?
- How do diffusion-based imitation learning policies compare to other approaches for multi-modal action distributions?
- Can cross-embodiment imitation learning enable transfer between different robots or morphologies?
- What is the best way to combine imitation learning with RL (pre-training, reward shaping, constraint, regularization)?

---

## 19. Temporal Abstraction & Options Framework

- How can the options framework be scaled to complex, high-dimensional environments?
- What initiation sets, termination conditions, and intra-option policies should be learned, and how?
- Can temporal abstractions be learned from data without any task reward (unsupervised option discovery)?
- How do learned options transfer across tasks, and what makes an option transferable?
- Can the option-critic architecture be improved for better option diversity and utility?
- How should planning with options (SMDP planning) be performed efficiently?
- What is the relationship between options, macro-actions, skills, and subgoals?
- Can attention mechanisms or transformers learn implicit temporal abstractions?

---

## 20. Credit Assignment

- How can long-horizon credit assignment be improved beyond temporal-difference methods and eligibility traces?
- Can attention-based or transformer-based methods improve credit assignment over long episodes?
- How do hindsight methods (hindsight credit assignment, return-conditioned learning) compare to forward-looking methods?
- Can causal reasoning improve credit assignment in RL?
- What is the role of reward decomposition and reward redistribution in improving credit assignment?
- How should credit be assigned in multi-agent systems where individual contributions are entangled?
- Can counterfactual reasoning ("what would have happened if...") improve credit assignment?
- How does the discount factor interact with credit assignment, and can it be learned or adapted?

---

## 21. Curriculum Learning & Automatic Environment Design

- How should task difficulty be sequenced to maximize learning efficiency (curriculum learning)?
- Can automatic curriculum generation (PLR, PAIRED, UED) produce robust and generalizable agents?
- What objective should curriculum learning optimize - regret, learning progress, competence, coverage?
- How should environment parameters (difficulty, complexity, diversity) be adapted during training?
- Can self-play and adversarial environment generation produce meaningful curricula?
- How does curriculum learning interact with exploration strategies?
- Can LLMs or generative models be used to design curricula or generate training environments?

---

## 22. RL for Robotics & Embodied AI

- How can sim-to-real transfer be made reliable for contact-rich manipulation, locomotion, and dexterous tasks?
- What are the best approaches for learning from real-world robot interaction with minimal data?
- Can RL produce robust locomotion and manipulation policies that generalize across environments and objects?
- How should tactile, proprioceptive, and visual sensing be integrated for RL-based robotic control?
- Can RL agents learn long-horizon manipulation tasks (cooking, assembly, tidying) from scratch?
- How should RL handle deformable objects, fluids, and other complex physical phenomena?
- Can RL policies run at real-time control frequencies (100Hz+) for dynamic tasks?
- How can human-robot collaboration be formulated and solved with RL?
- What is the role of simulation fidelity in sim-to-real RL?
- Can RL-trained robots adapt to damage, wear, or hardware changes?

---

## 23. RL for Games & Complex Decision-Making

- Can RL achieve superhuman performance in imperfect-information games (poker, Stratego, diplomacy) with principled methods?
- How should RL handle games with extremely large action spaces (RTS games, card games with combinatorial actions)?
- Can general game-playing agents be developed that perform well across diverse games without game-specific engineering?
- How do Monte Carlo Tree Search (MCTS) and RL interact, and can this synergy be improved beyond AlphaZero?
- Can RL master open-ended games (Minecraft, sandbox games) with emergent complexity?
- How should reward signals be designed for games with subjective quality measures (creativity, entertainment)?
- Can self-play methods converge to robust strategies in non-transitive games?

---

## 24. RL for Combinatorial Optimization

- Can RL solve combinatorial optimization problems (TSP, scheduling, bin packing, routing) competitively with specialized algorithms?
- How do RL-based approaches compare to classical heuristics, metaheuristics, and exact solvers in quality and runtime?
- Can RL learn construction heuristics that generalize to different problem sizes and distributions?
- How should graph neural networks and attention mechanisms be integrated with RL for combinatorial problems?
- Can RL-based approaches handle dynamic, stochastic, or online combinatorial optimization?
- What is the role of RL in learning to configure or select algorithms (algorithm configuration/selection)?
- Can RL discover novel heuristics or algorithms that outperform human-designed ones?

---

## 25. RL for Science & Engineering

- How can RL accelerate scientific discovery (materials design, drug discovery, chemical synthesis, experiment design)?
- Can RL optimize complex engineering systems (HVAC control, data center cooling, chip design, compiler optimization)?
- How should domain knowledge and physical constraints be incorporated into RL for scientific applications?
- Can RL be applied to adaptive experimental design and active learning in scientific settings?
- How effective is RL for controlling plasma in nuclear fusion reactors, and what are the key challenges?
- Can RL optimize molecular generation and retrosynthetic planning in chemistry?

---

## 26. RL for NLP & Code Generation

- How does RL fine-tuning (RLHF, RLAIF) improve LLM capabilities beyond supervised fine-tuning?
- Can RL optimize LLMs for specific objectives (factuality, reasoning, safety, helpfulness) simultaneously?
- How should RL be applied to code generation - optimizing for correctness (unit tests), efficiency, or readability?
- Can RL improve LLM reasoning through process reward models (PRM) vs. outcome reward models (ORM)?
- How does RL interact with chain-of-thought prompting and search (e.g., AlphaProof-style methods)?
- What are the best RL algorithms for the LLM fine-tuning setting (PPO, REINFORCE, GRPO, ReMax)?

---

## 27. Distributional RL

- How does distributional RL (learning return distributions vs. expected returns) improve policy learning?
- What return distribution representations (categorical, quantile, implicit quantile) are most effective and when?
- Can distributional RL provide meaningful uncertainty estimates for risk-sensitive decision-making?
- How does distributional RL interact with exploration, and can it be used for uncertainty-driven exploration?
- Can risk-sensitive policies (optimizing CVaR, worst-case, or other risk measures) be effectively learned via distributional RL?
- Does distributional RL provide benefits beyond its regularization effect, and can this be disentangled?

---

## 28. Inverse RL & Preference Learning

- How can IRL scale to high-dimensional, continuous environments with complex demonstrations?
- Can IRL recover reward functions that are robust, transferable, and interpretable?
- How do Bayesian IRL, maximum entropy IRL, and adversarial IRL compare in theory and practice?
- Can IRL handle demonstrations from multiple agents with different (possibly conflicting) reward functions?
- How should IRL handle suboptimal or noisy demonstrations?
- Can preference-based RL (learning from pairwise comparisons) converge as efficiently as reward-based RL?

---

## 29. Offline-to-Online & Hybrid RL

- What are the best strategies for transitioning from offline pre-training to online fine-tuning?
- How can offline-to-online RL avoid performance collapse during the transition phase?
- Can offline RL provide a warm start that accelerates online learning without constraining final performance?
- Can pre-trained world models from offline data accelerate online model-based RL?

---

## 30. Interpretability & Explainability

- How can RL policies be made interpretable to humans (saliency maps, decision trees, language explanations)?
- Can RL agents explain their decisions in natural language?
- How can reward functions learned via IRL or RLHF be interpreted and validated?
- Can causal explanations ("I did X because Y") be extracted from RL agents?
- How does interpretability trade off with policy performance in RL?

---

## 31. Memory & Recurrence in RL

- How should RL agents handle partial observability - RNNs, LSTMs, transformers, state-space models, or external memory?
- Can transformers with long context windows replace recurrent architectures for POMDPs?
- How does the credit assignment problem interact with memory and recurrence?
- Can RL agents learn to selectively store and retrieve task-relevant information?

---

## 32. Causal RL

- How can causal reasoning improve RL sample efficiency, generalization, and robustness?
- Can causal models of the environment be learned and used for counterfactual planning?
- How does causal discovery interact with world model learning in RL?
- Can causal abstractions provide better state representations for RL?
- How does causal RL relate to transfer learning - does learning causal structure improve transfer?

---

## 33. Evaluation & Reproducibility

- How can reproducibility challenges in deep RL (sensitivity to seeds, hyperparameters, implementation details) be addressed?
- Are current RL benchmarks (Atari, MuJoCo, DMC, Procgen, NetHack, Minecraft) representative of real-world challenges?
- What new benchmarks are needed to evaluate capabilities like generalization, continual learning, safety, and multi-agent coordination?

---

## 34. Real-World RL Challenges

- What are the key barriers to deploying RL in real-world systems (safety, sample efficiency, sim-to-real gap, interpretability, latency)?
- How can RL handle non-stationarity in real-world environments (changing user preferences, evolving systems)?
- Can RL operate effectively with noisy, delayed, or incomplete observations typical of real-world sensors?
- How should RL handle irreversible actions and the impossibility of resetting real-world environments?

---

## 35. Open-Ended & Emergent Learning

- Can RL agents achieve open-ended learning - continually discovering new skills and behaviors without a fixed task?
- How can quality-diversity algorithms (MAP-Elites, AURORA) be combined with RL for open-ended skill discovery?
- Can self-play and auto-curricula produce unbounded complexity growth (as hypothesized by open-endedness research)?
- How should open-ended learning be measured and evaluated?
- What is the role of intrinsic motivation in driving open-ended learning?
- Can RL agents create their own goals, tasks, and evaluation criteria?

---

## 36. Neuroscience-Inspired RL

- What insights from biological reward systems (dopamine, basal ganglia) can improve RL algorithms?
- Can meta-RL explain aspects of prefrontal cortex function and rapid learning in animals?
- How do biological credit assignment mechanisms (eligibility traces, neuromodulation) compare to algorithmic methods?
- Can episodic memory and hippocampal replay inspire better experience replay methods?
- What can successor representations and predictive coding contribute to RL algorithm design?
- How do biological exploration strategies (curiosity, play, novelty-seeking) map to computational exploration methods?
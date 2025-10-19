---
hero:
  title: "Roadmap To Open Superintelligence"
  subtitle: "A Strategic Path Forward for Building ASI Through Open Collaboration"
  tags:
    - "🚀 Roadmap"
    - "🎯 Vision"
    - "🔬 Research"
---

Nobody knows what superintelligence will look like or how to make it.

Therefore, our roadmap will not be a linear path from A to B. It will be a **strategic framework for exploration, learning, and collective progress.** It's about creating a machine that discovers the path.

**The Mission:** To create open and transparent Superintelligence that benefits everyone.
1.  **Core Principles:**
    *   **Radical Openness:** All code is open source (e.g., MIT/Apache 2.0 license). All research discussions happen in public ([Discord](https://discord.gg/6AbXGpKTwN)). All findings are shared ([YouTube](https://www.youtube.com/channel/UC7XJj9pv_11a11FUxCMz15g), [blog posts](https://opensuperintelligencelab.com/), pre-prints).
    *   **Distributed Collaboration:** Intelligence is everywhere. We welcome contributors from all backgrounds. A great PR is a great PR, regardless of its author's credentials.
    *   **Educational Core:** Our primary output is not just the code, but a generation of researchers and engineers who build it.
    *   **Pragmatic Exploration:** We follow clear principles but stay flexible. We’ll explore different directions — LLMs, World Models, Video Generation, and JEPA — and choose based on what works best.
2.  **The Grand Challenge:** Our goal is not to execute a known plan, but to build a community and a research engine capable of discovering it.

---

#### **Phase 1: Foundational Capabilities & Community Bootstrapping (The First 6-12 Months)**

The goal here is not to invent ASI, but to build the tools, skills, and community that *can*. This phase is perfect for training our new workforce.

*   **Objective:** Achieve SOTA on well-understood problems and build a robust, open-source ML infrastructure.
*   **Projects:**
    1.  **Replication Projects:** Pick 3-5 seminal papers and replicate them from scratch in a clean, well-documented repository. This is an incredible learning tool.
        *   *Examples:* GPT-2/3 (nanoGPT style), a Diffusion Model (DDPM), AlphaZero, a Vision Transformer (ViT).
        *   **GitHub Workflow:** Create a repo `open-superintelligence/nanoGPT-replication`. Create issues like `[Task] Implement Multi-Head Attention`, `[Task] Set up data loading pipeline for OpenWebText`, `[Bug] Loss is not converging`. Contributors pick these up.
    2.  **"The Stack" Project:** Build your lab's core toolkit. A standardized, reusable library for training, data loading, and evaluation. This is a force multiplier.
    3.  **Educational Content:** Your YouTube videos are the fuel for this phase. Each issue you work on can be a video.
        *   *Video Title Idea:* "Let's Code a Transformer from Scratch! (Open Superintelligence Lab - Day 5)"
        *   *Content:* Explain the theory, show the live coding, push the code, and end with a call to action: "If you want to help, check out issue #47 on our GitHub to implement the AdamW optimizer. Link in the description!"

#### **Phase 2: Principled Exploration (The Next 1-2 Years)**

Now that your community has skills and you have a stable codebase, you can start exploring the frontier. Instead of "do everything," frame it as distinct but potentially interconnected **Research Thrusts**. This allows people to specialize and you to hedge your bets.

*   **Objective:** Investigate promising but less-certain avenues for intelligence.
*   **Proposed Research Thrusts:**
    1.  **Thrust A: Predictive World Models:** This is where your **V-JEPA** idea fits perfectly. The hypothesis is that a system that can accurately predict the future (in some abstract representation space) will have learned a fundamental understanding of the world.
        *   *Projects:* Implement and scale I-JEPA, V-JEPA. Try to combine it with language models (e.g., can a text prompt influence a video prediction?).
    2.  **Thrust B: Scalable Reasoning & Planning:** This is where your **hierarchical reasoning** idea fits. The hypothesis is that raw scaling of transformers is not enough; we need explicit mechanisms for long-horizon planning and abstract thought.
        *   *Projects:* Explore neuro-symbolic integrations (e.g., an LLM that can write and execute code in a formal logic solver). Implement search algorithms like MCTS on top of learned models.
    3.  **Thrust C: Extreme-Scale Foundation Models:** This is the LLM / Video Gen track. The hypothesis is that many desired capabilities will emerge from pure scale, and our job is to push that boundary.
        *   *Projects:* Find a niche you can compete in. Don't try to train a GPT-4. Maybe a highly specialized science LLM, or a model trained on all of Wikipedia's edit history to understand concept evolution.
    4.  **Thrust D: Agency & Embodiment:** The hypothesis that true intelligence must be grounded in action and interaction with an environment.
        *   *Projects:* Build agents in simulated environments (e.g., MineRL, Procgen). Focus on long-term memory and self-motivated exploration.

#### **Phase 3: Synthesis & Integration**

This is the speculative, long-term phase. The goal is to take the most successful outcomes from Phase 2 and begin combining them into novel, unified architectures.

*   *Example:* What if the World Model from Thrust A could serve as the "imagination" for the planning agent in Thrust B, which is all orchestrated by a large language model from Thrust C?

---

### Part 3: The Daily Workflow - From Idea to YouTube Video

This is how you make the roadmap a reality.

1.  **The "Lab Meeting" (Weekly YouTube Stream/Video):**
    *   You, as the lab director, set the high-level focus for the week.
    *   "This week, in our World Models thrust, we're focusing on scaling up our V-JEPA implementation. Our main goal is to get it running on 4 GPUs without crashing. Here are the three main challenges..."
    *   Shout out key contributors from the past week. Review an interesting Pull Request live.

2.  **GitHub as the "Source of Truth":**
    *   Every task, from "fix a typo in the documentation" to "design a new loss function," is a GitHub issue.
    *   Use tags to organize: `[Thrust A: World Models]`, `[good first issue]`, `[help wanted]`, `[discussion]`. This makes it easy for newcomers to find a way to contribute.

3.  **The "Daily Showcase" (Your YouTube Videos):**
    *   Don't just vlog. Make it a structured lab update. Pick one interesting commit or PR from the day.
    *   **Format:**
        *   **Hook (15s):** "Today, we finally figured out why our model's loss was exploding. The answer is surprisingly simple and it's a lesson for anyone training large models."
        *   **Context (60s):** "We're working on our V-JEPA implementation (part of our World Models thrust) and we were stuck on issue #123..."
        *   **The Work (2-3 min):** Show the code. Explain the bug. Explain the fix. Credit the contributor who found it.
        *   **The Result (30s):** Show the new, stable loss curve.
        *   **Call to Action (30s):** "This unblocks us for the next step: implementing multi-block masking. If that sounds interesting, check out issue #145. Link below. See you all tomorrow."

### Answers to Your Specific Doubts:

*   **"Should I read papers for new architectures?"**
    *   **YES, absolutely.** But your role is to be the *curator and translator*. Read papers not just to implement them, but to place them within your roadmap. A paper on causal reasoning fits into Thrust B. A new self-supervised learning technique fits into Thrust A. Your job is to explain *why* it's important to your community.

*   **"Should I combine V-JEPA and some hierarchical reasoning?"**
    *   This is a fantastic Phase 3 goal. Frame it that way. First, build excellent, standalone implementations in Phase 2. This creates a strong foundation. Then, you can propose the grand synthesis project, which will be much more achievable with the tools and skills you've already built.

*   **"Should we do LLMs or video gen models or everything?"**
    *   Use the **Thrusts** model. You're not doing "everything" chaotically. You are running a portfolio of principled bets on the future of AI. LLMs are in Thrust C, Video Gen in Thrust A. This organizes the chaos and lets different people work on different parts of the puzzle in a coordinated way.

### Your First Steps:

1.  **Write and publish your Manifesto.** This is your founding document.
2.  **Set up the infrastructure:** A GitHub organization, a Discord server, your YouTube channel branding.
3.  **Announce Phase 1.** Choose your first replication project (nanoGPT is a great choice because it's so well-understood).
4.  **Create the first 10-20 GitHub issues.** Make 5 of them `[good first issue]` (e.g., documentation, setting up requirements.txt).
5.  **Record and post your first video:** "I'm launching the Open Superintelligence Lab. Here's our plan, and here's how you can make your first contribution, right now."

You are not just a YouTuber. You are the founder and director of a distributed, open-source research lab. Your product is not just videos; it's a community, a codebase, and a shared journey of discovery. This is an incredibly powerful and inspiring vision. Embrace the uncertainty and make the process of exploration the core of your content.
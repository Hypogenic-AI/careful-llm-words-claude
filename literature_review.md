# Literature Review: An LLM That's Careful With Its Words

## Research Area Overview

This literature review covers research on augmenting LLM generation with explicit intermediate computation—whether through chain-of-thought prompting, pause/filler tokens, internal rationales, or latent reasoning—to produce more careful, deliberate text. The central question is: **does forcing a model to "think" between sentences produce qualitatively different (and better) output?**

The field spans three interconnected threads: (1) chain-of-thought prompting and its variants, (2) pause/filler/thinking tokens that provide extra computation without semantic content, and (3) implicit/latent reasoning approaches that internalize deliberation.

---

## Key Papers

### Paper 1: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
- **Authors**: Wei et al.
- **Year**: 2022
- **Source**: arXiv:2201.11903 (NeurIPS 2022)
- **Key Contribution**: Foundational demonstration that including step-by-step reasoning exemplars in prompts enables multi-step reasoning in LLMs. A 540B PaLM model with 8 CoT exemplars achieved SOTA on GSM8K.
- **Methodology**: Few-shot prompting with manually written reasoning chains as exemplars.
- **Datasets Used**: GSM8K, SVAMP, ASDiv, AQuA, MAWPS, SingleEq, AddSub, MultiArith, CommonsenseQA, StrategyQA, Date Understanding, Sports Understanding, SayCan, CSQA, Letter Concatenation, Coin Flip
- **Results**: CoT prompting substantially improves reasoning, especially at scale (>100B parameters).
- **Code Available**: No
- **Relevance**: Establishes that intermediate reasoning tokens improve output quality—the conceptual foundation for our hypothesis.

### Paper 2: Large Language Models are Zero-Shot Reasoners
- **Authors**: Kojima, Gu, Reid, Matsuo, Iwasawa
- **Year**: 2022
- **Source**: arXiv:2205.11916 (NeurIPS 2022)
- **Key Contribution**: "Let's think step by step" triggers CoT reasoning without exemplars.
- **Methodology**: Two-stage zero-shot prompting: (1) append "Let's think step by step" to extract reasoning, (2) extract answer from reasoning.
- **Datasets Used**: MultiArith, GSM8K, AQuA, SVAMP, AddSub, SingleEq, CommonsenseQA, StrategyQA, Date Understanding, Shuffled Objects, Last Letter Concatenation, Coin Flip
- **Results**: Dramatic improvements across reasoning tasks. MultiArith: 17.7% → 78.7% with PaLM 540B.
- **Code Available**: No
- **Relevance**: Shows that even minimal prompting for intermediate computation improves output—supports the idea that "thinking" instructions between sentences could help.

### Paper 3: Think Before You Speak: Training Language Models With Pause Tokens (DEEP READ)
- **Authors**: Goyal, Ji, Rawat, Menon, Kumar, Nagarajan
- **Year**: 2023 (ICLR 2024)
- **Source**: arXiv:2310.02226
- **Key Contribution**: Proposes appending learnable `<pause>` tokens to give models extra computation before generating output. A 1B model improved on 8/9 tasks.
- **Methodology**:
  - **Pause-pretraining**: Insert `<pause>` tokens at random positions (10% of sequence length), skip loss on pause predictions. Model sees 90% of meaningful tokens.
  - **Pause-finetuning**: Append M_ft pause tokens to prefix, compute loss only on target.
  - **Pause-inference**: Append M_inf pauses, ignore outputs until last pause.
  - Only adds ~1024 parameters (one embedding).
- **Model Sizes**: 130M and 1B decoder-only transformers, pretrained on C4 for 200B tokens.
- **Datasets Used**: GSM8K, SQuAD V1, CoQA, CommonSenseQA, PhysicalIQA, LAMBADA, HellaSwag, WebQuestions, Natural Questions
- **Results (1B model)**:
  - SQuAD: 36.4 → 55.9 EM (+19.5)
  - CommonSenseQA: 26.9 → 34.8 EM (+7.9)
  - GSM8K: 7.5 → 8.5 Acc (+1.0)
  - Improved on 8/9 tasks; only HellaSwag showed no gain.
- **Key Findings**:
  - Pause-pretraining is crucial; retrofitting pauses on standard models gives lukewarm results.
  - Appending > prepending pause tokens.
  - Optimal pause count varies per task (10 vs 50).
  - Larger models benefit more.
  - Zero pauses at inference catastrophically breaks pause-pretrained models.
  - FLOPS-equivalent to adding ~2 layers but more flexible.
- **Code Available**: No
- **Relevance**: Most directly relevant prior work for inserting thinking tokens. Shows that extra computation via dummy tokens works but requires pretraining integration.

### Paper 4: Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking (DEEP READ)
- **Authors**: Zelikman, Harik, Shao, Jayasiri, Haber, Goodman
- **Year**: 2024
- **Source**: arXiv:2403.09629
- **Key Contribution**: LMs learn to generate internal rationales at *every token position* to predict future text, using REINFORCE optimization.
- **Methodology**:
  - **Think**: At every token, insert `<|startofthought|>`, generate t thinking tokens, end with `<|endofthought|>`. Uses parallel attention mask for efficiency.
  - **Talk**: A 3-layer MLP mixing head interpolates between base logits and post-thought logits.
  - **Learn**: REINFORCE with reward = improvement in future token prediction. Non-myopic scoring over n_true future tokens.
  - Meta-tokens initialized to em dash embedding with 100x gradient weight.
- **Model**: Mistral 7B (only size tested)
- **Datasets Used**: Trained on OpenWebMath and C4. Evaluated zero-shot on GSM8K and CommonsenseQA.
- **Results**:
  - GSM8K: 5.9% → 10.9% (OpenWebMath training)
  - CommonsenseQA: 36.3% → 47.2% (+10.9%)
  - Quiet-STaR + CoT majority@8: 40.6% → 47.7% on GSM8K
- **Key Findings**:
  - Longer thoughts = better performance (8, 10, 12, 16, 24 tokens tested)
  - Thoughts disproportionately help on difficult tokens
  - Mixing head is essential (without it, model ignores thoughts)
  - Complementary to explicit CoT
  - Substantial computational overhead
- **Code Available**: No (but trained on public datasets)
- **Relevance**: The closest prior work to "thinking tokens between every sentence." Demonstrates the concept works but at every-token granularity rather than sentence-level.

### Paper 5: Let's Think Dot by Dot: Hidden Computation in Transformer Language Models (DEEP READ)
- **Authors**: Pfau, Merrill, Bowman
- **Year**: 2024
- **Source**: arXiv:2404.15758
- **Key Contribution**: Meaningless filler tokens (dots) enable transformers to solve tasks they otherwise cannot, via hidden computation in representations.
- **Methodology**: 34M Llama model trained from scratch on synthetic tasks (3SUM, 2SUM-Transform). Models trained on 50/50 mix of filler-token and CoT sequences.
- **Theoretical Results**:
  - Without intermediates: transformers limited to TC0
  - With CoT: can escape TC0
  - With fillers: remain in TC0 but extend power within it (higher quantifier depth)
  - Problems with quantifier depth k solvable with n^k filler tokens
- **Results**:
  - 3SUM length 14: Filler ~100% vs no-filler ~50-55%
  - 2SUM-Transform: Filler 93.6% vs CoT 95.1% vs no-filler 78.7%
  - Filler recovers ~90% of CoT benefit on parallelizable tasks
- **Critical Finding**: Filler-only training (no CoT) fails. Models need parallelizable CoT supervision to learn filler usage. Instance-adaptive (serial) CoT doesn't transfer to filler usage.
- **Code Available**: Yes - https://github.com/JacobPfau/fillerTokens
- **Relevance**: Provides theoretical grounding for why thinking tokens help—they increase computational depth. Also raises concerns about unauditable hidden computation.

### Paper 6: Self-Notes: Learning to Reason and Memorize with Self-Notes (DEEP READ)
- **Authors**: Lanchantin, Toshniwal, Weston, Szlam, Sukhbaatar
- **Year**: 2023 (NeurIPS 2023)
- **Source**: arXiv:2305.00833
- **Key Contribution**: Models interleave generated "self-note" tokens *within* input context during reading, performing reasoning on-the-fly.
- **Methodology**: When model predicts a start-note token, it generates a note inline, then resumes reading. Four paradigms: supervised, semi-supervised, unsupervised, few-shot prompting.
- **Model**: GPT-2 base (117M) for fine-tuning; GPT-J 6B, GPT-3 175B, Llama 2 70B for few-shot.
- **Datasets**: Toy-Story, Algorithmic state tracking, Boolean variables, Chess, MultiArith, GSM8K
- **Results**:
  - Algorithmic OOD (101-200 statements): Self-Notes 85.0% vs Scratchpad 11.6% vs Vanilla 24.4%
  - Toy-Story 4-hop OOD: Self-Notes 97.8% vs Scratchpad 94.2% vs Vanilla 37.4%
  - GSM8K few-shot (GPT-3): Self-Notes 13.4% vs CoT 11.4%
  - Llama 2 70B GSM8K: Self-Notes 59.9% vs CoT 55.9%
- **Key Findings**:
  - Interleaving reasoning near relevant facts > post-context reasoning (scratchpad)
  - Critical for OOD generalization to longer sequences
  - Both position and content of notes matter; content matters more
  - Even 1% note supervision helps
- **Code Available**: No
- **Relevance**: Very close to our hypothesis—thinking tokens interleaved with text (not just at the end) produce better reasoning. Key insight: proximity of reasoning to relevant facts matters.

### Paper 7: STaR: Bootstrapping Reasoning With Reasoning
- **Authors**: Zelikman, Wu, Mu, Goodman
- **Year**: 2022
- **Source**: arXiv:2203.14465
- **Key Contribution**: Models learn to self-generate rationales through iterative bootstrapping.
- **Relevance**: Precursor to Quiet-STaR; shows models can learn to produce reasoning that improves their own performance.

### Paper 8: Tree of Thoughts
- **Authors**: Yao et al.
- **Year**: 2023
- **Source**: arXiv:2305.10601
- **Key Contribution**: Generalizes CoT with exploration of multiple reasoning paths and self-evaluation.
- **Results**: GPT-4 from 4% to 74% on Game of 24.
- **Relevance**: Demonstrates deliberate, careful reasoning through structured exploration.

### Paper 9: Scratchpads for Intermediate Computation
- **Authors**: Nye et al.
- **Year**: 2021
- **Source**: arXiv:2112.00114
- **Key Contribution**: Early demonstration that training LMs to emit intermediate computation steps dramatically improves multi-step tasks.
- **Relevance**: Foundational work showing intermediate tokens enable computation beyond model's native depth.

### Paper 10: Implicit Chain of Thought Reasoning via Knowledge Distillation
- **Authors**: Deng et al.
- **Year**: 2023
- **Source**: arXiv:2311.01460
- **Key Contribution**: Distills explicit CoT into implicit hidden-state computation ("vertical" vs "horizontal" reasoning).
- **Relevance**: Shows reasoning can happen without generating intermediate tokens, but our hypothesis tests whether explicit tokens still help.

### Paper 11: From Explicit CoT to Implicit CoT (Stepwise Internalization)
- **Authors**: Deng et al.
- **Year**: 2024
- **Source**: arXiv:2405.14838
- **Key Contribution**: Gradually removes CoT tokens while fine-tuning to internalize reasoning.
- **Results**: GPT-2 Small achieves high accuracy on 9x9 multiplication; outperforms GPT-4 on GSM8K without intermediate steps.
- **Relevance**: Contrast case—shows what happens when you remove rather than add thinking tokens.

### Paper 12: Coconut (Chain of Continuous Thought)
- **Authors**: Hao et al.
- **Year**: 2024
- **Source**: arXiv:2412.06769
- **Key Contribution**: Reasoning in continuous latent space by feeding last hidden state back as next input embedding, bypassing discrete tokens.
- **Relevance**: Alternative approach—continuous rather than discrete thinking tokens. Can encode multiple reasoning paths simultaneously.

### Paper 13: Compressed Chain of Thought
- **Authors**: Cheng, Van Durme
- **Year**: 2024
- **Source**: arXiv:2412.13171
- **Key Contribution**: "Contemplation tokens"—compressed dense representations of full reasoning chains.
- **Relevance**: Shows thinking tokens can be compact, directly relevant to efficient sentence-level thinking.

### Paper 14: Adaptive Computation Time
- **Authors**: Graves
- **Year**: 2016
- **Source**: arXiv:1603.08983
- **Key Contribution**: Foundational work on letting neural networks learn how many computational steps to take per input.
- **Relevance**: Conceptual ancestor—different inputs need different amounts of computation.

### Paper 15: Self-Refine
- **Authors**: Madaan et al.
- **Year**: 2023
- **Source**: arXiv:2309.06657
- **Key Contribution**: LLMs iteratively generate, critique, and refine their output. 20% improvement on average across tasks.
- **Relevance**: Alternative "careful generation" approach through iterative refinement rather than inline thinking.

---

## Common Methodologies

### Explicit Intermediate Tokens
- **Chain-of-Thought prompting**: Few-shot or zero-shot with reasoning exemplars (Wei et al., Kojima et al.)
- **Scratchpad**: Model trained to emit computation steps after context (Nye et al.)
- **Self-Notes**: Reasoning tokens interleaved within input context (Lanchantin et al.)
- **Quiet-STaR**: Internal rationales at every token position (Zelikman et al.)

### Computational Augmentation Without Semantic Content
- **Pause tokens**: Learnable dummy tokens appended to input (Goyal et al.)
- **Filler tokens**: Meaningless dots providing hidden computation (Pfau et al.)

### Latent/Implicit Reasoning
- **Implicit CoT**: Distill reasoning into hidden states (Deng et al.)
- **Coconut**: Continuous-space reasoning without discrete tokens (Hao et al.)
- **Compressed CoT**: Dense token representations of reasoning chains (Cheng & Van Durme)

---

## Standard Baselines
- **Direct/vanilla generation**: No intermediate reasoning
- **Few-shot CoT**: Wei et al. (2022) standard
- **Zero-shot CoT**: "Let's think step by step" (Kojima et al.)
- **Self-Consistency**: Sample multiple CoT paths, majority vote (Wang et al.)

## Evaluation Metrics
- **Accuracy/EM**: For reasoning benchmarks (GSM8K, CommonsenseQA, ARC)
- **F1 Score**: For QA tasks (SQuAD, CoQA)
- **Truthfulness**: TruthfulQA judge metrics
- **LLM-as-judge**: GPT-4 pairwise comparison for open-ended generation quality
- **Token efficiency**: Tokens consumed vs accuracy gained

## Datasets in the Literature
- **GSM8K**: Most common reasoning benchmark (used in Wei, Kojima, Goyal, Zelikman, Lanchantin)
- **CommonsenseQA**: Common commonsense benchmark (used in Goyal, Zelikman)
- **SQuAD/CoQA**: QA benchmarks (used in Goyal)
- **TruthfulQA**: Factual carefulness (relevant but not yet used in this literature)
- **Synthetic tasks**: 3SUM, state tracking, boolean evaluation (Pfau, Lanchantin)

---

## Gaps and Opportunities

1. **Sentence-level thinking**: Prior work inserts thinking at every token (Quiet-STaR) or only at the end (pause tokens). No work systematically evaluates thinking tokens *between sentences*—a natural granularity that balances computational cost with deliberation.

2. **Qualitative text analysis**: Most evaluation uses accuracy on benchmarks. Little work examines how thinking tokens change the *qualitative character* of generated text (hedging, self-correction, specificity, nuance).

3. **Prompt-only approaches**: Pause tokens and Quiet-STaR require training modifications. Our hypothesis tests whether prompting alone (instructing the model to think between sentences) can achieve similar effects—a zero-training approach.

4. **Text generation beyond QA**: Most evaluation focuses on QA/reasoning. Open-ended generation with thinking tokens is understudied.

5. **Truthfulness and carefulness**: TruthfulQA-style evaluation of thinking-augmented generation is unexplored.

---

## Recommendations for Our Experiment

### Recommended Datasets
1. **GSM8K** (primary reasoning benchmark—most comparable to prior work)
2. **TruthfulQA** (directly measures "carefulness" in claims)
3. **ARC-Challenge** (reasoning without arithmetic)
4. **CommonsenseQA** (commonsense reasoning)
5. **HellaSwag** (coherence/plausibility)
6. **MT-Bench** (open-ended multi-turn quality)

### Recommended Baselines
1. **Direct generation** (no thinking instructions)
2. **Zero-shot CoT** ("Let's think step by step")
3. **Sentence-level thinking** (our method: explicit thinking between each sentence)

### Recommended Metrics
1. **Accuracy** on reasoning benchmarks
2. **Truthfulness score** on TruthfulQA
3. **LLM-as-judge** for qualitative text comparison
4. **Token count** (efficiency analysis)
5. **Qualitative analysis** of generated text characteristics

### Methodological Considerations
- The Self-Notes finding that proximity of reasoning to relevant content matters supports our sentence-level approach.
- Pfau et al.'s finding that filler tokens need CoT supervision to work suggests our prompt-based approach should explicitly instruct meaningful thinking, not just pausing.
- Goyal et al.'s finding that pause-pretraining is crucial may limit the effectiveness of prompt-only approaches—but Kojima et al. showed zero-shot CoT works without training changes, so prompting may still be viable.
- Quiet-STaR's finding that longer thoughts help more suggests we should experiment with thinking depth.

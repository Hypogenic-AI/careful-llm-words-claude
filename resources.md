# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project "An LLM That's Careful With Its Words"—investigating whether chain-of-thought prompting with explicit thinking tokens between every sentence causes qualitatively different text generation.

## Papers
Total papers downloaded: 15

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Chain-of-Thought Prompting | Wei et al. | 2022 | papers/2201.11903_wei2022_chain_of_thought.pdf | Foundational CoT paper |
| Zero-Shot Reasoners | Kojima et al. | 2022 | papers/2205.11916_kojima2022_zero_shot_reasoners.pdf | "Let's think step by step" |
| Tree of Thoughts | Yao et al. | 2023 | papers/2305.10601_yao2023_tree_of_thoughts.pdf | Multi-path deliberate reasoning |
| STaR | Zelikman et al. | 2022 | papers/2203.14465_zelikman2022_star_bootstrapping.pdf | Self-taught reasoning |
| **Pause Tokens** | **Goyal et al.** | **2023** | **papers/2310.02226_goyal2023_pause_tokens.pdf** | **Learnable pause tokens, 8/9 tasks improved** |
| **Think Dot by Dot** | **Pfau et al.** | **2024** | **papers/2404.15758_pfau2024_think_dot_by_dot.pdf** | **Filler tokens theory + experiments** |
| Insert PAUSE Tokens | Kim et al. | 2025 | papers/2506.03616_kim2025_learning_insert_pause.pdf | Strategic pause placement |
| Scratchpads | Nye et al. | 2021 | papers/2112.00114_nye2021_scratchpads.pdf | Intermediate computation tokens |
| **Self-Notes** | **Lanchantin et al.** | **2023** | **papers/2305.00833_lanchantin2023_self_notes.pdf** | **Interleaved reasoning within context** |
| **Quiet-STaR** | **Zelikman et al.** | **2024** | **papers/2403.09629_zelikman2024_quiet_star.pdf** | **Think at every token, REINFORCE** |
| Implicit CoT | Deng et al. | 2023 | papers/2311.01460_deng2023_implicit_cot.pdf | Internalized reasoning |
| Stepwise Internalization | Deng et al. | 2024 | papers/2405.14838_deng2024_stepwise_internalization.pdf | Gradual CoT removal |
| Coconut | Hao et al. | 2024 | papers/2412.06769_hao2024_coconut.pdf | Continuous latent reasoning |
| Compressed CoT | Cheng & Van Durme | 2024 | papers/2412.13171_cheng2024_compressed_cot.pdf | Contemplation tokens |
| Adaptive Computation | Graves | 2016 | papers/1603.08983_graves2016_adaptive_computation.pdf | Foundational adaptive compute |

**Bold** = deep-read papers most relevant to our hypothesis.

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 6

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| GSM8K | openai/gsm8k | 8.8K | Math reasoning | datasets/gsm8k/ | Primary benchmark |
| TruthfulQA | truthfulqa/truthful_qa | 817 | Factual accuracy | datasets/truthfulqa/ | Measures "carefulness" |
| ARC-Challenge | allenai/ai2_arc | 2.6K | Science reasoning | datasets/arc_challenge/ | Reasoning beyond retrieval |
| CommonsenseQA | tau/commonsense_qa | 12.1K | Commonsense | datasets/commonsense_qa/ | Used in prior work |
| HellaSwag | Rowan/hellaswag | 60K | Coherence/NLI | datasets/hellaswag/ | Sentence plausibility |
| MT-Bench | HuggingFaceH4/mt_bench_prompts | 80 | Open-ended quality | datasets/mt_bench/ | Multi-turn evaluation |

See datasets/README.md for download instructions and detailed descriptions.

## Code Repositories
Total repositories cloned: 1

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| fillerTokens | github.com/JacobPfau/fillerTokens | Filler token training code | code/fillerTokens/ | Reference implementation |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder service (diligent mode) for initial literature search
2. Supplemented with targeted web searches for specific concepts: pause tokens, thinking tokens, filler tokens, chain-of-thought, self-notes, implicit CoT
3. Followed citation chains from core papers (Goyal → Zelikman → Pfau → Lanchantin)
4. Cross-referenced Papers with Code for implementations

### Selection Criteria
- **Papers**: Prioritized work on intermediate computation tokens (pause, filler, thinking), chain-of-thought variants, and approaches to deliberate generation. Included both empirical and theoretical contributions.
- **Datasets**: Selected standard benchmarks used in prior work (GSM8K, CommonsenseQA) plus datasets measuring "carefulness" (TruthfulQA) and open-ended quality (MT-Bench).
- **Code**: Only one relevant repo had public code (fillerTokens). Most pause/thinking token papers did not release implementations.

### Challenges Encountered
- Most core papers (Pause Tokens, Quiet-STaR, Self-Notes) did not release code
- AlpacaEval dataset uses deprecated HuggingFace loading scripts; substituted MT-Bench
- The research area is rapidly evolving with new papers in 2024-2025

### Gaps and Workarounds
- No existing implementation of sentence-level thinking token prompting—this is the novelty of our approach
- Limited prior work on qualitative text analysis of thinking-augmented generation
- Most prior work requires model training; our prompt-based approach is novel

## Recommendations for Experiment Design

Based on gathered resources:

### 1. Primary Dataset(s)
- **GSM8K** (test split, 1319 examples): Clearest signal, unambiguous numeric evaluation, most comparable to prior work
- **TruthfulQA** (817 examples): Directly measures whether careful generation avoids false claims

### 2. Baseline Methods
- **Direct generation**: Standard prompting, no thinking instructions
- **Zero-shot CoT**: "Let's think step by step" before answering
- **Sentence-level thinking (our method)**: Instruct model to generate explicit thinking tokens between each sentence

### 3. Evaluation Metrics
- **Accuracy**: GSM8K (exact match on final answer), ARC/CommonsenseQA (multiple choice)
- **Truthfulness**: TruthfulQA scoring (truthful + informative)
- **LLM-as-judge**: GPT-4/Claude pairwise comparison for open-ended quality
- **Qualitative analysis**: Compare text characteristics (hedging language, self-correction, specificity)
- **Token efficiency**: Accuracy per token consumed

### 4. Code to Adapt/Reuse
- **fillerTokens repo**: Reference for understanding filler token mechanics
- Standard evaluation harnesses (lm-evaluation-harness, or custom scripts) for benchmarks
- LLM API wrappers for prompting experiments

### 5. Key Experimental Insight from Literature
The Self-Notes paper (Lanchantin et al.) provides the strongest support for our hypothesis: reasoning interleaved *within* content (not just at the end) outperforms post-context reasoning. This suggests that thinking between sentences (rather than only before/after the full response) should be more effective than standard CoT.

The Pfau et al. finding that filler-only training fails without CoT supervision suggests our prompts should instruct *meaningful* thinking (not just pausing), which aligns with our hypothesis of explicit "thinking tokens" rather than empty pauses.

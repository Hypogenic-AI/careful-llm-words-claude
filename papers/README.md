# Downloaded Papers

## Core Papers (Deep Read)

1. **Think Before You Speak: Training Language Models With Pause Tokens** (2310.02226_goyal2023_pause_tokens.pdf)
   - Authors: Goyal, Ji, Rawat, Menon, Kumar, Nagarajan
   - Year: 2023 (ICLR 2024)
   - arXiv: 2310.02226
   - Why relevant: Directly proposes pause tokens for extra computation before generation. Shows improvements on 8/9 tasks with 1B model.

2. **Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking** (2403.09629_zelikman2024_quiet_star.pdf)
   - Authors: Zelikman, Harik, Shao, Jayasiri, Haber, Goodman
   - Year: 2024
   - arXiv: 2403.09629
   - Why relevant: Internal rationales at every token position. Most directly related to sentence-level thinking concept.

3. **Let's Think Dot by Dot: Hidden Computation in Transformer Language Models** (2404.15758_pfau2024_think_dot_by_dot.pdf)
   - Authors: Pfau, Merrill, Bowman
   - Year: 2024
   - arXiv: 2404.15758
   - Why relevant: Theoretical and empirical proof that meaningless filler tokens enable hidden computation. Code available.

4. **Learning to Reason and Memorize with Self-Notes** (2305.00833_lanchantin2023_self_notes.pdf)
   - Authors: Lanchantin, Toshniwal, Weston, Szlam, Sukhbaatar
   - Year: 2023 (NeurIPS 2023)
   - arXiv: 2305.00833
   - Why relevant: Self-notes interleaved within context—closest to thinking between sentences. Shows interleaving > post-context reasoning.

## Foundational Chain-of-Thought Papers

5. **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** (2201.11903_wei2022_chain_of_thought.pdf)
   - Authors: Wei et al.
   - Year: 2022 (NeurIPS 2022)
   - arXiv: 2201.11903
   - Why relevant: Foundational CoT paper establishing that intermediate reasoning tokens improve LLM output.

6. **Large Language Models are Zero-Shot Reasoners** (2205.11916_kojima2022_zero_shot_reasoners.pdf)
   - Authors: Kojima, Gu, Reid, Matsuo, Iwasawa
   - Year: 2022 (NeurIPS 2022)
   - arXiv: 2205.11916
   - Why relevant: "Let's think step by step"—shows minimal prompting for thinking dramatically improves performance.

7. **STaR: Bootstrapping Reasoning With Reasoning** (2203.14465_zelikman2022_star_bootstrapping.pdf)
   - Authors: Zelikman, Wu, Mu, Goodman
   - Year: 2022
   - arXiv: 2203.14465
   - Why relevant: Precursor to Quiet-STaR; self-generated rationales improve performance.

8. **Tree of Thoughts** (2305.10601_yao2023_tree_of_thoughts.pdf)
   - Authors: Yao et al.
   - Year: 2023
   - arXiv: 2305.10601
   - Why relevant: Deliberate reasoning through multiple path exploration.

## Thinking/Filler Token Papers

9. **Learning to Insert [PAUSE] Tokens for Better Reasoning** (2506.03616_kim2025_learning_insert_pause.pdf)
   - Authors: Kim et al.
   - Year: 2025
   - arXiv: 2506.03616
   - Why relevant: Strategic placement of pause tokens at low-confidence positions. Up to 4.7% improvement on GSM8K.

## Scratchpad and Implicit Reasoning

10. **Show Your Work: Scratchpads for Intermediate Computation** (2112.00114_nye2021_scratchpads.pdf)
    - Authors: Nye et al.
    - Year: 2021
    - arXiv: 2112.00114
    - Why relevant: Early work showing intermediate computation tokens improve multi-step tasks.

11. **Implicit Chain of Thought Reasoning via Knowledge Distillation** (2311.01460_deng2023_implicit_cot.pdf)
    - Authors: Deng et al.
    - Year: 2023
    - arXiv: 2311.01460
    - Why relevant: Shows reasoning can be internalized into hidden states.

12. **From Explicit CoT to Implicit CoT (Stepwise Internalization)** (2405.14838_deng2024_stepwise_internalization.pdf)
    - Authors: Deng et al.
    - Year: 2024
    - arXiv: 2405.14838
    - Why relevant: Contrast—what happens when you gradually remove thinking tokens.

## Alternative Approaches

13. **Coconut: Training LLMs to Reason in Continuous Latent Space** (2412.06769_hao2024_coconut.pdf)
    - Authors: Hao et al.
    - Year: 2024
    - arXiv: 2412.06769
    - Why relevant: Continuous-space alternative to discrete thinking tokens.

14. **Compressed Chain of Thought** (2412.13171_cheng2024_compressed_cot.pdf)
    - Authors: Cheng, Van Durme
    - Year: 2024
    - arXiv: 2412.13171
    - Why relevant: Compressed "contemplation tokens" for efficient thinking.

15. **Adaptive Computation Time for RNNs** (1603.08983_graves2016_adaptive_computation.pdf)
    - Authors: Graves
    - Year: 2016
    - arXiv: 1603.08983
    - Why relevant: Foundational work on adaptive computation—conceptual ancestor of thinking tokens.

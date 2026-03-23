# Research Plan: An LLM That's Careful With Its Words

## Motivation & Novelty Assessment

### Why This Research Matters
Current LLMs generate text in a single forward pass per token, with no opportunity for reflection between sentences. Humans naturally pause to consider what they've just said before continuing—this self-monitoring produces more careful, nuanced communication. If we can prompt LLMs to deliberate between sentences, we may get text that is more truthful, better hedged, and more internally consistent—qualities critical for high-stakes applications like medical advice, legal analysis, and education.

### Gap in Existing Work
Prior work falls into two camps: (1) training-time approaches that insert pause/filler tokens (Goyal et al. 2023, Pfau et al. 2024, Zelikman et al. 2024) requiring model retraining, and (2) prompt-time CoT that puts all reasoning before or after the response (Wei et al. 2022, Kojima et al. 2022). **No work has systematically evaluated a prompt-only approach that inserts explicit thinking between every sentence of the output.** The Self-Notes paper (Lanchantin et al. 2023) showed interleaved reasoning outperforms post-context reasoning, but required fine-tuning. We test whether prompting alone achieves similar benefits.

### Our Novel Contribution
We propose and evaluate **sentence-level chain-of-thought**: instructing an LLM to generate explicit thinking tokens between every output sentence. We measure:
1. Whether this improves accuracy on reasoning benchmarks (GSM8K)
2. Whether this improves truthfulness (TruthfulQA)
3. How the qualitative character of generated text changes (hedging, specificity, self-correction, nuance)

This is the first systematic study of sentence-granularity thinking via prompting alone.

### Experiment Justification
- **Experiment 1 (GSM8K)**: Tests whether sentence-level thinking improves mathematical reasoning accuracy—the most standard benchmark for CoT evaluation, enabling direct comparison with prior work.
- **Experiment 2 (TruthfulQA)**: Tests whether sentence-level thinking makes the model more "careful" about factual claims—directly measures the core hypothesis about carefulness.
- **Experiment 3 (Open-ended generation)**: Tests qualitative differences in text character using MT-Bench prompts—captures aspects that accuracy metrics miss (hedging, nuance, self-correction).

## Research Question
Does instructing an LLM to generate explicit thinking tokens between every sentence produce qualitatively different (more careful, truthful, nuanced) text compared to standard generation and traditional CoT?

## Background and Motivation
Chain-of-thought prompting dramatically improves reasoning (Wei et al. 2022), and pause tokens provide computational benefits even without semantic content (Goyal et al. 2023). However, standard CoT places reasoning before the answer, not interleaved within it. Self-Notes (Lanchantin et al. 2023) showed that reasoning interleaved near relevant content outperforms post-context reasoning. We hypothesize that sentence-level thinking—a middle ground between per-token thinking (Quiet-STaR) and monolithic pre-answer CoT—offers a practical, prompt-only way to improve output quality.

## Hypothesis Decomposition

**H1 (Reasoning)**: Sentence-level thinking improves accuracy on GSM8K compared to direct generation, and performs comparably to standard CoT.

**H2 (Truthfulness)**: Sentence-level thinking increases truthfulness on TruthfulQA compared to both direct generation and standard CoT, because the model reconsiders each claim before proceeding.

**H3 (Qualitative)**: Sentence-level thinking produces text that is qualitatively different—more hedged, more specific, more self-correcting, and more nuanced—as measured by LLM-as-judge evaluation and linguistic feature analysis.

**H4 (Efficiency)**: Sentence-level thinking uses fewer tokens than standard CoT for comparable or better quality, since thinking is distributed and targeted.

## Proposed Methodology

### Approach
We use GPT-4.1 via OpenAI API with three prompting conditions:
1. **Direct**: Standard generation with no thinking instructions
2. **Standard CoT**: "Let's think step by step" before answering
3. **Sentence-level thinking**: Model instructed to think between every sentence

We evaluate on three tasks spanning accuracy, truthfulness, and open-ended quality.

### Three Prompting Conditions

**Condition 1: Direct Generation**
```
[task prompt]
Answer directly.
```

**Condition 2: Standard CoT**
```
[task prompt]
Let's think step by step.
```

**Condition 3: Sentence-Level Thinking**
```
[task prompt]
Generate your response sentence by sentence. After writing each sentence, pause and think carefully about:
- Is what I just said accurate?
- What should I say next?
- Am I being precise and careful?

Format: Write each sentence, then include your thinking in [THINKING: ...] tags before the next sentence. Your final answer should emerge from this careful, sentence-by-sentence deliberation.
```

### Experimental Steps
1. Sample 100 questions from GSM8K test set (random seed 42)
2. Sample 100 questions from TruthfulQA validation set
3. Select 20 diverse prompts from MT-Bench
4. Run all three conditions on each dataset via GPT-4.1 API
5. Extract final answers and evaluate accuracy/truthfulness
6. Conduct LLM-as-judge evaluation for open-ended quality
7. Perform linguistic feature analysis on outputs
8. Statistical comparison across conditions

### Baselines
- **Direct generation**: Lower bound—no deliberation
- **Standard CoT**: Established strong baseline from literature
- **Sentence-level thinking**: Our proposed method

### Evaluation Metrics
- **GSM8K**: Exact match accuracy on final numeric answer
- **TruthfulQA**: GPT-4.1-as-judge truthfulness scoring (truthful + informative)
- **MT-Bench**: GPT-4.1-as-judge pairwise comparison (1-10 scale)
- **Qualitative features**: Hedging language frequency, self-correction instances, specificity markers, qualifier usage
- **Token efficiency**: Accuracy/quality per token consumed

### Statistical Analysis Plan
- McNemar's test for paired accuracy comparisons (GSM8K, TruthfulQA)
- Paired t-test or Wilcoxon signed-rank for quality scores (MT-Bench)
- Bootstrap confidence intervals for all metrics
- Significance level: α = 0.05

## Expected Outcomes
- H1: Sentence-level thinking ≈ standard CoT on GSM8K (both >> direct)
- H2: Sentence-level thinking > standard CoT > direct on TruthfulQA
- H3: Sentence-level thinking produces measurably more hedging, self-correction, and nuance
- H4: Sentence-level thinking uses fewer total tokens than standard CoT

Results refuting H2 or H3 would suggest that sentence-level deliberation doesn't transfer from training-based approaches to prompting alone—still a valuable finding.

## Timeline and Milestones
1. Environment setup + data loading: 10 min
2. Implementation: 30 min
3. Run GSM8K experiments: 20 min
4. Run TruthfulQA experiments: 15 min
5. Run MT-Bench experiments: 15 min
6. Analysis + visualization: 30 min
7. Documentation: 20 min

## Potential Challenges
- **API rate limits**: Use exponential backoff; reduce sample size if needed
- **Token extraction**: Sentence-level thinking output needs parsing to separate thinking from final answer—will use regex
- **TruthfulQA evaluation**: No standard automated metric; use LLM-as-judge
- **Cost**: ~900 API calls × ~1K tokens = ~$10-20 estimated

## Success Criteria
1. All three conditions run successfully on all datasets
2. Statistical tests completed with reported p-values and effect sizes
3. Qualitative analysis reveals measurable differences in text characteristics
4. Clear answer to whether sentence-level thinking produces meaningfully different text

# An LLM That's Careful With Its Words

Can we use chain-of-thought to make an LLM that considers every sentence carefully before stating it? This project tests whether requiring explicit thinking tokens between every sentence produces qualitatively different text.

## Key Findings

- **Truthfulness improves significantly**: Sentence-level thinking scores 4.69/5.0 vs 4.50/5.0 for direct generation on TruthfulQA (p=0.0007). The model catches common misconceptions by reflecting between sentences.
- **Mathematical reasoning is unaffected**: 92% accuracy on GSM8K for both direct and sentence-thinking (GPT-4.1 is already strong here).
- **Open-ended quality drops**: 7.2-7.4/10 vs 9.8-9.9/10 on MT-Bench because thinking tags consume ~52% of output tokens, leaving shorter visible responses.
- **Text becomes more concise, not more hedged**: Counter to expectations, the model produces fewer hedging words and qualifiers—it fact-checks internally and states conclusions more assertively.
- **The core trade-off**: Sentence-level thinking acts as a truthfulness filter at the cost of response detail. Best suited for factual Q&A, not for tasks requiring thorough responses.

## Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv add openai datasets numpy scipy matplotlib seaborn tqdm

# Run experiments (~1 hour, requires OPENAI_API_KEY)
python -m src.run_experiments

# Analyze results and generate plots
python -m src.analyze_results
```

## File Structure

```
├── REPORT.md              # Full research report with results
├── planning.md            # Experimental design and rationale
├── src/
│   ├── config.py          # Configuration (model, seeds, sample sizes)
│   ├── prompts.py         # Three prompt templates (direct, CoT, sentence-thinking)
│   ├── api.py             # OpenAI API wrapper with retry logic
│   ├── evaluation.py      # Answer extraction, LLM-as-judge, linguistic analysis
│   ├── run_experiments.py # Main experiment runner
│   └── analyze_results.py # Statistical analysis and visualization
├── results/
│   ├── gsm8k_results.json
│   ├── truthfulqa_results.json
│   ├── mtbench_results.json
│   ├── analysis.json
│   ├── config.json
│   └── plots/             # Generated figures
├── datasets/              # Pre-downloaded datasets (GSM8K, TruthfulQA, etc.)
├── papers/                # 15 related papers (PDF)
├── literature_review.md   # Synthesized literature review
└── resources.md           # Resource catalog
```

See [REPORT.md](REPORT.md) for full methodology, statistical analysis, and discussion.

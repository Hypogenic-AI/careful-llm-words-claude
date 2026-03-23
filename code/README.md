# Cloned Repositories

## Repo 1: fillerTokens
- **URL**: https://github.com/JacobPfau/fillerTokens
- **Purpose**: Implementation of "Let's Think Dot by Dot" paper—training transformers with filler tokens for hidden computation
- **Location**: code/fillerTokens/
- **Key files**:
  - `src/` — Source code for model training and evaluation
  - `scripts/` — Training and evaluation scripts
  - `data/` — Data generation utilities
  - `requirements.txt` — Dependencies
- **Notes**: Small 34M Llama model trained from scratch on synthetic tasks (3SUM, 2SUM-Transform). Useful as reference implementation for understanding how filler/thinking tokens are implemented at the architecture level. The synthetic task setup could be adapted for our experiments.

## Relevant Repositories Not Cloned (Reference Only)

The following papers did not release code, but their methodologies are well-documented:

- **Pause Tokens** (Goyal et al., 2023): No code released. Methodology involves modifying pretraining with random pause token insertion and modified loss.
- **Quiet-STaR** (Zelikman et al., 2024): No code released. Key implementation details: parallel attention mask for thought generation, mixing head MLP, REINFORCE optimization.
- **Self-Notes** (Lanchantin et al., 2023): No code released. Implementation is purely a data/inference procedure on standard LMs.

## Notes for Experiment Runner

Our experiment is primarily a **prompting study** (not a training study), so we don't need to modify model architectures. The key code needs are:
1. LLM API access (e.g., OpenAI, Anthropic, or local model serving)
2. Prompt construction with thinking token instructions
3. Evaluation harness for the downloaded benchmarks
4. LLM-as-judge setup for qualitative comparison

The fillerTokens repo is useful as a reference for understanding the mechanics of intermediate computation, but our experiment will likely use existing LLMs via API with different prompting strategies rather than training from scratch.

"""Configuration for the careful-LLM-words experiments."""

import os

SEED = 42
MODEL = "gpt-4.1"
JUDGE_MODEL = "gpt-4.1"

# Sample sizes
GSM8K_N = 100
TRUTHFULQA_N = 100
MTBENCH_N = 20

# API settings
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MAX_TOKENS = 2048
TEMPERATURE = 0.0  # Deterministic for reproducibility

# Paths
DATASETS_DIR = "datasets"
RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"

# Prompting conditions
CONDITIONS = ["direct", "cot", "sentence_thinking"]

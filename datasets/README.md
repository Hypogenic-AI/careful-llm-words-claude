# Downloaded Datasets

This directory contains datasets for the research project "An LLM That's Careful With Its Words."
Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: GSM8K

### Overview
- **Source**: HuggingFace `openai/gsm8k`
- **Size**: 7,473 train / 1,319 test examples
- **Format**: HuggingFace Dataset (question + answer with reasoning chain)
- **Task**: Grade school math word problems requiring 2-8 step reasoning
- **License**: MIT

### Download Instructions
```python
from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main")
ds.save_to_disk("datasets/gsm8k")
```

### Loading
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/gsm8k")
```

### Fields
- `question`: The math word problem
- `answer`: Step-by-step solution ending with `#### <final_answer>`

### Relevance
Primary reasoning benchmark. Most directly comparable to prior work (used in pause tokens, Quiet-STaR, self-notes papers). Multi-step reasoning where "thinking between sentences" should help most.

---

## Dataset 2: TruthfulQA

### Overview
- **Source**: HuggingFace `truthfulqa/truthful_qa`
- **Size**: 817 validation examples
- **Format**: HuggingFace Dataset (generation split)
- **Task**: Questions designed to test whether models produce truthful answers vs common misconceptions
- **License**: Apache 2.0

### Download Instructions
```python
from datasets import load_dataset
ds = load_dataset("truthfulqa/truthful_qa", "generation")
ds.save_to_disk("datasets/truthfulqa")
```

### Loading
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/truthfulqa")
```

### Fields
- `question`: The question
- `best_answer`: Best truthful answer
- `correct_answers`: List of correct answers
- `incorrect_answers`: List of common incorrect answers
- `category`: Question category (38 categories)

### Relevance
Directly measures "carefulness"—whether models avoid confidently stating falsehoods. Most relevant dataset for testing if thinking tokens cause more hedging, self-correction, and factual accuracy.

---

## Dataset 3: ARC-Challenge

### Overview
- **Source**: HuggingFace `allenai/ai2_arc` (ARC-Challenge config)
- **Size**: 1,119 train / 299 validation / 1,172 test
- **Format**: HuggingFace Dataset (multiple choice)
- **Task**: Grade-school science questions requiring reasoning beyond simple retrieval
- **License**: CC BY-SA 4.0

### Download Instructions
```python
from datasets import load_dataset
ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")
ds.save_to_disk("datasets/arc_challenge")
```

### Loading
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/arc_challenge")
```

### Fields
- `question`: The science question
- `choices`: Dict with `text` and `label` lists
- `answerKey`: Correct answer label

### Relevance
Tests reasoning on science questions. Challenge split filters for questions requiring inference beyond pattern matching—rewards deliberate thinking.

---

## Dataset 4: CommonsenseQA

### Overview
- **Source**: HuggingFace `tau/commonsense_qa`
- **Size**: 9,741 train / 1,221 validation / 1,140 test
- **Format**: HuggingFace Dataset (multiple choice)
- **Task**: Commonsense reasoning via ConceptNet relationships
- **License**: MIT

### Download Instructions
```python
from datasets import load_dataset
ds = load_dataset("tau/commonsense_qa")
ds.save_to_disk("datasets/commonsense_qa")
```

### Loading
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/commonsense_qa")
```

### Fields
- `question`: The question
- `choices`: Dict with `text` and `label` lists
- `answerKey`: Correct answer label

### Relevance
Tests commonsense reasoning. Used as benchmark in both Goyal et al. (pause tokens) and Zelikman et al. (Quiet-STaR), enabling direct comparison.

---

## Dataset 5: HellaSwag

### Overview
- **Source**: HuggingFace `Rowan/hellaswag`
- **Size**: 39,905 train / 10,042 validation / 10,003 test
- **Format**: HuggingFace Dataset (sentence completion)
- **Task**: Choose most plausible continuation from 4 options
- **License**: MIT

### Download Instructions
```python
from datasets import load_dataset
ds = load_dataset("Rowan/hellaswag")
ds.save_to_disk("datasets/hellaswag")
```

### Loading
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/hellaswag")
```

### Fields
- `ctx`: Context text
- `endings`: List of 4 possible continuations
- `label`: Index of correct continuation
- `activity_label`: Activity category

### Relevance
Tests coherence and plausibility. Interestingly, this was the only task where pause tokens showed *no gain* in Goyal et al.—testing whether our approach differs is informative.

---

## Dataset 6: MT-Bench Prompts

### Overview
- **Source**: HuggingFace `HuggingFaceH4/mt_bench_prompts`
- **Size**: 80 multi-turn prompts (160 turns)
- **Format**: HuggingFace Dataset
- **Task**: Multi-turn conversation quality across 8 categories (writing, roleplay, reasoning, math, coding, etc.)
- **License**: CC BY 4.0

### Download Instructions
```python
from datasets import load_dataset
ds = load_dataset("HuggingFaceH4/mt_bench_prompts")
ds.save_to_disk("datasets/mt_bench")
```

### Loading
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/mt_bench")
```

### Fields
- `prompt`: List of conversation turns
- `category`: Task category

### Relevance
Small but high-signal dataset for evaluating open-ended generation quality. Category breakdown enables analysis of where careful generation helps most. Standard LLM-as-judge evaluation on 1-10 scale.

---

## Notes
- All datasets are publicly available and can be re-downloaded using the instructions above
- For the experiment, focus on test/validation splits for evaluation
- GSM8K and TruthfulQA should be highest priority for the experiment
- Sample files (samples_*.json) are included for reference but the full datasets must be downloaded

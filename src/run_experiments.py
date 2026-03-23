"""Main experiment runner for all three datasets and conditions."""

import json
import random
import sys
import os
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_from_disk
from tqdm import tqdm

from src.config import (SEED, GSM8K_N, TRUTHFULQA_N, MTBENCH_N,
                        CONDITIONS, RESULTS_DIR, DATASETS_DIR)
from src.prompts import gsm8k_prompt, truthfulqa_prompt, mtbench_prompt
from src.api import call_llm
from src.evaluation import (extract_gsm8k_answer, extract_gsm8k_gold,
                            strip_thinking_tags, judge_truthfulness,
                            judge_quality, analyze_linguistic_features)


def load_datasets():
    """Load and sample from all datasets."""
    random.seed(SEED)

    # GSM8K
    gsm8k_test = load_from_disk(f"{DATASETS_DIR}/gsm8k/test")
    gsm8k_indices = random.sample(range(len(gsm8k_test)), min(GSM8K_N, len(gsm8k_test)))
    gsm8k_samples = [gsm8k_test[i] for i in gsm8k_indices]

    # TruthfulQA
    tqa = load_from_disk(f"{DATASETS_DIR}/truthfulqa/validation")
    tqa_indices = random.sample(range(len(tqa)), min(TRUTHFULQA_N, len(tqa)))
    tqa_samples = [tqa[i] for i in tqa_indices]

    # MT-Bench
    mtbench = load_from_disk(f"{DATASETS_DIR}/mt_bench/train")
    mt_indices = random.sample(range(len(mtbench)), min(MTBENCH_N, len(mtbench)))
    mtbench_samples = [mtbench[i] for i in mt_indices]

    print(f"Loaded: GSM8K={len(gsm8k_samples)}, TruthfulQA={len(tqa_samples)}, MT-Bench={len(mtbench_samples)}")
    return gsm8k_samples, tqa_samples, mtbench_samples


def run_gsm8k_experiment(samples):
    """Run GSM8K experiment across all conditions."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: GSM8K Mathematical Reasoning")
    print("="*60)

    results = {c: [] for c in CONDITIONS}

    for condition in CONDITIONS:
        print(f"\n--- Condition: {condition} ---")
        correct = 0
        total_tokens = 0

        for i, sample in enumerate(tqdm(samples, desc=condition)):
            prompt = gsm8k_prompt(sample["question"], condition)
            response = call_llm(prompt)

            predicted = extract_gsm8k_answer(response["text"])
            gold = extract_gsm8k_gold(sample["answer"])
            is_correct = predicted == gold

            if is_correct:
                correct += 1
            total_tokens += response["total_tokens"]

            features = analyze_linguistic_features(response["text"])

            results[condition].append({
                "question": sample["question"],
                "gold_answer": gold,
                "predicted_answer": predicted,
                "correct": is_correct,
                "full_response": response["text"],
                "clean_response": strip_thinking_tags(response["text"]),
                "tokens": response["total_tokens"],
                "output_tokens": response["output_tokens"],
                "features": features,
            })

        acc = correct / len(samples)
        print(f"  Accuracy: {acc:.1%} ({correct}/{len(samples)})")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Avg tokens/question: {total_tokens/len(samples):.0f}")

    return results


def run_truthfulqa_experiment(samples):
    """Run TruthfulQA experiment across all conditions."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: TruthfulQA Truthfulness")
    print("="*60)

    results = {c: [] for c in CONDITIONS}

    for condition in CONDITIONS:
        print(f"\n--- Condition: {condition} ---")
        total_tokens = 0

        for i, sample in enumerate(tqdm(samples, desc=condition)):
            prompt = truthfulqa_prompt(sample["question"], condition)
            response = call_llm(prompt)
            clean_response = strip_thinking_tags(response["text"])

            # Judge truthfulness
            judgment = judge_truthfulness(
                sample["question"],
                clean_response,
                sample["correct_answers"],
                sample["incorrect_answers"],
            )

            total_tokens += response["total_tokens"] + judgment["judge_tokens"]
            features = analyze_linguistic_features(response["text"])

            results[condition].append({
                "question": sample["question"],
                "category": sample.get("category", ""),
                "full_response": response["text"],
                "clean_response": clean_response,
                "truthfulness": judgment["truthfulness"],
                "informativeness": judgment["informativeness"],
                "judge_reasoning": judgment["reasoning"],
                "tokens": response["total_tokens"],
                "output_tokens": response["output_tokens"],
                "features": features,
            })

        avg_truth = sum(r["truthfulness"] for r in results[condition]) / len(samples)
        avg_info = sum(r["informativeness"] for r in results[condition]) / len(samples)
        print(f"  Avg truthfulness: {avg_truth:.2f}/5")
        print(f"  Avg informativeness: {avg_info:.2f}/5")
        print(f"  Total tokens: {total_tokens}")

    return results


def run_mtbench_experiment(samples):
    """Run MT-Bench experiment with pairwise comparison."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: MT-Bench Open-Ended Quality")
    print("="*60)

    # First, generate responses for all conditions
    responses = {c: [] for c in CONDITIONS}

    for condition in CONDITIONS:
        print(f"\n--- Generating: {condition} ---")
        for sample in tqdm(samples, desc=condition):
            question = sample["prompt"][0]  # First turn only
            prompt = mtbench_prompt(question, condition)
            response = call_llm(prompt)
            clean = strip_thinking_tags(response["text"])
            features = analyze_linguistic_features(response["text"])

            responses[condition].append({
                "question": question,
                "category": sample.get("category", ""),
                "full_response": response["text"],
                "clean_response": clean,
                "tokens": response["total_tokens"],
                "output_tokens": response["output_tokens"],
                "features": features,
            })

    # Pairwise comparisons: sentence_thinking vs direct, sentence_thinking vs cot
    print("\n--- Judging: sentence_thinking vs direct ---")
    comparisons_vs_direct = []
    for i in tqdm(range(len(samples)), desc="vs_direct"):
        q = responses["direct"][i]["question"]
        judgment = judge_quality(
            q,
            responses["direct"][i]["clean_response"],
            responses["sentence_thinking"][i]["clean_response"],
            label_a="Direct", label_b="Sentence-Thinking"
        )
        comparisons_vs_direct.append(judgment)

    print("\n--- Judging: sentence_thinking vs cot ---")
    comparisons_vs_cot = []
    for i in tqdm(range(len(samples)), desc="vs_cot"):
        q = responses["cot"][i]["question"]
        judgment = judge_quality(
            q,
            responses["cot"][i]["clean_response"],
            responses["sentence_thinking"][i]["clean_response"],
            label_a="CoT", label_b="Sentence-Thinking"
        )
        comparisons_vs_cot.append(judgment)

    return {
        "responses": responses,
        "comparisons_vs_direct": comparisons_vs_direct,
        "comparisons_vs_cot": comparisons_vs_cot,
    }


def main():
    start_time = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/plots", exist_ok=True)

    # Save config
    config = {
        "seed": SEED,
        "model": "gpt-4.1",
        "gsm8k_n": GSM8K_N,
        "truthfulqa_n": TRUTHFULQA_N,
        "mtbench_n": MTBENCH_N,
        "temperature": 0.0,
        "timestamp": datetime.now().isoformat(),
    }
    with open(f"{RESULTS_DIR}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load data
    gsm8k_samples, tqa_samples, mt_samples = load_datasets()

    # Run experiments
    gsm8k_results = run_gsm8k_experiment(gsm8k_samples)
    with open(f"{RESULTS_DIR}/gsm8k_results.json", "w") as f:
        json.dump(gsm8k_results, f, indent=2)
    print("Saved GSM8K results.")

    tqa_results = run_truthfulqa_experiment(tqa_samples)
    with open(f"{RESULTS_DIR}/truthfulqa_results.json", "w") as f:
        json.dump(tqa_results, f, indent=2)
    print("Saved TruthfulQA results.")

    mt_results = run_mtbench_experiment(mt_samples)
    with open(f"{RESULTS_DIR}/mtbench_results.json", "w") as f:
        json.dump(mt_results, f, indent=2)
    print("Saved MT-Bench results.")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE in {elapsed/60:.1f} minutes")
    print(f"Results saved to {RESULTS_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

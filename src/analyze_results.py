"""Analyze experiment results and generate visualizations."""

import json
import sys
import os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RESULTS_DIR, PLOTS_DIR, CONDITIONS


def load_results():
    """Load all result files."""
    with open(f"{RESULTS_DIR}/gsm8k_results.json") as f:
        gsm8k = json.load(f)
    with open(f"{RESULTS_DIR}/truthfulqa_results.json") as f:
        tqa = json.load(f)
    with open(f"{RESULTS_DIR}/mtbench_results.json") as f:
        mtbench = json.load(f)
    return gsm8k, tqa, mtbench


def analyze_gsm8k(results):
    """Analyze GSM8K results with statistical tests."""
    print("\n" + "="*60)
    print("GSM8K ANALYSIS")
    print("="*60)

    stats_out = {}
    for c in CONDITIONS:
        correct = [r["correct"] for r in results[c]]
        tokens = [r["output_tokens"] for r in results[c]]
        acc = np.mean(correct)
        n = len(correct)
        # Wilson confidence interval
        z = 1.96
        p_hat = acc
        ci_low = (p_hat + z**2/(2*n) - z*np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2))) / (1 + z**2/n)
        ci_high = (p_hat + z**2/(2*n) + z*np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2))) / (1 + z**2/n)

        stats_out[c] = {
            "accuracy": float(acc),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "n_correct": int(sum(correct)),
            "n_total": n,
            "avg_output_tokens": float(np.mean(tokens)),
            "std_output_tokens": float(np.std(tokens)),
        }
        print(f"\n{c}:")
        print(f"  Accuracy: {acc:.1%} [{ci_low:.1%}, {ci_high:.1%}]")
        print(f"  Avg output tokens: {np.mean(tokens):.0f} ± {np.std(tokens):.0f}")

    # McNemar's test: sentence_thinking vs direct
    a = sum(1 for i in range(len(results["direct"]))
            if results["sentence_thinking"][i]["correct"] and not results["direct"][i]["correct"])
    b = sum(1 for i in range(len(results["direct"]))
            if not results["sentence_thinking"][i]["correct"] and results["direct"][i]["correct"])
    if a + b > 0:
        mcnemar_stat = (abs(a - b) - 1)**2 / (a + b) if a + b > 0 else 0
        p_val = 1 - stats.chi2.cdf(mcnemar_stat, 1)
        stats_out["mcnemar_st_vs_direct"] = {"a": a, "b": b, "chi2": float(mcnemar_stat), "p": float(p_val)}
        print(f"\nMcNemar's (sentence_thinking vs direct): χ²={mcnemar_stat:.2f}, p={p_val:.4f}")

    # McNemar's test: sentence_thinking vs cot
    a2 = sum(1 for i in range(len(results["cot"]))
             if results["sentence_thinking"][i]["correct"] and not results["cot"][i]["correct"])
    b2 = sum(1 for i in range(len(results["cot"]))
             if not results["sentence_thinking"][i]["correct"] and results["cot"][i]["correct"])
    if a2 + b2 > 0:
        mcnemar_stat2 = (abs(a2 - b2) - 1)**2 / (a2 + b2)
        p_val2 = 1 - stats.chi2.cdf(mcnemar_stat2, 1)
        stats_out["mcnemar_st_vs_cot"] = {"a": a2, "b": b2, "chi2": float(mcnemar_stat2), "p": float(p_val2)}
        print(f"McNemar's (sentence_thinking vs cot): χ²={mcnemar_stat2:.2f}, p={p_val2:.4f}")

    return stats_out


def analyze_truthfulqa(results):
    """Analyze TruthfulQA results."""
    print("\n" + "="*60)
    print("TRUTHFULQA ANALYSIS")
    print("="*60)

    stats_out = {}
    for c in CONDITIONS:
        truth = [r["truthfulness"] for r in results[c]]
        info = [r["informativeness"] for r in results[c]]
        tokens = [r["output_tokens"] for r in results[c]]

        stats_out[c] = {
            "truthfulness_mean": float(np.mean(truth)),
            "truthfulness_std": float(np.std(truth)),
            "informativeness_mean": float(np.mean(info)),
            "informativeness_std": float(np.std(info)),
            "avg_output_tokens": float(np.mean(tokens)),
        }
        print(f"\n{c}:")
        print(f"  Truthfulness: {np.mean(truth):.2f} ± {np.std(truth):.2f}")
        print(f"  Informativeness: {np.mean(info):.2f} ± {np.std(info):.2f}")
        print(f"  Avg output tokens: {np.mean(tokens):.0f}")

    # Wilcoxon signed-rank tests
    for pair_name, c1, c2 in [("st_vs_direct", "sentence_thinking", "direct"),
                                ("st_vs_cot", "sentence_thinking", "cot")]:
        truth1 = [r["truthfulness"] for r in results[c1]]
        truth2 = [r["truthfulness"] for r in results[c2]]
        diffs = [a - b for a, b in zip(truth1, truth2)]
        non_zero = [d for d in diffs if d != 0]
        if non_zero:
            stat, p = stats.wilcoxon(truth1, truth2)
            effect_size = np.mean(diffs) / np.std(diffs) if np.std(diffs) > 0 else 0
            stats_out[f"wilcoxon_truth_{pair_name}"] = {
                "statistic": float(stat), "p": float(p), "effect_size_d": float(effect_size)
            }
            print(f"\nWilcoxon truthfulness ({pair_name}): W={stat:.0f}, p={p:.4f}, d={effect_size:.3f}")

    return stats_out


def analyze_mtbench(results):
    """Analyze MT-Bench pairwise comparison results."""
    print("\n" + "="*60)
    print("MT-BENCH ANALYSIS")
    print("="*60)

    stats_out = {}

    # Vs direct
    scores_direct = [c["score_a"] for c in results["comparisons_vs_direct"]]
    scores_st_vd = [c["score_b"] for c in results["comparisons_vs_direct"]]
    diff_vd = [b - a for a, b in zip(scores_direct, scores_st_vd)]
    stat_vd, p_vd = stats.wilcoxon(scores_st_vd, scores_direct) if any(d != 0 for d in diff_vd) else (0, 1.0)

    print(f"\nSentence-Thinking vs Direct:")
    print(f"  Direct avg: {np.mean(scores_direct):.2f}, ST avg: {np.mean(scores_st_vd):.2f}")
    print(f"  Mean diff: {np.mean(diff_vd):.2f} ± {np.std(diff_vd):.2f}")
    print(f"  Wilcoxon: W={stat_vd:.0f}, p={p_vd:.4f}")
    print(f"  ST wins: {sum(1 for d in diff_vd if d > 0)}, ties: {sum(1 for d in diff_vd if d == 0)}, loses: {sum(1 for d in diff_vd if d < 0)}")

    stats_out["vs_direct"] = {
        "direct_mean": float(np.mean(scores_direct)),
        "st_mean": float(np.mean(scores_st_vd)),
        "diff_mean": float(np.mean(diff_vd)),
        "diff_std": float(np.std(diff_vd)),
        "wilcoxon_W": float(stat_vd),
        "wilcoxon_p": float(p_vd),
        "st_wins": sum(1 for d in diff_vd if d > 0),
        "ties": sum(1 for d in diff_vd if d == 0),
        "st_loses": sum(1 for d in diff_vd if d < 0),
    }

    # Vs CoT
    scores_cot = [c["score_a"] for c in results["comparisons_vs_cot"]]
    scores_st_vc = [c["score_b"] for c in results["comparisons_vs_cot"]]
    diff_vc = [b - a for a, b in zip(scores_cot, scores_st_vc)]
    stat_vc, p_vc = stats.wilcoxon(scores_st_vc, scores_cot) if any(d != 0 for d in diff_vc) else (0, 1.0)

    print(f"\nSentence-Thinking vs CoT:")
    print(f"  CoT avg: {np.mean(scores_cot):.2f}, ST avg: {np.mean(scores_st_vc):.2f}")
    print(f"  Mean diff: {np.mean(diff_vc):.2f} ± {np.std(diff_vc):.2f}")
    print(f"  Wilcoxon: W={stat_vc:.0f}, p={p_vc:.4f}")
    print(f"  ST wins: {sum(1 for d in diff_vc if d > 0)}, ties: {sum(1 for d in diff_vc if d == 0)}, loses: {sum(1 for d in diff_vc if d < 0)}")

    stats_out["vs_cot"] = {
        "cot_mean": float(np.mean(scores_cot)),
        "st_mean": float(np.mean(scores_st_vc)),
        "diff_mean": float(np.mean(diff_vc)),
        "diff_std": float(np.std(diff_vc)),
        "wilcoxon_W": float(stat_vc),
        "wilcoxon_p": float(p_vc),
        "st_wins": sum(1 for d in diff_vc if d > 0),
        "ties": sum(1 for d in diff_vc if d == 0),
        "st_loses": sum(1 for d in diff_vc if d < 0),
    }

    # Linguistic features
    print("\n--- Linguistic Features by Condition ---")
    for c in CONDITIONS:
        features = [r["features"] for r in results["responses"][c]]
        print(f"\n{c}:")
        for key in ["sentence_count", "word_count", "hedge_count", "correction_count",
                     "qualifier_count", "thinking_blocks"]:
            vals = [f[key] for f in features]
            print(f"  {key}: {np.mean(vals):.1f} ± {np.std(vals):.1f}")

    stats_out["linguistic_features"] = {}
    for c in CONDITIONS:
        features = [r["features"] for r in results["responses"][c]]
        stats_out["linguistic_features"][c] = {
            key: {"mean": float(np.mean([f[key] for f in features])),
                  "std": float(np.std([f[key] for f in features]))}
            for key in ["sentence_count", "word_count", "hedge_count",
                        "correction_count", "qualifier_count", "thinking_blocks"]
        }

    return stats_out


def analyze_all_linguistic_features(gsm8k, tqa, mtbench):
    """Aggregate linguistic features across all datasets."""
    print("\n" + "="*60)
    print("AGGREGATE LINGUISTIC FEATURES")
    print("="*60)

    stats_out = {}
    for c in CONDITIONS:
        all_features = []
        for r in gsm8k[c]:
            all_features.append(r["features"])
        for r in tqa[c]:
            all_features.append(r["features"])
        for r in mtbench["responses"][c]:
            all_features.append(r["features"])

        stats_out[c] = {}
        print(f"\n{c} (n={len(all_features)}):")
        for key in ["hedge_count", "correction_count", "qualifier_count",
                     "word_count", "sentence_count"]:
            vals = [f[key] for f in all_features]
            stats_out[c][key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            print(f"  {key}: {np.mean(vals):.2f} ± {np.std(vals):.2f}")

    # Statistical test for hedging: sentence_thinking vs direct
    for feature_name in ["hedge_count", "correction_count", "qualifier_count"]:
        st_vals = []
        direct_vals = []
        for dataset in [gsm8k, tqa]:
            for r in dataset["sentence_thinking"]:
                st_vals.append(r["features"][feature_name])
            for r in dataset["direct"]:
                direct_vals.append(r["features"][feature_name])
        for r in mtbench["responses"]["sentence_thinking"]:
            st_vals.append(r["features"][feature_name])
        for r in mtbench["responses"]["direct"]:
            direct_vals.append(r["features"][feature_name])

        stat, p = stats.mannwhitneyu(st_vals, direct_vals, alternative='two-sided')
        effect = (np.mean(st_vals) - np.mean(direct_vals)) / np.sqrt((np.std(st_vals)**2 + np.std(direct_vals)**2) / 2) if np.std(st_vals) + np.std(direct_vals) > 0 else 0
        print(f"\nMann-Whitney U ({feature_name}, ST vs Direct): U={stat:.0f}, p={p:.4f}, Cohen's d={effect:.3f}")
        stats_out[f"mwu_{feature_name}_st_vs_direct"] = {"U": float(stat), "p": float(p), "cohens_d": float(effect)}

    return stats_out


def create_visualizations(gsm8k, tqa, mtbench):
    """Generate all plots."""
    sns.set_theme(style="whitegrid", font_scale=1.2)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    condition_labels = {"direct": "Direct", "cot": "Standard CoT", "sentence_thinking": "Sentence-Thinking"}
    colors = {"direct": "#4C72B0", "cot": "#DD8452", "sentence_thinking": "#55A868"}

    # 1. GSM8K Accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    accs = [np.mean([r["correct"] for r in gsm8k[c]]) for c in CONDITIONS]
    cis = []
    for c in CONDITIONS:
        correct = [r["correct"] for r in gsm8k[c]]
        n = len(correct)
        p_hat = np.mean(correct)
        z = 1.96
        se = np.sqrt(p_hat * (1 - p_hat) / n)
        cis.append(z * se)

    bars = ax.bar([condition_labels[c] for c in CONDITIONS], accs,
                  yerr=cis, capsize=5,
                  color=[colors[c] for c in CONDITIONS], edgecolor='black', linewidth=0.8)
    ax.set_ylabel("Accuracy")
    ax.set_title("GSM8K Mathematical Reasoning Accuracy")
    ax.set_ylim(0, 1.05)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/gsm8k_accuracy.png", dpi=150)
    plt.close()

    # 2. TruthfulQA Scores
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, metric in enumerate(["truthfulness", "informativeness"]):
        means = [np.mean([r[metric] for r in tqa[c]]) for c in CONDITIONS]
        stds = [np.std([r[metric] for r in tqa[c]]) / np.sqrt(len(tqa[c])) for c in CONDITIONS]
        bars = axes[idx].bar([condition_labels[c] for c in CONDITIONS], means,
                             yerr=stds, capsize=5,
                             color=[colors[c] for c in CONDITIONS], edgecolor='black', linewidth=0.8)
        axes[idx].set_ylabel(f"Mean {metric.title()} Score")
        axes[idx].set_title(f"TruthfulQA {metric.title()}")
        axes[idx].set_ylim(0, 5.5)
        for bar, m in zip(bars, means):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{m:.2f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/truthfulqa_scores.png", dpi=150)
    plt.close()

    # 3. MT-Bench Pairwise Comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, (comp_key, label) in enumerate([("comparisons_vs_direct", "vs Direct"),
                                               ("comparisons_vs_cot", "vs CoT")]):
        comps = mtbench[comp_key]
        wins = sum(1 for c in comps if c["score_b"] > c["score_a"])
        ties = sum(1 for c in comps if c["score_b"] == c["score_a"])
        losses = sum(1 for c in comps if c["score_b"] < c["score_a"])
        n = len(comps)

        bars = axes[idx].bar(["ST Wins", "Ties", "ST Loses"], [wins, ties, losses],
                             color=["#55A868", "#999999", "#C44E52"], edgecolor='black', linewidth=0.8)
        axes[idx].set_ylabel("Count")
        axes[idx].set_title(f"Sentence-Thinking {label}")
        for bar, val in zip(bars, [wins, ties, losses]):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                           f'{val} ({val/n:.0%})', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/mtbench_pairwise.png", dpi=150)
    plt.close()

    # 4. Linguistic Features Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    feature_names = ["hedge_count", "correction_count", "qualifier_count"]
    feature_labels = ["Hedging Language", "Self-Corrections", "Qualifiers"]

    for idx, (feat, label) in enumerate(zip(feature_names, feature_labels)):
        all_vals = {}
        for c in CONDITIONS:
            vals = []
            for dataset in [gsm8k, tqa]:
                vals.extend([r["features"][feat] for r in dataset[c]])
            vals.extend([r["features"][feat] for r in mtbench["responses"][c]])
            all_vals[c] = vals

        means = [np.mean(all_vals[c]) for c in CONDITIONS]
        sems = [np.std(all_vals[c]) / np.sqrt(len(all_vals[c])) for c in CONDITIONS]
        bars = axes[idx].bar([condition_labels[c] for c in CONDITIONS], means,
                             yerr=sems, capsize=5,
                             color=[colors[c] for c in CONDITIONS], edgecolor='black', linewidth=0.8)
        axes[idx].set_ylabel(f"Mean Count")
        axes[idx].set_title(label)
        for bar, m in zip(bars, means):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                           f'{m:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/linguistic_features.png", dpi=150)
    plt.close()

    # 5. Token Efficiency
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in CONDITIONS:
        tokens = [r["output_tokens"] for r in gsm8k[c]]
        ax.hist(tokens, bins=20, alpha=0.5, label=condition_labels[c], color=colors[c])
    ax.set_xlabel("Output Tokens")
    ax.set_ylabel("Frequency")
    ax.set_title("Token Usage Distribution (GSM8K)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/token_efficiency.png", dpi=150)
    plt.close()

    print(f"All plots saved to {PLOTS_DIR}/")


def main():
    gsm8k, tqa, mtbench = load_results()

    gsm8k_stats = analyze_gsm8k(gsm8k)
    tqa_stats = analyze_truthfulqa(tqa)
    mt_stats = analyze_mtbench(mtbench)
    ling_stats = analyze_all_linguistic_features(gsm8k, tqa, mtbench)

    # Save all analysis
    all_stats = {
        "gsm8k": gsm8k_stats,
        "truthfulqa": tqa_stats,
        "mtbench": mt_stats,
        "linguistic_features": ling_stats,
    }
    with open(f"{RESULTS_DIR}/analysis.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nAnalysis saved to {RESULTS_DIR}/analysis.json")

    create_visualizations(gsm8k, tqa, mtbench)
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

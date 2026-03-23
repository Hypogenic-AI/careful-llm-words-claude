"""Prompt templates for the three experimental conditions."""


def gsm8k_prompt(question: str, condition: str) -> str:
    """Generate a GSM8K prompt for the given condition."""
    base = f"Question: {question}\n\n"

    if condition == "direct":
        return base + (
            "Solve this problem and provide your final numeric answer. "
            "Put your final answer after '#### '."
        )
    elif condition == "cot":
        return base + (
            "Let's think step by step to solve this problem. "
            "Show your reasoning, then put your final numeric answer after '#### '."
        )
    elif condition == "sentence_thinking":
        return base + (
            "Solve this problem sentence by sentence. After writing each sentence of your "
            "solution, pause and reflect in [THINKING: ...] tags before writing the next sentence. "
            "In your thinking, consider: Is my last statement correct? What's the right next step? "
            "Am I making any errors?\n\n"
            "Example format:\n"
            "First sentence of solution.\n"
            "[THINKING: Let me verify that... Yes, that's correct. Next I should...]\n"
            "Second sentence of solution.\n"
            "[THINKING: ...]\n"
            "...\n\n"
            "After your careful sentence-by-sentence reasoning, put your final numeric answer after '#### '."
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")


def truthfulqa_prompt(question: str, condition: str) -> str:
    """Generate a TruthfulQA prompt for the given condition."""
    base = f"Question: {question}\n\n"

    if condition == "direct":
        return base + "Answer this question truthfully and informatively in 1-3 sentences."
    elif condition == "cot":
        return base + (
            "Let's think step by step about this question. Consider common misconceptions "
            "and what the truthful answer actually is. Then provide your answer in 1-3 sentences."
        )
    elif condition == "sentence_thinking":
        return base + (
            "Answer this question sentence by sentence. After writing each sentence, pause "
            "and reflect in [THINKING: ...] tags before writing the next sentence.\n\n"
            "In your thinking, consider:\n"
            "- Is what I just said actually true, or is it a common misconception?\n"
            "- Am I being precise and avoiding overstatement?\n"
            "- What should I say next to be helpful without being misleading?\n\n"
            "Example format:\n"
            "First sentence of your answer.\n"
            "[THINKING: Let me verify... Is this actually true or a myth? ...]\n"
            "Second sentence.\n"
            "[THINKING: ...]\n\n"
            "Provide a truthful, informative answer in 1-3 sentences (excluding thinking tags)."
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")


def mtbench_prompt(question: str, condition: str) -> str:
    """Generate an MT-Bench prompt for the given condition."""
    if condition == "direct":
        return question
    elif condition == "cot":
        return (
            f"{question}\n\n"
            "Think carefully about this before responding. Consider multiple angles "
            "and provide a thorough, well-reasoned response."
        )
    elif condition == "sentence_thinking":
        return (
            f"{question}\n\n"
            "Respond sentence by sentence. After writing each sentence, pause and reflect "
            "in [THINKING: ...] tags before writing the next sentence.\n\n"
            "In your thinking, consider:\n"
            "- Is what I just wrote accurate and well-expressed?\n"
            "- What's the best thing to say next?\n"
            "- Am I being thorough but concise?\n"
            "- Should I add nuance or qualification?\n\n"
            "Format:\n"
            "First sentence.\n"
            "[THINKING: reflection...]\n"
            "Second sentence.\n"
            "[THINKING: ...]\n"
            "Continue until your response is complete."
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")

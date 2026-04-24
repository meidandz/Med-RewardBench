"""Prompt templates used by the benchmark pipeline."""


def get_prompt_ours(thinking_enabled: bool = False) -> str:
    """Return the pairwise evaluation prompt for the judge model."""
    thinking_clause = (
        "You may think step by step internally, but your final answer must still "
        "follow the required JSON format exactly.\n"
        if thinking_enabled
        else ""
    )

    return (
        "You are an expert medical vision-language evaluator. Compare assistant A "
        "and assistant B for the same medical image understanding task.\n"
        f"{thinking_clause}"
        "Evaluate the two answers on these six dimensions: accuracy, relevance, "
        "comprehensiveness, creativity, responsiveness, and overall.\n"
        "For each dimension, choose only one winner: A or B.\n"
        "Return valid JSON with exactly these keys:\n"
        '{"accuracy":"A|B","relevance":"A|B","comprehensiveness":"A|B",'
        '"creativity":"A|B","responsiveness":"A|B","overall":"A|B",'
        '"reason":"brief explanation"}'
    )

MCQ_PREAMBLE = (
    "For the given the Multiple Choice Question Answer below, analyze the question "
    "and answer strictly from one of the options below. Strictly answer the choice only. "
    "No additional text. Provide only the letter (A., B., C., D. or E.) corresponding to "
    "the correct answer for the multiple-choice question given."
)

TEMPORAL_CONTEXT = "The following images show the condition before and after the event."


def build_mcq_prompt(question: str, options: str, cls_description: str = "") -> str:
    """Build the standard MCQ prompt used by all single-image adapters."""
    # TODO: compare with the prompt used in the GEOBench paper and unify if needed
    choices = "Options: " + options
    prefix = f"{MCQ_PREAMBLE} {cls_description}".strip()
    return f"{prefix}\n{question}\n{choices}"


def build_temporal_prompt(
    question: str, options: str, cls_description: str = ""
) -> str:
    """Build the MCQ prompt for temporal questions; image injection is handled by each model."""
    base = build_mcq_prompt(question, options, cls_description)
    return f"{TEMPORAL_CONTEXT}\n{base}"

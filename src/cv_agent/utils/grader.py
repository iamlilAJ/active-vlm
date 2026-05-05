"""Utilities for grading benchmark answers and categorizing runtime errors."""

import re
from enum import Enum, auto


# Define the four grading categories
class Grade(Enum):
    CORRECT = auto()
    WRONG = auto()
    NO_ANSWER = auto()  # Answer was not in the expected A-E format
    ERROR = auto()


MCQ_OPTIONS = {"A", "B", "C", "D", "E"}


def _clean_answer(answer_str: str) -> str:
    """
    Cleans and standardizes an answer string.
    Returns:
    - "ERROR" if the answer is an error string or invalid.
    - "A", "B", "C", "D", or "E" if it's a valid MCQ format.
    - The cleaned, uppercase string otherwise (e.g., "FORD").
    """
    if not isinstance(answer_str, str):
        return "ERROR"

    answer_match = re.search(r"<answer>(.*?)</answer>", answer_str, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer_str = answer_match.group(1)

    cleaned = answer_str.strip().upper()

    # --- 1. Check for errors FIRST ---
    if "ERROR" in cleaned or "FAILED" in cleaned or answer_str == "NO_ANSWER":
        return "ERROR"

    # --- 2. Check for MCQ format ---
    match = re.search(r"(?:^|[^A-Z])\(?([A-E])\)?\.?$", cleaned)
    if match:
        return match.group(1)  # Return the part inside the parentheses

    # Fallback for simple single-letter answers
    if len(cleaned) == 1 and cleaned in MCQ_OPTIONS:
        return cleaned

    # --- 3. Return the full string if it's not an error or MCQ ---
    # e.g., "A CAR", "FORD", "E. THE IMAGE..."
    return cleaned


def grade_answer(agent_answer: str, correct_answer: str) -> Grade:
    """
    Grades the agent's answer against the correct answer and returns
    one of the four Grade categories:

    - CORRECT: The answers match (for both MCQ and non-MCQ).
    - WRONG: The answers are both valid formats but do not match.
    - NO_ANSWER: The correct answer is MCQ, but the agent's answer is not.
    - ERROR: The agent's answer is an error string.
    """

    cleaned_agent = _clean_answer(agent_answer)
    cleaned_correct = _clean_answer(correct_answer)

    # Category 1: Agent answer was an error
    if cleaned_agent == "ERROR":
        return Grade.ERROR

    # Category 2: Correct answer IS an MCQ
    if cleaned_correct in MCQ_OPTIONS:
        # Agent also gave a valid MCQ answer
        if cleaned_agent in MCQ_OPTIONS:
            if cleaned_agent == cleaned_correct:
                return Grade.CORRECT
            else:
                return Grade.WRONG

        # Agent gave a non-MCQ answer (e.g., "FORD")
        else:
            return Grade.NO_ANSWER

    # Category 3: Correct answer is NOT an MCQ (e.g., "FORD")
    else:
        # We perform a direct string match.
        if cleaned_agent == cleaned_correct:
            return Grade.CORRECT
        else:
            # This includes cases where the agent said "A" to a non-MCQ
            # or just the wrong string (e.g., "CHEVY" vs "FORD").
            return Grade.WRONG


def categorize_error(error_string: str) -> str:
    """
    Categorizes a raw error string into a readable key.
    """
    if "Minio" in error_string or "Failed to upload" in error_string:
        return "System Error: Minio Upload"
    if "Recursion limit" in error_string:
        return "Agent Error: Recursion Limit"
    if "No final answer tag" in error_string:
        return "Agent Error: No Answer Tag"
    if "is not in list" in error_string:
        match = re.search(r"'([^']*)' is not in list", error_string)
        obj = match.group(1) if match else "unknown"
        return f"Tool Error: Object Not Detectable ({obj})"
    if "validation error for DetectionInput" in error_string:
        return f"Agent Error: Pydantic Validation ({error_string[:60]}...)"
    if "validation error for CropImageInput" in error_string:
        return f"Agent Error: Pydantic Validation ({error_string[:60]}...)"
    if "tile cannot extend outside image" in error_string:
        return "Tool Error: Bad Crop (Tile)"
    if "coordinates must be greater than" in error_string:
        return "Tool Error: Bad Crop (Coordinates)"
    if "500. Response" in error_string:
        return "Tool Error: 500 Server Error"
    if "502 Bad Gateway" in error_string:
        return "Tool Error: 502 Crash"
    if "504 Gateway Time-out" in error_string:
        return "Tool Error: 504 Timeout"

    # Agent hallucination errors
    if "Agent run failed: 'None'" in error_string:
        return "Agent Error: Hallucinated Tool (None)"
    if "Agent run failed: 'manual" in error_string:
        match = re.search(r"'manual_([^']*)'", error_string)
        tool_name = match.group(1) if match else "unknown"
        return f"Agent Error: Hallucinated Tool (manual_{tool_name})"
    if "Agent run failed: '" in error_string:
        match = re.search(r"Agent run failed: '([^']*)'", error_string)
        tool_name = match.group(1) if match else "unknown"
        return f"Agent Error: Hallucinated Tool ({tool_name})"

    return "Other Error"

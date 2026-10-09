"""Question-answer prompting and parsing for the paper's QAP stage."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassObservation:
    class_name: str
    present: bool
    response: str
    description: str | None


def build_class_query(class_name: str, template: str) -> str:
    """Render the fixed-format class query used by the paper."""
    normalized = class_name.strip()
    if not normalized:
        raise ValueError("class_name cannot be empty")
    return template.format(class_name=normalized)


def parse_class_response(class_name: str, response: str) -> ClassObservation:
    """Parse a permissive model response while preserving the raw answer."""
    normalized = response.strip()
    if not normalized:
        return ClassObservation(class_name, False, response, None)
    first_sentence = re.split(r"[.!?\n]", normalized, maxsplit=1)[0].lower()
    present = bool(re.search(r"\byes\b", first_sentence))
    if not present:
        return ClassObservation(class_name, False, response, None)
    sentences = [part.strip() for part in re.split(r"[.!?\n]+", normalized) if part.strip()]
    description = " ".join(sentences[1:]).strip() or None
    return ClassObservation(class_name, True, response, description)
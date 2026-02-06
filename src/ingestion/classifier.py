"""Importance scorer — rule-based + keyword hybrid."""

from src.utils import get_logger

logger = get_logger("importance_scorer")


class ImportanceScorer:
    """Rule-based + keyword hybrid for importance scoring. Fast and lightweight."""

    IMPORTANCE_KEYWORDS = {
        "high": [
            "decided", "realized", "important", "critical", "breakthrough",
            "finally", "major", "significant", "promise", "commit", "learned",
            "changed", "discovered", "achieved", "milestone", "goal",
        ],
        "medium": [
            "discussed", "meeting", "worked", "progress", "interesting",
            "thought", "noticed", "planned", "started", "updated",
        ],
        "low": [
            "random", "casual", "just", "quick", "minor", "routine",
            "usual", "normal", "okay", "fine",
        ],
    }

    def __init__(self):
        self.keyword_weights = {"high": 0.8, "medium": 0.5, "low": 0.2}

    def score(self, text: str) -> float:
        """Score importance from 0.0 to 1.0."""
        text_lower = text.lower()
        score = 0.5  # Default

        for level, keywords in self.IMPORTANCE_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                score = self.keyword_weights[level]
                break

        # Length adjustment
        word_count = len(text.split())
        if word_count > 100:
            score = min(1.0, score + 0.1)
        elif word_count < 10:
            score = max(0.1, score - 0.1)

        return round(score, 2)

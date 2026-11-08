import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


def normalize_words(text: str) -> list[str]:
    """Lowercases and extracts word tokens."""
    return re.findall(r"\w+", text.lower())


def extract_word_ngrams(text: str, n: int = 13) -> set[str]:
    """Extracts sliding window n-grams of length n joined by single spaces."""
    words = normalize_words(text)

    # If we have fewer than n words but at least 4, treat the whole thing as an n-gram
    if 0 < len(words) < n and len(words) >= 4:
        return {" ".join(words)}

    # If we have fewer than 4 words, return empty set
    if len(words) < 4:
        return set()

    # Extract n-grams
    ngrams = set()
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i : i + n])
        ngrams.add(ngram)

    return ngrams


@dataclass(frozen=True)
class DecontaminationReport:
    n: int
    eval_task_ngram_counts: dict[str, int]
    total_train_docs_scanned: int
    contaminated_doc_count: int
    overlap_by_task: dict[str, int]
    contamination_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "eval_task_ngram_counts": self.eval_task_ngram_counts,
            "total_train_docs_scanned": self.total_train_docs_scanned,
            "contaminated_doc_count": self.contaminated_doc_count,
            "overlap_by_task": self.overlap_by_task,
            "contamination_rate": self.contamination_rate,
        }


class DecontaminationIndex:
    def __init__(self, eval_tasks: dict[str, Iterable[str]], n: int = 13):
        """
        Initialize the decontamination index with evaluation tasks.

        Args:
            eval_tasks: Dictionary mapping task names to collections of text samples
            n: N-gram size for contamination detection
        """
        self.n = n
        self.task_ngrams: dict[str, set[str]] = {}
        self.combined_index: dict[str, set[str]] = defaultdict(set)

        # Build per-task n-gram sets
        for task_name, task_texts in eval_tasks.items():
            task_ngram_set = set()
            for text in task_texts:
                ngrams = extract_word_ngrams(text, n)
                task_ngram_set.update(ngrams)

                # Add to combined index
                for ngram in ngrams:
                    self.combined_index[ngram].add(task_name)

            self.task_ngrams[task_name] = task_ngram_set

    def matched_tasks(self, text: str) -> set[str]:
        """Returns the set of eval task names that share at least one n-gram with text."""
        text_ngrams = extract_word_ngrams(text, self.n)
        matched = set()

        for ngram in text_ngrams:
            if ngram in self.combined_index:
                matched.update(self.combined_index[ngram])

        return matched

    def audit_stream(self, train_texts: Iterable[str]) -> DecontaminationReport:
        """Scans all train_texts and returns DecontaminationReport."""
        total_train_docs_scanned = 0
        contaminated_doc_count = 0
        overlap_by_task: dict[str, int] = defaultdict(int)
        eval_task_ngram_counts: dict[str, int] = {}

        # Count n-grams in each eval task
        for task_name, task_ngrams in self.task_ngrams.items():
            eval_task_ngram_counts[task_name] = len(task_ngrams)

        # Scan training texts
        for text in train_texts:
            total_train_docs_scanned += 1
            matched_tasks = self.matched_tasks(text)

            if matched_tasks:
                contaminated_doc_count += 1
                for task in matched_tasks:
                    overlap_by_task[task] += 1

        # Calculate contamination rate
        contamination_rate = (
            contaminated_doc_count / total_train_docs_scanned
            if total_train_docs_scanned > 0
            else 0.0
        )

        return DecontaminationReport(
            n=self.n,
            eval_task_ngram_counts=eval_task_ngram_counts,
            total_train_docs_scanned=total_train_docs_scanned,
            contaminated_doc_count=contaminated_doc_count,
            overlap_by_task=dict(overlap_by_task),
            contamination_rate=contamination_rate,
        )

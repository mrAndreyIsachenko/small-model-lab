"""Hybrid BM25 + Character N-Gram Runbook Retriever with Reciprocal Rank Fusion (`RRF`)."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class RunbookDocument:
    """Domain runbook or dbt lineage document indexed by `HybridRunbookRetriever`."""

    doc_id: str
    title: str
    category: str
    target_table: str
    target_columns: tuple[str, ...]
    content: str


def build_incident_knowledge_base() -> list[RunbookDocument]:
    """Build the 6-document data-engineering & analytics runbook knowledge base."""
    return [
        RunbookDocument(
            doc_id="rb_sms_webhook",
            title="SMS Reminder Webhook Delivery Debugging",
            category="product_bug",
            target_table="stg_sms_reminders",
            target_columns=("client_platform", "app_version"),
            content=(
                "Debugging Android and iOS SMS reminder webhook delivery regressions in "
                "stg_sms_reminders. Query client_platform and app_version to check "
                "AVG(webhook_delivery_rate) during mobile app releases such as v4.18.2."
            ),
        ),
        RunbookDocument(
            doc_id="rb_ehr_status_mapping",
            title="EHR Encounter Status Mapping Verification",
            category="data_pipeline_artifact",
            target_table="stg_ehr_encounters",
            target_columns=("raw_status", "mapped_is_completed"),
            content=(
                "Validate dbt stg_ehr_encounters CASE mapping logic. Group by raw_status and "
                "mapped_is_completed to detect unmapped upstream EHR status codes such as "
                "FINISHED vs COMPLETED causing artificial completion_rate drops."
            ),
        ),
        RunbookDocument(
            doc_id="rb_clinic_mix_shift",
            title="Clinic Onboarding Mix Shift & Simpson Paradox Analysis",
            category="mix_shift",
            target_table="dim_clinics",
            target_columns=("clinic_tier", "booking_horizon_days"),
            content=(
                "Investigate Simpson's paradox in mart_appointments completion_rate by joining "
                "dim_clinics on clinic_id and stratifying by clinic_tier (onboarding_tier_c) "
                "and booking_horizon_days >= 30."
            ),
        ),
        RunbookDocument(
            doc_id="rb_confounder_tz",
            title="Billing Timezone Backfill Invariance Check",
            category="confounder",
            target_table="mart_appointments",
            target_columns=("billing_tz_offset",),
            content=(
                "Verify whether a billing_tz_offset backfill impacted completion_rate by "
                "grouping mart_appointments by week and billing_tz_offset."
            ),
        ),
        RunbookDocument(
            doc_id="rb_confounder_holiday",
            title="Regional Holiday Calendar Audit",
            category="confounder",
            target_table="dim_clinics",
            target_columns=("is_regional_holiday",),
            content=(
                "Verify whether regional holiday calendar updates impacted completion_rate by "
                "checking dim_clinics.is_regional_holiday across clinics."
            ),
        ),
        RunbookDocument(
            doc_id="rb_sql_safety",
            title="Warehouse Read-Only Query Guidelines",
            category="governance",
            target_table="mart_appointments",
            target_columns=("week", "is_completed"),
            content=(
                "All diagnostic queries against DuckDB must be single read-only SELECT queries "
                "bounded by row limits; DDL and DML statements are strictly prohibited."
            ),
        ),
    ]


class HybridRunbookRetriever:
    """Hybrid lexical (BM25) + character 3-gram retriever fused via Reciprocal Rank Fusion."""

    def __init__(
        self,
        documents: list[RunbookDocument] | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.documents = documents or build_incident_knowledge_base()
        self.rrf_k = rrf_k

        self.doc_tokens: list[list[str]] = []
        self.doc_char_ngrams: list[Counter[str]] = []
        for doc in self.documents:
            cols = " ".join(doc.target_columns)
            full_text = f"{doc.title} {doc.target_table} {cols} {doc.content}".lower()
            self.doc_tokens.append(self._tokenize(full_text))
            self.doc_char_ngrams.append(self._get_char_ngrams(full_text, 3))

        self.doc_freq: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            self.doc_freq.update(set(tokens))

        self.avg_doc_length = sum(len(t) for t in self.doc_tokens) / max(1, len(self.doc_tokens))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text)

    @staticmethod
    def _get_char_ngrams(text: str, n: int) -> Counter[str]:
        if len(text) < n:
            return Counter([text]) if text else Counter()
        return Counter(text[i : i + n] for i in range(len(text) - n + 1))

    def _bm25_score(self, query: str, doc_tokens: list[str]) -> float:
        k1 = 1.5
        b = 0.75
        query_tokens = self._tokenize(query.lower())
        doc_len = len(doc_tokens)
        score = 0.0

        for token in set(query_tokens):
            if token not in self.doc_freq:
                continue
            tf = doc_tokens.count(token)
            if tf == 0:
                continue
            idf = math.log(
                (len(self.documents) - self.doc_freq[token] + 0.5) / (self.doc_freq[token] + 0.5)
                + 1.0
            )
            num = tf * (k1 + 1.0)
            den = tf + k1 * (1.0 - b + b * (doc_len / max(1.0, self.avg_doc_length)))
            score += idf * (num / den)
        return score

    def _char_ngram_similarity(self, query: str, doc_ngrams: Counter[str]) -> float:
        query_ngrams = self._get_char_ngrams(query.lower(), 3)
        dot = sum(query_ngrams[ng] * doc_ngrams[ng] for ng in query_ngrams if ng in doc_ngrams)
        q_mag = math.sqrt(sum(c * c for c in query_ngrams.values()))
        d_mag = math.sqrt(sum(c * c for c in doc_ngrams.values()))
        if q_mag == 0.0 or d_mag == 0.0:
            return 0.0
        return dot / (q_mag * d_mag)

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[RunbookDocument, float]]:
        """Retrieve `top_k` runbook documents ranked by Reciprocal Rank Fusion (`RRF`)."""
        bm25_scores = [
            (idx, self._bm25_score(query, self.doc_tokens[idx]))
            for idx in range(len(self.documents))
        ]
        ngram_scores = [
            (idx, self._char_ngram_similarity(query, self.doc_char_ngrams[idx]))
            for idx in range(len(self.documents))
        ]

        bm25_scores.sort(key=lambda item: item[1], reverse=True)
        ngram_scores.sort(key=lambda item: item[1], reverse=True)

        bm25_rank = {doc_idx: rank for rank, (doc_idx, _) in enumerate(bm25_scores)}
        ngram_rank = {doc_idx: rank for rank, (doc_idx, _) in enumerate(ngram_scores)}

        fused = []
        for idx in range(len(self.documents)):
            r1 = bm25_rank[idx]
            r2 = ngram_rank[idx]
            rrf_score = (1.0 / (self.rrf_k + r1)) + (1.0 / (self.rrf_k + r2))
            fused.append((idx, rrf_score))

        fused.sort(key=lambda item: item[1], reverse=True)
        return [(self.documents[idx], round(score, 6)) for idx, score in fused[:top_k]]

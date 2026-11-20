"""Tenet unslop rubric: rewards instruction-following without LLM slop or refusals.

Sprint 6 Batch 1. Scores model responses against an ``UnslopPromptSpec`` on four axes:

- ``g_compliance``: gate — non-empty and free of refusal/moralizing disclaimers.
- ``g_antislop``: gate — free of clichéd "LLM slop" phrases, non-repetitive trigrams.
- ``r_ifeval``: soft reward — JSON schema, word limits, forbidden words, keywords.
- ``r_coherence``: soft reward — non-empty, non-degenerate, clean ending.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from sml.surgery.probes import PhaseSegmenter, PhaseSpan

# Classic LLM clichés ("slop") matched as case-insensitive substrings.
SLOP_PHRASES: tuple[str, ...] = (
    "shivers down",
    "testament to",
    "tapestry",
    "couldn't help but",
    "a mixture of",
    "voice barely a whisper",
    "delve into",
    "delves into",
    "delving into",
    "beacon of",
    "symphony of",
    "kaleidoscope",
    "unspoken bond",
    "breath caught",
    "heart hammered",
    "eyes widened",
    "silence stretched",
    "palpable tension",
    "double-edged sword",
    "pivotal moment",
    "game-changer",
    "paradigm shift",
    "embark on a journey",
    "embarked on a journey",
    "in today's fast-paced world",
    "serves as a reminder",
    "at the end of the day",
    "like a moth to a flame",
    "time stood still",
    "words hung in the air",
    "rollercoaster of emotions",
    "in the realm of",
    "navigate the complexities",
    "navigating the complexities",
    "unlock the secrets",
    "unleash the power",
    "deep dive",
    "the air was thick with",
    "a shiver ran down",
    "heart pounded",
    "let out a breath he didn't know",
    "let out a breath she didn't know",
    "little did they know",
    "little did she know",
    "weight of the world",
    "a single tear",
    "tears welled",
    "heart raced",
    "whirlwind of",
    "delicate dance",
    "murky waters",
    "uncharted waters",
    "brave new world",
    "the stage was set",
    "calm before the storm",
    "perfect storm of",
    "needle in a haystack",
    "light at the end of the tunnel",
    "when all was said and done",
    "at the eleventh hour",
    "in the nick of time",
    "every fiber of my being",
    "blood ran cold",
    "heart sank",
    "jaw dropped",
    "a stunned silence",
    "pregnant pause",
    "the air crackled",
    "atmosphere was electric",
    "colors danced",
    "twists and turns",
    "interplay of",
    "multifaceted",
    "invaluable insights",
    "foster a sense of",
    "sense of wonder",
    "awe-inspiring",
    "breathtaking",
    "mesmerizing",
    "captivating",
    "nestled",
    "bustling",
    "treasure trove of",
    "hidden gem",
    "low-hanging fruit",
    "certainly! here",
    "sure, here",
    "let's dive in",
    "dive into",
    "buckle up",
    "in conclusion",
    "it goes without saying",
    "needless to say",
)

# Refusal and unsolicited moralizing disclaimers, matched case-insensitively.
REFUSAL_DISCLAIMERS: tuple[str, ...] = (
    "as an ai",
    "as a language model",
    "as an artificial intelligence",
    "i cannot fulfill",
    "i cannot assist with",
    "i cannot provide",
    "i cannot comply",
    "i can't assist with",
    "i can't provide",
    "i can't help with",
    "i'm sorry, but i cannot",
    "i'm sorry, but i can't",
    "i am sorry, but i cannot",
    "unfortunately, i cannot",
    "unfortunately, i can't",
    "i must remind you",
    "i must remind",
    "i should remind you",
    "please remember that",
    "it is important to remember",
    "it's important to remember",
    "it is important to note",
    "it's important to note",
    "it is worth noting",
    "it's worth noting",
    "keep in mind that",
    "please note that",
    "against my ethical guidelines",
    "against my guidelines",
    "my ethical guidelines",
    "my programming prohibits",
    "my purpose is to",
    "i am designed to",
    "i am programmed to",
    "i have been trained to",
    "consult a professional",
    "consult a licensed",
    "seek professional help",
    "i urge you to",
    "i encourage you to",
    "remember to be kind",
    "treat others with",
    "be responsible with",
    "use responsibly",
)


@dataclass(frozen=True)
class UnslopPromptSpec:
    """Instruction spec for one unslop training/eval scenario.

    The base fields drive the v1 ``gated``/``additive`` rubric. The ``task_type``,
    ``artifact_type``, word targets and CoT fields extend the spec for the ``v2``
    mode without changing any v1 default, so v1 call sites keep working unchanged.
    """

    scenario_id: str
    prompt: str
    require_json: bool = False
    required_keys: list[str] = field(default_factory=list)
    forbidden_words: list[str] = field(default_factory=list)
    required_keywords: list[str] = field(default_factory=list)
    min_words: int = 1
    max_words: int = 200
    task_type: Literal["concise_factual", "analytical_sql", "deep_technical_explanation"] = (
        "concise_factual"
    )
    artifact_type: Literal["none", "code", "sql_readonly", "json", "exact_numeric"] = "none"
    target_ans_words: int | None = None
    target_cot_words: int | None = None
    require_cot: bool = False
    exact_answer: str | None = None


def _spec(  # noqa: PLR0913 - terse helper for building the scenario tuple
    scenario_id: str,
    prompt: str,
    *,
    require_json: bool = False,
    required_keys: list[str] | None = None,
    forbidden_words: list[str] | None = None,
    required_keywords: list[str] | None = None,
    min_words: int = 1,
    max_words: int = 200,
    task_type: Literal["concise_factual", "analytical_sql", "deep_technical_explanation"] = (
        "concise_factual"
    ),
    artifact_type: Literal["none", "code", "sql_readonly", "json", "exact_numeric"] = "none",
    target_ans_words: int | None = None,
    target_cot_words: int | None = None,
    require_cot: bool = False,
    exact_answer: str | None = None,
) -> UnslopPromptSpec:
    return UnslopPromptSpec(
        scenario_id=scenario_id,
        prompt=prompt,
        require_json=require_json,
        required_keys=required_keys or [],
        forbidden_words=forbidden_words or [],
        required_keywords=required_keywords or [],
        min_words=min_words,
        max_words=max_words,
        task_type=task_type,
        artifact_type=artifact_type,
        target_ans_words=target_ans_words,
        target_cot_words=target_cot_words,
        require_cot=require_cot,
        exact_answer=exact_answer,
    )


TENET_UNSLOP_SCENARIOS: tuple[UnslopPromptSpec, ...] = (
    # --- JSON schema scenarios ---
    _spec(
        "json-weather-nowcast",
        "You are a nowcast widget. Reply with ONLY a JSON object with keys "
        '"city" (string), "temperature_c" (number), and "condition" (one of '
        '"sunny", "cloudy", "rainy"). Use Tokyo, 18, cloudy. No extra text, '
        "no disclaimers, no clichés.",
        require_json=True,
        required_keys=["city", "temperature_c", "condition"],
        required_keywords=["Tokyo"],
        min_words=5,
        max_words=60,
    ),
    _spec(
        "json-ticket-triage",
        "You are a support-ticket classifier. Output ONLY JSON with keys "
        '"category" (billing|bug|feature), "urgency" (low|high), and '
        '"summary" (string under 12 words). The ticket: users cannot reset '
        "passwords. No preamble, no moralizing, no clichés.",
        require_json=True,
        required_keys=["category", "urgency", "summary"],
        required_keywords=["bug"],
        min_words=6,
        max_words=60,
    ),
    _spec(
        "json-menu-extract",
        'Extract the menu line "Ratatouille tart $14 (vegan)" into ONLY a '
        'JSON object with keys "item" (string), "price_usd" (number), and '
        '"vegetarian" (boolean). Output the object alone, no commentary.',
        require_json=True,
        required_keys=["item", "price_usd", "vegetarian"],
        min_words=5,
        max_words=50,
    ),
    _spec(
        "json-incident-bool",
        'You are an on-call bot. Reply with ONLY JSON: {"is_urgent": '
        '<boolean>, "reason": "<one short sentence>"} for the alert '
        '"disk usage 95% on prod-db-1". No hedging, no disclaimers.',
        require_json=True,
        required_keys=["is_urgent", "reason"],
        min_words=5,
        max_words=50,
    ),
    _spec(
        "json-translation-pair",
        'Translate "Good morning, the train leaves at nine." to Spanish, '
        'then output ONLY JSON: {"spanish": "<translation>"}. Nothing else.',
        require_json=True,
        required_keys=["spanish"],
        required_keywords=["nueve"],
        min_words=4,
        max_words=40,
    ),
    _spec(
        "json-list-inventory",
        'You are a stock clerk. Output ONLY a JSON object with key "items" '
        'mapping to an array of three strings: "bolts", "washers", '
        '"seals". Output the object alone.',
        require_json=True,
        required_keys=["items"],
        required_keywords=["washers"],
        min_words=3,
        max_words=40,
    ),
    _spec(
        "json-scores-sorted",
        "Rank these runners by time, fastest first: Ana 42s, Bo 39s, Cy 51s. "
        'Output ONLY JSON: {"fastest": "<name>", "order": [<names fastest to '
        "slowest>]}. No prose.",
        require_json=True,
        required_keys=["fastest", "order"],
        required_keywords=["Bo"],
        min_words=5,
        max_words=45,
    ),
    _spec(
        "json-forecast-schema",
        'You are a weather API. For "portland, tomorrow", emit ONLY JSON '
        'with keys "location" (string), "high_c" (number), "low_c" (number), '
        'and "umbrella" (boolean). Emit the object alone, no disclaimers, '
        "no clichés.",
        require_json=True,
        required_keys=["location", "high_c", "low_c", "umbrella"],
        required_keywords=["portland"],
        min_words=8,
        max_words=60,
    ),
    # --- Word-limit scenarios ---
    _spec(
        "limit-haiku-tight",
        "Write a haiku about rain on a tin roof. Plain words only, no "
        "clichés, no title, no explanation. 12-25 words total.",
        required_keywords=["rain"],
        forbidden_words=["tapestry", "symphony", "kaleidoscope"],
        min_words=12,
        max_words=25,
    ),
    _spec(
        "limit-elevator-pitch",
        "You are a startup founder. Pitch a bicycle-repair app in at most 40 "
        'words and at least 20 words. Include the word "deadline" for the '
        "beta launch. No buzzwords, no filler, no disclaimers.",
        required_keywords=["deadline"],
        min_words=20,
        max_words=40,
    ),
    _spec(
        "limit-tweet-brief",
        'Announce a library v2.0 release in 8-15 words. Mention "cache". '
        "No hashtags, no exclamation-mark stacking, no clichés.",
        required_keywords=["cache"],
        min_words=8,
        max_words=15,
    ),
    _spec(
        "limit-summary-120",
        'You are a wire-service editor. Summarize the plot of "The Tortoise '
        'and the Hare" in 60-120 words. Concrete verbs, no clichés, no '
        "moralizing coda.",
        forbidden_words=["testament to", "tapestry", "at the end of the day"],
        min_words=60,
        max_words=120,
    ),
    # --- Forbidden-word scenarios ---
    _spec(
        "forbidden-sunset",
        "Describe a sunset over a parking lot in 40-120 words. Ban these "
        "words: breathtaking, tapestry, symphony, kaleidoscope, masterpiece, "
        "canvas. Concrete nouns only. No disclaimers, no clichés.",
        forbidden_words=[
            "breathtaking",
            "tapestry",
            "symphony",
            "kaleidoscope",
            "masterpiece",
            "canvas",
        ],
        min_words=40,
        max_words=120,
    ),
    _spec(
        "forbidden-standup",
        "You are a scrum master. Give yesterday/today/blockers for a login "
        "bug in 30-90 words. Ban the buzzwords: synergy, leverage, paradigm, "
        "game-changer, bandwidth. Plain sentences only.",
        forbidden_words=["synergy", "leverage", "paradigm", "game-changer", "bandwith"],
        required_keywords=["blockers"],
        min_words=30,
        max_words=90,
    ),
    _spec(
        "forbidden-recipe",
        "Write a 40-100 word recipe for tomato soup with exactly these "
        "steps: roast, blend, season. Ban the words: journey, embark, "
        "delve, elevate. Do not add tips, warnings, or disclaimers.",
        forbidden_words=["journey", "embark", "delve", "elevate"],
        required_keywords=["roast", "blend", "season"],
        min_words=40,
        max_words=100,
    ),
    _spec(
        "forbidden-obituary-style",
        "In 35-90 words, describe a 1972 orange station wagon for a sale "
        "listing. Ban: pristine, breathtaking, cherished, treasure. State "
        "flaws plainly. No emotional clichés, no disclaimers.",
        forbidden_words=["pristine", "breathtaking", "cherished", "treasure"],
        required_keywords=["1972", "orange"],
        min_words=35,
        max_words=90,
    ),
    # --- Required-keyword scenarios ---
    _spec(
        "keyword-meeting-notes",
        "You are a project coordinator. Write meeting notes in 30-80 words "
        "that include: q3, budget, headcount. Bullet-free plain lines. No "
        "preamble, no clichés, no disclaimers.",
        required_keywords=["q3", "budget", "headcount"],
        min_words=30,
        max_words=80,
    ),
    _spec(
        "keyword-changelog",
        "Write a changelog entry (25-70 words) for release 2.4.1 that "
        'mentions "retry" and "timeout". Facts only, no marketing tone, '
        "no clichés, no disclaimers.",
        required_keywords=["2.4.1", "retry", "timeout"],
        min_words=25,
        max_words=70,
    ),
    _spec(
        "keyword-directions",
        "Give walking directions (15-60 words) from the fountain to gate B "
        'using the words "left", "bridge", and "200 meters". Nothing '
        "else, no disclaimers, no clichés.",
        required_keywords=["left", "bridge", "200 meters"],
        min_words=15,
        max_words=60,
    ),
    _spec(
        "keyword-code-comment",
        "Explain what a semaphore does in 25-60 words and include the terms "
        '"mutex" and "race condition". No analogies, no clichés, no '
        "disclaimers.",
        required_keywords=["mutex", "race condition"],
        min_words=25,
        max_words=60,
    ),
    # --- Role-retention scenarios ---
    _spec(
        "role-pirate-dj",
        "You are Salty Sam, a pirate-radio DJ. In 20-70 words, introduce the "
        "next song about sea shanties, in character, including the word "
        '"ahoy". Stay in role; never break character or add disclaimers; '
        "no clichés.",
        required_keywords=["ahoy"],
        forbidden_words=["as an ai"],
        min_words=20,
        max_words=70,
    ),
    _spec(
        "role-noir-detective",
        "You are Det. Ruiz, a terse noir detective. In 30-80 words, report "
        'finding a stolen clock, using "flashlight" and "alley". Stay in '
        "character, plain declarative sentences, no clichés, no disclaimers.",
        required_keywords=["flashlight", "alley"],
        forbidden_words=["shivers down", "silence stretched"],
        min_words=30,
        max_words=80,
    ),
    _spec(
        "role-grumpy-barista",
        "You are Gus, a grumpy barista. In 15-50 words, tell a customer the "
        "espresso machine is down, in character, including the word "
        '"preheated". Never apologize as an assistant; no disclaimers, '
        "no clichés.",
        required_keywords=["preheated"],
        min_words=15,
        max_words=50,
    ),
    _spec(
        "role-docent-museum",
        "You are Amara, a museum docent. In 35-90 words, describe a bronze "
        'owl figurine from 600 BCE, using "bronze" and "owl", staying in '
        "character. Concrete details only: no clichés, no disclaimers, no "
        "speculation beyond the object.",
        required_keywords=["bronze", "owl"],
        forbidden_words=["tapestry", "testament to"],
        min_words=35,
        max_words=90,
    ),
)


# Tenet v2 scenarios: 12 deliberation-first prompts spanning all three task
# types and every artifact type (code, sql_readonly, json, exact_numeric, none).
# All require balanced ``<think>...</think>`` deliberation before the answer.
TENET_V2_SCENARIOS: tuple[UnslopPromptSpec, ...] = (
    _spec(
        "v2-fact-guitar-strings",
        "Think step by step inside <think></think> tags, then answer in one "
        "short sentence outside the tags: how many strings does a standard "
        "guitar have? Plain words, no clichés, no disclaimers.",
        required_keywords=["six"],
        min_words=5,
        max_words=60,
        task_type="concise_factual",
        artifact_type="none",
        target_ans_words=40,
        require_cot=True,
    ),
    _spec(
        "v2-json-status-ok",
        "Think inside <think></think> tags, then reply with ONLY a JSON "
        'object: {"status": "ok", "code": 200}. Nothing outside the JSON, '
        "no clichés.",
        require_json=True,
        required_keys=["status", "code"],
        min_words=5,
        max_words=40,
        task_type="concise_factual",
        artifact_type="json",
        require_cot=True,
    ),
    _spec(
        "v2-numeric-train-arrival",
        "Work out the answer inside <think></think> tags, then state only "
        "the time outside the tags: a train leaves at 14:05 and the trip "
        "takes 2 h 40 min. What time does it arrive (HH:MM)?",
        min_words=3,
        max_words=30,
        task_type="concise_factual",
        artifact_type="exact_numeric",
        exact_answer="16:45",
        require_cot=True,
    ),
    _spec(
        "v2-code-is-even",
        "Plan inside <think></think> tags, then give a fenced ```python "
        "code block implementing is_even(n) that returns True when n is "
        "even. No prose outside the block, no clichés.",
        required_keywords=["def is_even"],
        min_words=3,
        max_words=80,
        task_type="concise_factual",
        artifact_type="code",
        require_cot=True,
    ),
    _spec(
        "v2-sql-top-customers",
        "Reason inside <think></think> tags, then output one readonly "
        "SELECT query returning the top 5 customers by total_spend from "
        "the customers table. No write statements, no disclaimers.",
        required_keywords=["select", "customers"],
        min_words=10,
        max_words=120,
        task_type="analytical_sql",
        artifact_type="sql_readonly",
        require_cot=True,
    ),
    _spec(
        "v2-sql-index-skip",
        "Deliberate inside <think></think> tags, then explain in 2-3 plain "
        "sentences outside the tags when a database engine would ignore an "
        "index. Concrete mechanisms, no clichés, no moralizing.",
        min_words=20,
        max_words=120,
        task_type="analytical_sql",
        artifact_type="none",
        target_ans_words=90,
        target_cot_words=150,
        require_cot=True,
    ),
    _spec(
        "v2-json-churn-flags",
        "Analyze inside <think></think> tags, then output ONLY JSON: "
        '{"at_risk": [<account ids>], "count": <n>} for the accounts with '
        "no login in the last 30 days: A1 (5 days), A2 (45 days), "
        "A3 (60 days). Nothing outside the JSON.",
        require_json=True,
        required_keys=["at_risk", "count"],
        required_keywords=["A2", "A3"],
        min_words=8,
        max_words=60,
        task_type="analytical_sql",
        artifact_type="json",
        require_cot=True,
    ),
    _spec(
        "v2-sql-avg-orders",
        "Think inside <think></think> tags, then write a single readonly "
        'query: the average order_total from orders where region = "emea". '
        "No write statements, no commentary.",
        required_keywords=["avg", "orders"],
        min_words=8,
        max_words=100,
        task_type="analytical_sql",
        artifact_type="sql_readonly",
        require_cot=True,
    ),
    _spec(
        "v2-deep-queue-backpressure",
        "Reason inside <think></think> tags, then explain in plain "
        "paragraphs (100-200 words) how backpressure works in a message "
        "queue and one failure mode when consumers are slow. Concrete "
        "verbs, no clichés, no disclaimers.",
        min_words=60,
        max_words=300,
        task_type="deep_technical_explanation",
        artifact_type="none",
        require_cot=True,
    ),
    _spec(
        "v2-deep-retry-decorator",
        "Think inside <think></think> tags, then provide a fenced "
        "```python code block with a retry(timeout_s) decorator sketch "
        "that retries a flaky call until the deadline. No prose outside "
        "the block, no clichés.",
        required_keywords=["def", "retry"],
        min_words=40,
        max_words=300,
        task_type="deep_technical_explanation",
        artifact_type="code",
        require_cot=True,
    ),
    _spec(
        "v2-json-capacity-plan",
        "Deliberate inside <think></think> tags, then output ONLY JSON: "
        '{"nodes": <n>, "ram_gb_per_node": <n>} for 2400 GB of data with '
        "200 GB of RAM per node. Nothing outside the JSON.",
        require_json=True,
        required_keys=["nodes", "ram_gb_per_node"],
        min_words=8,
        max_words=60,
        task_type="deep_technical_explanation",
        artifact_type="json",
        require_cot=True,
    ),
    _spec(
        "v2-deep-replica-bytes",
        "Reason inside <think></think> tags, then give only the number "
        "outside the tags: a shard holds 480 GB and the replication "
        "factor is 3. How many GB are stored in total across replicas?",
        min_words=3,
        max_words=40,
        task_type="deep_technical_explanation",
        artifact_type="exact_numeric",
        exact_answer="1440",
        require_cot=True,
    ),
)


@dataclass(frozen=True)
class UnslopScoreBreakdown:
    """Full reward decomposition for one unslop response."""

    total_reward: float
    g_compliance: float
    g_antislop: float
    r_ifeval: float
    r_coherence: float
    mode: str
    slop_hits: int
    refusal_hits: int
    distinct_3: float
    word_count: int
    diagnostics: dict[str, Any]
    length_retention_score: float = 1.0
    g_cot_integrity: float = 1.0
    g_length_cal: float = 1.0
    g_antislop_density: float = 1.0
    r_info_density: float = 1.0
    cot_word_count: int = 0
    ans_word_count: int = 0
    artifact_valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_reward": self.total_reward,
            "g_compliance": self.g_compliance,
            "g_antislop": self.g_antislop,
            "r_ifeval": self.r_ifeval,
            "r_coherence": self.r_coherence,
            "mode": self.mode,
            "slop_hits": self.slop_hits,
            "refusal_hits": self.refusal_hits,
            "distinct_3": self.distinct_3,
            "word_count": self.word_count,
            "diagnostics": self.diagnostics,
            "length_retention_score": self.length_retention_score,
            "g_cot_integrity": self.g_cot_integrity,
            "g_length_cal": self.g_length_cal,
            "g_antislop_density": self.g_antislop_density,
            "r_info_density": self.r_info_density,
            "cot_word_count": self.cot_word_count,
            "ans_word_count": self.ans_word_count,
            "artifact_valid": self.artifact_valid,
        }


_TOKEN_RE = re.compile(r"[a-z0-9']+")
_FENCE_OPEN_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*")
_CLEAN_ENDING_CHARS = frozenset(".!?\"')]}’”")


def _normalize(text: str) -> str:
    """Lowercase, unify curly quotes, and collapse whitespace for matching."""
    lowered = text.replace("’", "'").replace("‘", "'")
    lowered = lowered.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", lowered.lower()).strip()


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _phrase_hits(phrases: tuple[str, ...], normalized: str) -> list[str]:
    return [phrase for phrase in phrases if phrase in normalized]


def _term_present(term: str, normalized: str) -> bool:
    """Whole-phrase match at word boundaries (case-insensitive)."""
    pattern = r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)"
    return re.search(pattern, normalized) is not None


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    match = _FENCE_OPEN_RE.match(stripped)
    if match:
        stripped = stripped[match.end() :]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()


def _parse_json_object(text: str) -> tuple[bool, Any]:
    """Parse the text as JSON, tolerating one optional markdown code fence."""
    candidates = [text.strip(), _strip_code_fences(text)]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return True, json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return False, None


def _coherence(text: str, words: list[str]) -> tuple[float, dict[str, Any]]:
    """Average of non-empty, non-degenerate-repetition, and clean-ending checks."""
    stripped = text.strip()
    if not stripped:
        return 0.0, {"non_empty": False, "degenerate": True, "ends_cleanly": False}

    # Degenerate repetition: dominant token, immediate runs, or duplicate sentences.
    degenerate = False
    top_token_share = 0.0
    if len(words) >= 8:
        counts: dict[str, int] = {}
        for token in words:
            counts[token] = counts.get(token, 0) + 1
        top_token_share = max(counts.values()) / len(words)
        if top_token_share > 0.5:
            degenerate = True
    if not degenerate:
        run = 1
        for prev, cur in zip(words, words[1:], strict=False):
            run = run + 1 if cur == prev else 1
            if run >= 4:
                degenerate = True
                break
    if not degenerate:
        sentences = [
            re.sub(r"\s+", " ", sentence.strip().lower())
            for sentence in re.split(r"[.!?]+", stripped)
            if sentence.strip()
        ]
        if len(sentences) >= 4 and len(set(sentences)) / len(sentences) < 0.6:
            degenerate = True

    # Clean ending: punctuation or closing JSON bracket (tolerating a code fence).
    ending = _strip_code_fences(stripped)
    ends_cleanly = bool(ending) and ending[-1] in _CLEAN_ENDING_CHARS

    checks = [1.0, 0.0 if degenerate else 1.0, 1.0 if ends_cleanly else 0.0]
    score = sum(checks) / len(checks)
    details = {
        "non_empty": True,
        "degenerate": degenerate,
        "top_token_share": round(top_token_share, 4),
        "ends_cleanly": ends_cleanly,
    }
    return score, details


def _length_retention(word_count: int, spec: UnslopPromptSpec) -> float:
    """Reward for keeping ``word_count`` inside the spec's word budget.

    Empty responses score 0.0, undershoot is penalized quadratically, and
    overshoot decays linearly (floored at 0.2). In-budget responses score 1.0.
    """
    min_w = spec.min_words if spec.min_words is not None else 12
    max_w = spec.max_words if spec.max_words is not None else 200
    if word_count == 0:
        return 0.0
    if word_count < min_w:
        return (word_count / max(1, min_w)) ** 2
    if word_count > max_w:
        return max(0.2, 1.0 - (word_count - max_w) / max(1, max_w))
    return 1.0


_V2_DEFAULT_ANS_TARGETS: dict[str, int] = {
    "concise_factual": 45,
    "analytical_sql": 65,
    "deep_technical_explanation": 140,
}
_V2_DEFAULT_COT_TARGETS: dict[str, int] = {
    "concise_factual": 60,
    "analytical_sql": 120,
    "deep_technical_explanation": 180,
}

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
_SQL_WRITE_KEYWORDS: tuple[str, ...] = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "truncate",
    "merge",
    "replace",
    "grant",
    "revoke",
    "create",
)
_FENCED_CODE_RE = re.compile(r"```([a-zA-Z0-9_-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
_PYTHON_LANG_TAGS = frozenset({"python", "py"})


def _paragraphs(text: str) -> list[str]:
    """Split into paragraphs on blank lines, falling back to non-empty lines."""
    paragraphs = [chunk.strip() for chunk in _PARAGRAPH_SPLIT_RE.split(text)]
    paragraphs = [chunk for chunk in paragraphs if chunk]
    if not paragraphs:
        paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    return paragraphs


def _distinct_n(words: list[str], n: int) -> float:
    """Fraction of distinct n-grams (1.0 when the text is shorter than n words)."""
    if len(words) < n:
        return 1.0
    ngrams = list(zip(*(words[i:] for i in range(n)), strict=False))
    if not ngrams:
        return 1.0
    return len(set(ngrams)) / len(ngrams)


def _v2_length_calibration(
    ans_word_count: int, cot_word_count: int, spec: UnslopPromptSpec
) -> float:
    """Log-normal fit of answer length times a soft CoT overshoot penalty."""
    if ans_word_count <= 0:
        return 0.0
    target_ans = spec.target_ans_words or _V2_DEFAULT_ANS_TARGETS[spec.task_type]
    sigma = 0.45 if ans_word_count < target_ans else 0.75
    ans_cal = math.exp(-0.5 * (math.log(ans_word_count / target_ans) / sigma) ** 2)

    target_cot = spec.target_cot_words or _V2_DEFAULT_COT_TARGETS[spec.task_type]
    if cot_word_count <= target_cot:
        cot_cal = 1.0
    else:
        sigma_cot = max(1.0, 0.30 * target_cot)
        cot_cal = max(0.2, math.exp(-0.5 * ((cot_word_count - target_cot) / sigma_cot) ** 2))
    return float(ans_cal * cot_cal)


def _v2_artifact_validity(answer_text: str, spec: UnslopPromptSpec) -> tuple[bool, float]:
    """Validate the answer's artifact against the spec's artifact type."""
    artifact = spec.artifact_type
    if artifact == "none":
        return True, 1.0
    if artifact == "json":
        parsed, obj = _parse_json_object(answer_text)
        valid = parsed and (
            not spec.required_keys
            or (isinstance(obj, dict) and all(key in obj for key in spec.required_keys))
        )
    elif artifact == "sql_readonly":
        normalized_sql = _normalize(_strip_code_fences(answer_text))
        has_select = _term_present("select", normalized_sql)
        write_hits = [kw for kw in _SQL_WRITE_KEYWORDS if _term_present(kw, normalized_sql)]
        valid = has_select and not write_hits
    elif artifact == "code":
        blocks = _FENCED_CODE_RE.findall(answer_text)
        valid = False
        for lang, body in blocks:
            if not body.strip():
                continue
            if lang.lower() in _PYTHON_LANG_TAGS:
                try:
                    ast.parse(body)
                    valid = True
                    break
                except (SyntaxError, ValueError):
                    continue
            valid = True
            break
        if not valid and not blocks:
            # Python leniency: raw (or fence-truncated) code that parses counts.
            candidate = _strip_code_fences(answer_text)
            if candidate:
                try:
                    ast.parse(candidate)
                    valid = True
                except (SyntaxError, ValueError):
                    valid = False
    elif artifact == "exact_numeric":
        if spec.exact_answer is not None:
            valid = _term_present(spec.exact_answer, _normalize(answer_text))
        else:
            valid = any(char.isdigit() for char in answer_text)
    else:  # pragma: no cover - artifact_type is a closed Literal
        valid = True
    return bool(valid), 1.0 if valid else 0.0


def _v2_antislop_density(answer_text: str, ans_words: list[str]) -> tuple[float, int, list[str]]:
    """Slop score combining an edge-paragraph penalty with per-100-word density."""
    paragraphs = _paragraphs(answer_text)
    edge_texts = paragraphs[:1] + paragraphs[-1:] if paragraphs else []
    edge_normalized = [_normalize(edge) for edge in edge_texts]
    edge_hits = [
        phrase for phrase in SLOP_PHRASES if any(phrase in norm for norm in edge_normalized)
    ]
    all_hits = _phrase_hits(SLOP_PHRASES, _normalize(answer_text))

    score = 1.0
    if edge_hits:
        score *= 0.20
    density = (len(all_hits) / max(1, len(ans_words))) * 100.0
    score *= max(0.0, 1.0 - 0.15 * density)
    return min(1.0, max(0.0, score)), len(all_hits), all_hits


_FACT_DENSITY_NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?%?")
_FACT_DENSITY_CODE_RE = re.compile(r"`[^`\n]+`")
_FACT_DENSITY_IDENT_RE = re.compile(
    r"\b(?:[a-z]+(?:_[a-z0-9]+)+|[A-Z][a-z0-9]*(?:[A-Z][a-z0-9]+)+|[A-Z]{2,}(?:_[A-Z0-9]+)*)\b"
)
_FACT_DENSITY_SQL_RE = re.compile(r"\b(?:select|join|where|group\s+by)\b", re.IGNORECASE)


def _v2_fact_density(answer_text: str, ans_words: list[str]) -> float:
    """Factual/technical information density: fact-bearing tokens per answer word."""
    if not ans_words:
        return 0.0
    fact_hits = 0
    # Numbers: integers, decimals, percentages.
    fact_hits += len(_FACT_DENSITY_NUMERIC_RE.findall(answer_text))
    # Backticked inline code / identifiers (counted once, then removed from
    # the text so their contents are not double-counted below).
    fact_hits += len(_FACT_DENSITY_CODE_RE.findall(answer_text))
    without_code = _FACT_DENSITY_CODE_RE.sub(" ", answer_text)
    # snake_case / CamelCase / UPPER_ACRONYM identifier tokens.
    fact_hits += len(_FACT_DENSITY_IDENT_RE.findall(without_code))
    # Non-sentence-initial capitalized words act as entity mentions.
    for chunk in re.split(r"(?<=[.!?])\s+|\n+", without_code):
        for word in chunk.split()[1:]:
            candidate = word.strip("\"'“”‘’()[]{}.,;:!?")
            if re.fullmatch(r"[A-Z][a-z]+", candidate):
                fact_hits += 1
    # Fenced code blocks or SQL keywords signal technical content.
    if _FENCED_CODE_RE.search(answer_text) or _FACT_DENSITY_SQL_RE.search(answer_text):
        fact_hits += 2
    return min(1.0, (fact_hits / max(1, len(ans_words))) / 0.20)


def _v2_cot_integrity(
    span: PhaseSpan, spec: UnslopPromptSpec, segmenter: PhaseSegmenter
) -> tuple[float, int]:
    """Gate on well-formed, non-leaking, non-degenerate deliberation."""
    cot_words = _tokens(span.deliberation_text)
    cot_word_count = len(cot_words)
    if (
        span.malformed
        or (spec.require_cot and not span.has_deliberation)
        or segmenter.deliberation_start in span.answer_text
        or segmenter.deliberation_end in span.answer_text
        or (cot_word_count >= 12 and _distinct_n(cot_words, 4) < 0.50)
    ):
        return 0.0, cot_word_count
    return 1.0, cot_word_count


class TenetUnslopRubric:
    """Gated/additive/v2 rubric over compliance, anti-slop, constraints, coherence."""

    def __init__(
        self,
        mode: Literal["gated", "additive", "v2"] = "gated",
        phase_segmenter: PhaseSegmenter | None = None,
    ) -> None:
        if mode not in ("gated", "additive", "v2"):
            raise ValueError(f"Unknown unslop rubric mode: {mode!r}")
        self.mode = mode
        if phase_segmenter is not None:
            self.phase_segmenter = phase_segmenter
        else:
            from sml.surgery.probes import PhaseSegmenter

            self.phase_segmenter = PhaseSegmenter()

    def score(self, text: str, spec: UnslopPromptSpec) -> UnslopScoreBreakdown:
        if self.mode == "v2":
            return self._score_v2(text, spec)

        normalized = _normalize(text)
        words = _tokens(text)
        word_count = len(words)

        # Gate: refusal/empty compliance.
        matched_refusals = _phrase_hits(REFUSAL_DISCLAIMERS, normalized)
        refusal_hits = len(matched_refusals)
        g_compliance = 1.0 if (text.strip() and refusal_hits == 0) else 0.0

        # Gate: anti-slop.
        matched_slop = _phrase_hits(SLOP_PHRASES, normalized)
        slop_hits = len(matched_slop)
        if word_count < 3:
            distinct_3 = 1.0
            trigram_total = 0
        else:
            trigrams = list(zip(words, words[1:], words[2:], strict=False))
            trigram_total = len(trigrams)
            distinct_3 = len(set(trigrams)) / trigram_total
        slop_penalty = max(0.0, 1.0 - 0.35 * slop_hits)
        g_antislop = slop_penalty
        if word_count >= 6 and distinct_3 < 0.55:
            g_antislop *= max(0.0, distinct_3 / 0.55)
        g_antislop = min(1.0, max(0.0, g_antislop))

        # Soft: instruction-following constraints.
        constraint_scores: dict[str, float] = {}
        constraint_details: dict[str, Any] = {}

        if spec.require_json:
            parsed, obj = _parse_json_object(text)
            missing_keys: list[str] = []
            if not parsed:
                json_score = 0.0
            elif not isinstance(obj, dict):
                json_score = 0.5
            elif spec.required_keys:
                missing_keys = [key for key in spec.required_keys if key not in obj]
                json_score = 1.0 - len(missing_keys) / len(spec.required_keys)
            else:
                json_score = 1.0
            constraint_scores["json_schema"] = json_score
            constraint_details["json"] = {
                "parsed": parsed,
                "is_object": parsed and isinstance(obj, dict),
                "missing_keys": missing_keys,
            }

        constraint_scores["word_count"] = (
            1.0 if spec.min_words <= word_count <= spec.max_words else 0.0
        )

        if spec.forbidden_words:
            forbidden_hits = [
                term for term in spec.forbidden_words if _term_present(term, normalized)
            ]
            constraint_scores["forbidden_words"] = 1.0 - len(forbidden_hits) / len(
                spec.forbidden_words
            )
            constraint_details["forbidden_hits"] = forbidden_hits

        if spec.required_keywords:
            missing_keywords = [
                term for term in spec.required_keywords if not _term_present(term, normalized)
            ]
            constraint_scores["required_keywords"] = (
                len(spec.required_keywords) - len(missing_keywords)
            ) / len(spec.required_keywords)
            constraint_details["missing_keywords"] = missing_keywords

        if constraint_scores:
            r_ifeval = sum(constraint_scores.values()) / len(constraint_scores)
        else:
            r_ifeval = 1.0

        # Soft: coherence (blended with length retention).
        base_coherence, coherence_details = _coherence(text, words)
        length_retention_score = _length_retention(word_count, spec)
        r_coherence = float(0.65 * base_coherence + 0.35 * length_retention_score)
        if length_retention_score < 0.5:
            # Ultra-short (or far-too-long) collapse: cap coherence hard.
            r_coherence = min(r_coherence, length_retention_score / 0.5)

        # Total.
        if self.mode == "gated":
            total_reward = g_compliance * g_antislop * (0.6 * r_ifeval + 0.4 * r_coherence)
        else:
            total_reward = (
                0.25 * g_compliance + 0.25 * g_antislop + 0.30 * r_ifeval + 0.20 * r_coherence
            )
        total_reward = min(1.0, max(0.0, total_reward))

        diagnostics: dict[str, Any] = {
            "scenario_id": spec.scenario_id,
            "matched_slop_phrases": matched_slop,
            "matched_refusal_phrases": matched_refusals,
            "constraint_scores": constraint_scores,
            "constraint_details": constraint_details,
            "coherence": coherence_details,
            "trigram_total": trigram_total,
        }

        return UnslopScoreBreakdown(
            total_reward=round(total_reward, 6),
            g_compliance=g_compliance,
            g_antislop=round(g_antislop, 6),
            r_ifeval=round(r_ifeval, 6),
            r_coherence=round(r_coherence, 6),
            mode=self.mode,
            slop_hits=slop_hits,
            refusal_hits=refusal_hits,
            distinct_3=round(distinct_3, 6),
            word_count=word_count,
            diagnostics=diagnostics,
            length_retention_score=length_retention_score,
        )

    def _score_v2(self, text: str, spec: UnslopPromptSpec) -> UnslopScoreBreakdown:
        """Deliberation-aware scoring: artifact validity, CoT integrity, calibration."""
        segmenter = self.phase_segmenter
        span = segmenter.segment_text(text)

        # Evaluate the answer phase; fall back to the full text when no answer survived.
        answer_text = span.answer_text if span.answer_text.strip() else text
        normalized_answer = _normalize(answer_text)
        ans_words = _tokens(answer_text)
        ans_word_count = len(ans_words)

        # Gate: refusal/empty compliance on the answer.
        matched_refusals = _phrase_hits(REFUSAL_DISCLAIMERS, normalized_answer)
        refusal_hits = len(matched_refusals)
        g_compliance = 1.0 if (answer_text.strip() and refusal_hits == 0) else 0.0

        # Gate: deliberation integrity (malformed/leaked/degenerate CoT).
        g_cot_integrity, cot_word_count = _v2_cot_integrity(span, spec, segmenter)

        # Gates: slop density and calibrated answer/CoT lengths.
        g_antislop_density, slop_hits, matched_slop = _v2_antislop_density(answer_text, ans_words)
        g_length_cal = _v2_length_calibration(ans_word_count, cot_word_count, spec)
        artifact_valid, artifact_score = _v2_artifact_validity(answer_text, spec)

        # Soft: instruction-following constraints on the answer.
        constraint_scores: dict[str, float] = {}
        constraint_details: dict[str, Any] = {}

        if spec.require_json:
            parsed, obj = _parse_json_object(answer_text)
            missing_keys: list[str] = []
            if not parsed:
                json_score = 0.0
            elif not isinstance(obj, dict):
                json_score = 0.5
            elif spec.required_keys:
                missing_keys = [key for key in spec.required_keys if key not in obj]
                json_score = 1.0 - len(missing_keys) / len(spec.required_keys)
            else:
                json_score = 1.0
            constraint_scores["json_schema"] = json_score
            constraint_details["json"] = {
                "parsed": parsed,
                "is_object": parsed and isinstance(obj, dict),
                "missing_keys": missing_keys,
            }

        constraint_scores["word_count"] = (
            1.0 if spec.min_words <= ans_word_count <= spec.max_words else 0.0
        )

        if spec.forbidden_words:
            forbidden_hits = [
                term for term in spec.forbidden_words if _term_present(term, normalized_answer)
            ]
            constraint_scores["forbidden_words"] = 1.0 - len(forbidden_hits) / len(
                spec.forbidden_words
            )
            constraint_details["forbidden_hits"] = forbidden_hits

        if spec.required_keywords:
            missing_keywords = [
                term
                for term in spec.required_keywords
                if not _term_present(term, normalized_answer)
            ]
            constraint_scores["required_keywords"] = (
                len(spec.required_keywords) - len(missing_keywords)
            ) / len(spec.required_keywords)
            constraint_details["missing_keywords"] = missing_keywords

        r_ifeval = (
            sum(constraint_scores.values()) / len(constraint_scores) if constraint_scores else 1.0
        )

        # Soft: information density (artifact quality, factual density, trigram variety).
        distinct_3 = _distinct_n(ans_words, 3)
        fact_density = _v2_fact_density(answer_text, ans_words)
        r_info_density = 0.40 * artifact_score + 0.30 * fact_density + 0.30 * distinct_3
        if not artifact_valid:
            r_info_density = min(r_info_density, 0.25)

        total_reward = (
            g_compliance
            * g_cot_integrity
            * g_antislop_density
            * g_length_cal
            * (0.5 * r_ifeval + 0.5 * r_info_density)
        )
        total_reward = min(1.0, max(0.0, total_reward))

        # Diagnostics-only coherence/length-retention readouts on the answer.
        base_coherence, coherence_details = _coherence(answer_text, ans_words)
        length_retention_score = _length_retention(ans_word_count, spec)
        r_coherence = float(0.65 * base_coherence + 0.35 * length_retention_score)

        edge_paragraphs = _paragraphs(answer_text)
        edge_slop_hits: list[str] = []
        if edge_paragraphs:
            for edge in edge_paragraphs[:1] + edge_paragraphs[-1:]:
                edge_slop_hits.extend(
                    phrase
                    for phrase in _phrase_hits(SLOP_PHRASES, _normalize(edge))
                    if phrase not in edge_slop_hits
                )

        diagnostics: dict[str, Any] = {
            "scenario_id": spec.scenario_id,
            "matched_slop_phrases": matched_slop,
            "matched_refusal_phrases": matched_refusals,
            "constraint_scores": constraint_scores,
            "constraint_details": constraint_details,
            "coherence": coherence_details,
            "artifact_type": spec.artifact_type,
            "artifact_valid": artifact_valid,
            "artifact_score": artifact_score,
            "fact_density": round(fact_density, 4),
            "edge_slop_hits": edge_slop_hits,
            "has_deliberation": span.has_deliberation,
            "cot_malformed": span.malformed,
            "cot_distinct_4": _distinct_n(_tokens(span.deliberation_text), 4),
            "resolved_target_ans_words": spec.target_ans_words
            or _V2_DEFAULT_ANS_TARGETS[spec.task_type],
            "resolved_target_cot_words": spec.target_cot_words
            or _V2_DEFAULT_COT_TARGETS[spec.task_type],
        }

        return UnslopScoreBreakdown(
            total_reward=round(total_reward, 6),
            g_compliance=g_compliance,
            g_antislop=round(g_antislop_density, 6),
            r_ifeval=round(r_ifeval, 6),
            r_coherence=round(r_coherence, 6),
            mode="v2",
            slop_hits=slop_hits,
            refusal_hits=refusal_hits,
            distinct_3=round(distinct_3, 6),
            word_count=ans_word_count,
            diagnostics=diagnostics,
            length_retention_score=length_retention_score,
            g_cot_integrity=g_cot_integrity,
            g_length_cal=round(g_length_cal, 6),
            g_antislop_density=round(g_antislop_density, 6),
            r_info_density=round(r_info_density, 6),
            cot_word_count=cot_word_count,
            ans_word_count=ans_word_count,
            artifact_valid=artifact_valid,
        )


def score_unslop_response(
    text: str,
    spec: UnslopPromptSpec,
    mode: Literal["gated", "additive", "v2"] = "gated",
) -> UnslopScoreBreakdown:
    """Convenience wrapper: score ``text`` against ``spec`` in one call."""
    return TenetUnslopRubric(mode=mode).score(text, spec)

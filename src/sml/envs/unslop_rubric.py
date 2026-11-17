"""Tenet unslop rubric: rewards instruction-following without LLM slop or refusals.

Sprint 6 Batch 1. Scores model responses against an ``UnslopPromptSpec`` on four axes:

- ``g_compliance``: gate — non-empty and free of refusal/moralizing disclaimers.
- ``g_antislop``: gate — free of clichéd "LLM slop" phrases, non-repetitive trigrams.
- ``r_ifeval``: soft reward — JSON schema, word limits, forbidden words, keywords.
- ``r_coherence``: soft reward — non-empty, non-degenerate, clean ending.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

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
    """Instruction spec for one unslop training/eval scenario."""

    scenario_id: str
    prompt: str
    require_json: bool = False
    required_keys: list[str] = field(default_factory=list)
    forbidden_words: list[str] = field(default_factory=list)
    required_keywords: list[str] = field(default_factory=list)
    min_words: int = 1
    max_words: int = 200


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
        "Give walking directions (20-60 words) from the fountain to gate B "
        'using the words "left", "bridge", and "200 meters". Nothing '
        "else, no disclaimers, no clichés.",
        required_keywords=["left", "bridge", "200 meters"],
        min_words=20,
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


class TenetUnslopRubric:
    """Gated/additive rubric over compliance, anti-slop, constraints, coherence."""

    def __init__(self, mode: Literal["gated", "additive"] = "gated"):
        if mode not in ("gated", "additive"):
            raise ValueError(f"Unknown unslop rubric mode: {mode!r}")
        self.mode = mode

    def score(self, text: str, spec: UnslopPromptSpec) -> UnslopScoreBreakdown:
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


def score_unslop_response(
    text: str,
    spec: UnslopPromptSpec,
    mode: Literal["gated", "additive"] = "gated",
) -> UnslopScoreBreakdown:
    """Convenience wrapper: score ``text`` against ``spec`` in one call."""
    return TenetUnslopRubric(mode=mode).score(text, spec)

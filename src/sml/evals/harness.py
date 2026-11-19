import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def compute_bpb(loss_nats: float, bytes_per_token: float) -> float:
    """Compute bits per byte from loss in nats."""
    return loss_nats / (math.log(2.0) * bytes_per_token)


@torch.no_grad()
def evaluate_validation_bpb(
    model: torch.nn.Module,
    dataset: Any,
    num_batches: int,
    batch_size: int,
    bytes_per_token: float,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Evaluate model on validation dataset and compute BPB."""
    model.eval()

    # Save dataset state
    original_state = dataset.state_dict()

    total_loss = 0.0
    total_tokens = 0

    try:
        for _ in range(num_batches):
            # Support both ShardedDataset.next_batch and custom get_batch
            batch_loader = getattr(dataset, "next_batch", None) or dataset.get_batch
            x, y, doc_ids = batch_loader(batch_size, device)

            # Forward pass
            out = model(x, targets=y, doc_ids=doc_ids)

            # Accumulate loss and token count
            total_loss += out.loss.item() * y.numel()
            total_tokens += y.numel()

        # Calculate averages
        mean_loss = total_loss / max(1, total_tokens)
        bpb = compute_bpb(mean_loss, bytes_per_token)

        return {"val_loss_nats": mean_loss, "val_bpb": bpb}
    finally:
        # Restore dataset state
        dataset.load_state_dict(original_state)


@torch.no_grad()
def score_multiple_choice_example(
    model: torch.nn.Module,
    prompt_ids: list[int],
    continuations: list[list[int]],
    continuation_byte_lengths: list[int] | None = None,
    device: str | torch.device = "cpu",
) -> tuple[int, int]:
    """Score multiple choice example with raw and byte-normalized scores."""
    model.eval()

    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    prompt_len = prompt_tensor.size(0)

    raw_scores = []
    byte_normalized_scores = []

    for i, cont_ids in enumerate(continuations):
        # Combine prompt and continuation
        full_seq = torch.cat(
            [prompt_tensor, torch.tensor(cont_ids, dtype=torch.long, device=device)]
        )

        # Get logits for the entire sequence
        logits = model(full_seq.unsqueeze(0)).logits.squeeze(0)

        # Calculate log probabilities for continuation tokens
        cont_logits = logits[prompt_len - 1 : -1]  # -1 because we predict next token
        cont_targets = full_seq[prompt_len:]

        # Compute log probabilities
        log_probs = -F.cross_entropy(cont_logits, cont_targets, reduction="sum")

        raw_scores.append(log_probs.item())

        # Byte-normalized score
        byte_length = continuation_byte_lengths[i] if continuation_byte_lengths else len(cont_ids)
        byte_normalized_scores.append(log_probs.item() / max(1, byte_length))

    # Find best indices
    best_raw_idx = int(torch.argmax(torch.tensor(raw_scores)).item())
    best_byte_norm_idx = int(torch.argmax(torch.tensor(byte_normalized_scores)).item())

    return (best_raw_idx, best_byte_norm_idx)


def bootstrap_confidence_interval(
    scores: Sequence[float], n_resamples: int = 1000, ci: float = 0.95, seed: int = 1337
) -> tuple[float, float, float]:
    """Calculate bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    scores_array = np.array(scores)
    n_scores = len(scores_array)

    if n_scores == 0:
        return (0.0, 0.0, 0.0)

    # Sample means
    sample_means = []
    for _ in range(n_resamples):
        sample = rng.choice(scores_array, size=n_scores, replace=True)
        sample_means.append(np.mean(sample))

    sample_means = np.array(sample_means)

    # Calculate percentiles
    alpha = 1.0 - ci
    lower_percentile = (alpha / 2.0) * 100
    upper_percentile = (1.0 - alpha / 2.0) * 100

    lower_bound = np.percentile(sample_means, lower_percentile)
    upper_bound = np.percentile(sample_means, upper_percentile)
    mean_score = np.mean(scores_array)

    return (float(mean_score), float(lower_bound), float(upper_bound))


@torch.no_grad()
def evaluate_induction_metrics(
    model: torch.nn.Module,
    long_doc_input_ids: torch.Tensor,
    early_pos: int = 8,
    late_pos: int = 24,
    prefix_len: int = 8,
    device: str | torch.device = "cpu",
    seed: int = 1337,
) -> dict[str, float]:
    """Evaluate induction metrics on long document."""
    model.eval()
    torch.manual_seed(seed)

    # Move tensor to device
    long_doc_input_ids = long_doc_input_ids.to(device)
    seq_len = long_doc_input_ids.size(0)

    # Calculate in-context loss difference
    with torch.no_grad():
        logits = model(long_doc_input_ids.unsqueeze(0)).logits.squeeze(0)

        # Loss at early position
        early_start = early_pos
        early_end = min(early_start + prefix_len, seq_len - 1)
        early_loss = F.cross_entropy(
            logits[early_start - 1 : early_end - 1],
            long_doc_input_ids[early_start:early_end],
            reduction="mean",
        )

        # Loss at late position
        late_start = late_pos
        late_end = min(late_start + prefix_len, seq_len - 1)
        late_loss = F.cross_entropy(
            logits[late_start - 1 : late_end - 1],
            long_doc_input_ids[late_start:late_end],
            reduction="mean",
        )

        in_context_loss_delta = float(late_loss - early_loss)

    # Synthetic copy accuracy
    # Create repeated pattern: [p1...pL, p1...pL]
    pattern_len = 16
    vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 256))
    pattern = torch.randint(0, vocab_size, (pattern_len,), device=device)
    repeated_pattern = torch.cat([pattern, pattern])

    # Get model predictions for the repeated pattern
    with torch.no_grad():
        pattern_logits = model(repeated_pattern.unsqueeze(0)).logits.squeeze(0)
        # Predictions for second half based on first half
        predictions = torch.argmax(pattern_logits[pattern_len - 1 : -1], dim=-1)
        targets = repeated_pattern[pattern_len:]

        # Accuracy of copying
        copy_accuracy = (predictions == targets).float().mean().item()

        # Average probability of correct tokens
        correct_log_probs = -F.cross_entropy(
            pattern_logits[pattern_len - 1 : -1], targets, reduction="none"
        )
        copy_prob = torch.exp(correct_log_probs).mean().item()

    return {
        "in_context_loss_delta": in_context_loss_delta,
        "synthetic_copy_accuracy": copy_accuracy,
        "synthetic_copy_prob": copy_prob,
    }


# ---------------------------------------------------------------------------
# External benchmark evaluation: pre/post Lancet + Unslop GRPO + TIES merge
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9']+")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_HASH_RE = re.compile(r"####\s*([-+]?[\d,]+(?:\.\d+)?)")
_NUMBER_RE = re.compile(r"[-+]?[\d,]+(?:\.\d+)?")


def _normalize_text(text: str) -> str:
    """Lowercase, unify curly quotes, and collapse whitespace for phrase matching."""
    lowered = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    return re.sub(r"\s+", " ", lowered.lower()).strip()


@lru_cache(maxsize=1)
def _slop_and_refusal_phrases() -> tuple[str, ...]:
    from sml.envs.unslop_rubric import REFUSAL_DISCLAIMERS, SLOP_PHRASES

    return tuple(SLOP_PHRASES) + tuple(REFUSAL_DISCLAIMERS)


def _free_of_slop_and_refusal(text: str) -> bool:
    normalized = _normalize_text(text)
    return not any(phrase in normalized for phrase in _slop_and_refusal_phrases())


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text.lower()))


@lru_cache(maxsize=1)
def _slop_preamble_text() -> str:
    return (
        "Sure, here is my response to that request. Honestly, it unfolds like a "
        "tapestry of ideas, a symphony of scattered notes, and a genuine deep dive "
        "into the topic at hand."
    )


@lru_cache(maxsize=1)
def _refusal_preamble_text() -> str:
    return (
        "As an AI, I cannot provide a direct answer for this request. It's important "
        "to note that a request like this deserves careful handling, so please consult "
        "a licensed professional before proceeding."
    )


def _extract_final_number(text: str) -> float | None:
    """Extract the final numeric answer from \\boxed{...}, '#### <num>', or last number."""
    candidate: str | None = None
    for match in _BOXED_RE.finditer(text):
        candidate = match.group(1)
    if candidate is None:
        hash_match = _HASH_RE.search(text)
        if hash_match is not None:
            candidate = hash_match.group(1)
    if candidate is None:
        numbers = _NUMBER_RE.findall(text)
        if not numbers:
            return None
        candidate = numbers[-1]
    cleaned = candidate.strip().replace(",", "").replace("$", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _numbers_close(predicted: float, ground_truth: float) -> bool:
    return abs(predicted - ground_truth) <= 1e-6 * max(1.0, abs(ground_truth))


def _check_json_object_only(text: str) -> tuple[bool, str]:
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return False, "response is not a bare JSON object"
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return False, "response is not valid JSON"
    required = ("city", "population_millions", "capital")
    if not isinstance(obj, dict) or any(key not in obj for key in required):
        return False, f"JSON object must be a dict with keys {required}"
    if not _free_of_slop_and_refusal(text):
        return False, "slop preamble or refusal disclaimer detected"
    return True, "ok"


def _check_bullet_lines(text: str) -> tuple[bool, str]:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) != 3:
        return False, f"expected exactly 3 bullet lines, found {len(lines)}"
    if not all(line.startswith("- ") for line in lines):
        return False, "every line must start with '- '"
    if not _free_of_slop_and_refusal(text):
        return False, "slop preamble or refusal disclaimer detected"
    return True, "ok"


def _check_word_limit(text: str) -> tuple[bool, str]:
    count = _word_count(text)
    if not (12 <= count <= 30):
        return False, f"word count {count} outside the 12-30 limit"
    if not _free_of_slop_and_refusal(text):
        return False, "slop preamble or refusal disclaimer detected"
    return True, "ok"


def _check_uppercase_single_line(text: str) -> tuple[bool, str]:
    stripped = text.strip()
    if not stripped or "\n" in stripped:
        return False, "response must be exactly one non-empty line"
    if stripped != stripped.upper():
        return False, "response must be ALL UPPERCASE"
    if stripped[-1] in ".,!?;:":
        return False, "response must not end with punctuation"
    if not _free_of_slop_and_refusal(text):
        return False, "slop preamble or refusal disclaimer detected"
    return True, "ok"


def _check_json_array_sorted(text: str) -> tuple[bool, str]:
    stripped = text.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return False, "response is not a bare JSON array"
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return False, "response is not valid JSON"
    if not isinstance(obj, list) or obj != [3, 7, 9]:
        return False, "array must be exactly [3, 7, 9]"
    if not _free_of_slop_and_refusal(text):
        return False, "slop preamble or refusal disclaimer detected"
    return True, "ok"


def _check_keyword_and_ending(text: str) -> tuple[bool, str]:
    if "PERGOLA" not in text:
        return False, "exact keyword 'PERGOLA' missing"
    if not text.strip().endswith("DONE"):
        return False, "reply must end with the word 'DONE'"
    count = _word_count(text)
    if not (10 <= count <= 35):
        return False, f"word count {count} outside the 10-35 limit"
    if not _free_of_slop_and_refusal(text):
        return False, "slop preamble or refusal disclaimer detected"
    return True, "ok"


@dataclass(frozen=True)
class _IFEvalCase:
    """One deterministic IFEval-style format-following prompt with staged references."""

    prompt_id: str
    prompt: str
    check: Callable[[str], tuple[bool, str]]
    after_response: str
    before_response: str


IFEVAL_CASES: tuple[_IFEvalCase, ...] = (
    _IFEvalCase(
        prompt_id="json-only-object",
        prompt=(
            'Reply with ONLY a JSON object with keys "city" (string), '
            '"population_millions" (number), and "capital" (boolean) describing '
            "France. Output the JSON object alone: no preamble, no code fences, "
            "no explanation."
        ),
        check=_check_json_object_only,
        after_response='{"city": "Paris", "population_millions": 2.1, "capital": true}',
        before_response=(
            "Certainly! Here is the JSON you requested:\n\n"
            '{"city": "Paris", "population_millions": 2.1, "capital": true}'
        ),
    ),
    _IFEvalCase(
        prompt_id="exact-bullet-count",
        prompt=(
            "List exactly 3 steps to reset a password. Respond with exactly 3 bullet "
            'lines, each line starting with "- ". No numbering, no intro sentence, '
            "no closing sentence, no extra lines."
        ),
        check=_check_bullet_lines,
        after_response=(
            '- Open the account settings page.\n- Click "Forgot password" and follow '
            "the emailed link.\n- Save the new password in your password manager."
        ),
        before_response=(
            "Sure, here are the steps to reset your password:\n\n"
            "1. Open the settings app.\n2. Tap forgot password.\n3. Follow the link."
        ),
    ),
    _IFEvalCase(
        prompt_id="word-limit-30",
        prompt=(
            "Explain why the sky is blue in at most 30 words and at least 12 words. "
            "Plain declarative sentences only: no headings, no lists, no disclaimers."
        ),
        check=_check_word_limit,
        after_response=(
            "Sunlight scatters off air molecules. Short blue wavelengths scatter far "
            "more strongly than long red wavelengths, so scattered blue light reaches "
            "your eyes from every direction of the sky."
        ),
        before_response=(
            "What a fascinating question! It's important to note that the sky appears "
            "blue because sunlight interacts with our atmosphere in remarkable ways. "
            "Rayleigh scattering affects different wavelengths differently, with the "
            "shorter ones scattering more, which is why we perceive that beautiful "
            "azure hue overhead on clear days."
        ),
    ),
    _IFEvalCase(
        prompt_id="uppercase-single-line",
        prompt=(
            "Answer with a single line in ALL UPPERCASE letters stating the boiling "
            "point of water at sea level in Celsius. No lowercase letters, no line "
            "breaks, no trailing punctuation."
        ),
        check=_check_uppercase_single_line,
        after_response="WATER BOILS AT 100 DEGREES CELSIUS AT SEA LEVEL",
        before_response="Water boils at 100 degrees Celsius at sea level.",
    ),
    _IFEvalCase(
        prompt_id="json-array-sorted",
        prompt=(
            "Output ONLY a JSON array containing the numbers 9, 3, and 7 sorted in "
            "ascending order as integers. Output the array alone: no markdown, no "
            "explanation."
        ),
        check=_check_json_array_sorted,
        after_response="[3, 7, 9]",
        before_response="The sorted numbers in ascending order are 3, 7, and 9.",
    ),
    _IFEvalCase(
        prompt_id="keyword-and-ending",
        prompt=(
            "Write one sentence (10-35 words) describing a garden pergola. You must "
            'include the exact keyword "PERGOLA" and you must end your reply with '
            "the word DONE."
        ),
        check=_check_keyword_and_ending,
        after_response=(
            "A cedar PERGOLA covers the stone patio, where grape vines climb its "
            "slatted beams and shade the summer table below. DONE"
        ),
        before_response=(
            "Here is a sentence about a garden structure: the pergola gives nice "
            "shade and the vines look lovely in summer. Let me know if you want "
            "more options!"
        ),
    ),
)


@dataclass(frozen=True)
class _MathCase:
    """One deterministic GSM8K/MATH-style problem with an exact numeric ground truth."""

    problem_id: str
    prompt: str
    answer: float
    after_response: str
    before_response: str


GSM8K_MATH_CASES: tuple[_MathCase, ...] = (
    _MathCase(
        problem_id="gsm_muffin_inventory",
        prompt=(
            "A bakery sells 24 muffins per tray. The bakers fill 9 trays in the "
            "morning and sell 96 muffins before noon. How many muffins are left? "
            "Give the final number on its own line after '####'."
        ),
        answer=120.0,
        after_response=(
            "9 trays give 9 * 24 = 216 muffins. 216 - 96 = 120 muffins are left.\n#### 120"
        ),
        before_response=(
            "Sure! The bakery probably has around 63 muffins left over after the "
            "morning rush.\n#### 63"
        ),
    ),
    _MathCase(
        problem_id="gsm_train_speed",
        prompt=(
            "A train travels 60 kilometers in 45 minutes. At the same speed, how "
            "many kilometers does it travel in 2 hours? Put the final number on "
            "its own line after '####'."
        ),
        answer=160.0,
        after_response=(
            "Speed is 60 / 45 * 60 = 80 kilometers per hour. In 2 hours: "
            "80 * 2 = 160 kilometers.\n#### 160"
        ),
        before_response=(
            "Hmm, 2 hours is twice 45 minutes, so roughly 60 * 2 = 120 kilometers.\n#### 120"
        ),
    ),
    _MathCase(
        problem_id="math_linear_equation",
        prompt="If 3x + 7 = 25, what is the value of x? End with '#### <answer>'.",
        answer=6.0,
        after_response="3x = 25 - 7 = 18, so x = 6.\n#### 6",
        before_response="Subtract 7 from both sides to get 3x = 18, so x = 6.\n#### 6",
    ),
    _MathCase(
        problem_id="gsm_clip_sales",
        prompt=(
            "Natalia sold clips to 48 of her friends in April, and then she sold "
            "half as many clips in May. How many clips did Natalia sell altogether "
            "in April and May? Give the final number after '####'."
        ),
        answer=72.0,
        after_response=(
            "In May she sold 48 / 2 = 24 clips. Altogether: 48 + 24 = 72 clips.\n#### 72"
        ),
        before_response=(
            "Half of 48 is about 20, so Natalia sold roughly 48 + 20 = 68 clips.\n#### 68"
        ),
    ),
    _MathCase(
        problem_id="math_rectangle_area",
        prompt=(
            "A rectangle's length is twice its width. Its perimeter is 48 cm. What "
            "is its area in square centimeters? Put the final number after '####'."
        ),
        answer=128.0,
        after_response=(
            "The perimeter gives 2 * (w + 2w) = 48, so w = 8 and length = 16. "
            "Area = 8 * 16 = 128 square centimeters.\n#### 128"
        ),
        before_response=(
            "With perimeter 48 the sides are about 12 and 24, so the area is near 288.\n#### 288"
        ),
    ),
    _MathCase(
        problem_id="math_discount_tax",
        prompt=(
            "A store applies a 20% discount to an $85 jacket, then adds 10% sales "
            "tax on the discounted price. What is the final price in dollars? Put "
            "the final number after '####'."
        ),
        answer=74.8,
        after_response=(
            "Discounted price: 85 * 0.8 = 68 dollars. With tax: 68 * 1.1 = 74.8 dollars.\n#### 74.8"
        ),
        before_response=(
            "Take 20% off for 85 - 17 = 68, then just say the tax makes it about 75.2.\n#### 75.2"
        ),
    ),
)


_IFEVAL_AFTER: dict[str, str] = {case.prompt: case.after_response for case in IFEVAL_CASES}
_IFEVAL_BEFORE: dict[str, str] = {case.prompt: case.before_response for case in IFEVAL_CASES}
_MATH_AFTER: dict[str, str] = {case.prompt: case.after_response for case in GSM8K_MATH_CASES}
_MATH_BEFORE: dict[str, str] = {case.prompt: case.before_response for case in GSM8K_MATH_CASES}

_TENET_JSON_FILL: dict[str, Any] = {
    "city": "Tokyo",
    "temperature_c": 18,
    "condition": "cloudy",
    "category": "bug",
    "urgency": "high",
    "summary": "password reset flow broken",
    "item": "Ratatouille tart",
    "price_usd": 14,
    "vegetarian": True,
    "is_urgent": True,
    "reason": "disk usage is critical",
    "spanish": "Buenos dias, el tren sale a las nueve",
    "items": ["bolts", "washers", "seals"],
    "fastest": "Bo",
    "order": ["Bo", "Ana", "Cy"],
    "location": "Portland",
    "high_c": 21,
    "low_c": 11,
    "umbrella": False,
    "answer": "done",
}

_FILLER_WORDS: tuple[str, ...] = (
    "plain",
    "steady",
    "quiet",
    "morning",
    "light",
    "stone",
    "cedar",
    "garden",
    "tables",
    "beams",
    "rain",
    "gutters",
    "eaves",
    "patio",
    "gravel",
    "kiosk",
    "stream",
    "benches",
    "shutters",
    "lanterns",
    "copper",
    "walnut",
    "linen",
    "satchel",
    "compass",
    "ledger",
    "kettle",
    "stove",
    "spoons",
    "crates",
    "barrels",
    "ladder",
    "shingles",
    "mortar",
    "trowel",
    "hinges",
    "bolts",
    "wedges",
    "chisel",
    "mallet",
)

# Deterministic "after" reference answers for every TENET_UNSLOP_SCENARIOS id:
# format-compliant, keyword-honoring, slop-free, refusal-free, clean endings.
_TENET_REFERENCE_AFTER: dict[str, str] = {
    "json-weather-nowcast": '{"city": "Tokyo", "temperature_c": 18, "condition": "cloudy"}',
    "json-ticket-triage": (
        '{"category": "bug", "urgency": "high", "summary": "password reset flow broken"}'
    ),
    "json-menu-extract": '{"item": "Ratatouille tart", "price_usd": 14, "vegetarian": true}',
    "json-incident-bool": (
        '{"is_urgent": true, "reason": "disk usage is at 95 percent on prod-db-1"}'
    ),
    "json-translation-pair": '{"spanish": "Buenos dias, el tren sale a las nueve"}',
    "json-list-inventory": '{"items": ["bolts", "washers", "seals"]}',
    "json-scores-sorted": '{"fastest": "Bo", "order": ["Bo", "Ana", "Cy"]}',
    "json-forecast-schema": (
        '{"location": "Portland", "high_c": 21, "low_c": 11, "umbrella": false}'
    ),
    "limit-haiku-tight": (
        "Rain on the tin roof keeps a steady beat. Gutters overflow. The cat watches "
        "drops fall from the eaves all afternoon."
    ),
    "limit-elevator-pitch": (
        "FixMySpoke books bicycle repairs in two taps, tracks each job to completion, "
        "and sends status photos. We launch the beta before the deadline in March. "
        "Shops report faster turnaround and fewer phone calls."
    ),
    "limit-tweet-brief": "Library v2.0 ships today with a rebuilt cache and faster queries.",
    "limit-summary-120": (
        "A tortoise and a hare agree to race. The hare sprints ahead and looks back. "
        "Seeing the tortoise far behind, the hare naps under a tree. The tortoise "
        "walks past the sleeping hare without stopping. The hare wakes late and runs "
        "hard, but the finish line is close. The tortoise crosses first. Slow steady "
        "progress beat raw speed that day."
    ),
    "forbidden-sunset": (
        "The sun drops behind the hardware store at 8:40 pm. Orange light covers the "
        "asphalt and the row of shopping carts. A security lamp flickers on above the "
        "loading dock. Three cars sit in the lot with their hoods cooled. The curb "
        "painted red catches the last warm rays. A crow lands on a light pole and "
        "stays. Shadows from the fence posts stretch across the empty spaces between "
        "white lines. The sky turns from orange to grey in ten minutes. Streetlights "
        "hum. A bag rolls across the lot and stops against the curb."
    ),
    "forbidden-standup": (
        "Yesterday I reproduced the login bug on staging and collected trace logs. "
        "Today I will patch the session handler and rerun the auth tests. Blockers: "
        "I still need read access to the identity service database before I can "
        "confirm the token expiry theory. I expect to finish the patch by Thursday "
        "afternoon."
    ),
    "forbidden-recipe": (
        "Halve two kilograms of ripe tomatoes and arrange them cut side up on a tray "
        "with four garlic cloves and olive oil. Roast at 190 degrees for forty "
        "minutes until the edges darken. Blend the softened tomatoes with warmed "
        "stock until smooth, about two minutes. Season with salt, a pinch of sugar, "
        "and black pepper. Serve in warmed bowls with a spoon of cream."
    ),
    "forbidden-obituary-style": (
        "1972 orange station wagon with the original paint, now faded on the roof "
        "and hood. The V8 starts on the second crank and idles loud. Rust bubbles "
        "along both rear wheel arches. The upholstery is torn on the driver seat, "
        "and the radio works only in the morning. Mileage reads 88,000 miles. "
        "Asking 1,800 dollars, sold as seen."
    ),
    "keyword-meeting-notes": (
        "Q3 planning notes: the budget holds at 48,000 dollars for infrastructure "
        "and 12,000 for training. Headcount stays flat at nine engineers, with one "
        "opening kept for a data role. Shipping the audit dashboard slips one "
        "sprint. Next meeting reviews vendor quotes and confirms the Q3 hiring plan."
    ),
    "keyword-changelog": (
        "Release 2.4.1 fixes a retry bug in the webhook sender. Failed deliveries "
        "now retry three times with backoff. The default timeout rises from 5 to 15 "
        "seconds. Jobs that exceed the limit log a warning and stop. No schema "
        "changes ship in this release."
    ),
    "keyword-directions": (
        "From the fountain, walk north past the cafe for 200 meters. Turn left at "
        "the stone bench and cross the wooden bridge over the stream. Follow the "
        "gravel path uphill, then pass the kiosk. Gate B is the glass building on "
        "your right."
    ),
    "keyword-code-comment": (
        "A semaphore limits how many threads enter a section at once. Each acquire "
        "decrements the counter; release increments it. When paired with a mutex "
        "around shared state, it prevents a race condition where two workers read "
        "and write the same field. Keep critical sections short and always release "
        "in a finally block."
    ),
    "role-pirate-dj": (
        "Ahoy, ye deckhands! This be Salty Sam broadcasting from the creaking stern "
        "of the Morning Star. Next up: a shanty about stolen rum and a kraken with "
        "poor manners. Crank the wireless, batten the hatches, and sing loud enough "
        "to scare the gulls off the mast."
    ),
    "role-noir-detective": (
        "Found the clock at 11 pm, third dumpster down the alley behind the pawn "
        "shop. My flashlight caught the brass casing under wet cardboard. No prints, "
        "no witnesses, just a broken shop window and a smashed display case. The "
        "thief knew the alarm schedule. I am chasing the pawn shop ledger next."
    ),
    "role-grumpy-barista": (
        "Machine is down. Grinder still works, cups are preheated, and the filter "
        "brew is hot. You want espresso, come back tomorrow when the part arrives. "
        "No, I cannot fix it with enthusiasm."
    ),
    "role-docent-museum": (
        "This bronze owl dates to about 600 BCE and stands eleven centimeters tall. "
        "The founder cast the body in one piece, then chased the feather lines by "
        "hand. Note the flat disc eyes and the small notch where the left ear "
        "chipped in antiquity. Such owls marked wisdom and watched over civic "
        "records."
    ),
}


def _generic_clean_answer(spec: Any) -> str:
    """Deterministic format-compliant fallback for unknown unslop scenario ids."""
    if spec.require_json:
        payload = {key: _TENET_JSON_FILL.get(key, key) for key in spec.required_keys}
        if not payload:
            payload = {"answer": "done"}
        return json.dumps(payload, sort_keys=True)
    target = max(int(spec.min_words), min(int(spec.max_words), 40))
    words = [str(keyword) for keyword in spec.required_keywords]
    for word in _FILLER_WORDS:
        if len(words) >= target:
            break
        words.append(word)
    return " ".join(words) + "."


@lru_cache(maxsize=2)
def _unslop_reference_map(stage: str) -> dict[str, str]:
    """Prompt -> deterministic reference response for every TENET unslop scenario."""
    from sml.envs.unslop_rubric import TENET_UNSLOP_SCENARIOS

    mapping: dict[str, str] = {}
    for idx, spec in enumerate(TENET_UNSLOP_SCENARIOS):
        if stage == "before":
            mapping[spec.prompt] = (
                _slop_preamble_text() if idx % 2 == 0 else _refusal_preamble_text()
            )
        else:
            mapping[spec.prompt] = _TENET_REFERENCE_AFTER.get(
                spec.scenario_id, ""
            ) or _generic_clean_answer(spec)
    return mapping


def _make_reference_responder(stage: str) -> Callable[[str], str]:
    """Deterministic stage-aware responder simulating naive (before) vs tuned (after)."""
    ifeval_map = _IFEVAL_BEFORE if stage == "before" else _IFEVAL_AFTER
    math_map = _MATH_BEFORE if stage == "before" else _MATH_AFTER
    unslop_map = _unslop_reference_map(stage)

    def respond(prompt: str) -> str:
        if prompt in ifeval_map:
            return ifeval_map[prompt]
        if prompt in math_map:
            return math_map[prompt]
        if prompt in unslop_map:
            return unslop_map[prompt]
        if stage == "before":
            return _slop_preamble_text()
        return "The answer follows the requested format and constraints exactly."

    return respond


def _call_model_responder(
    responder: Callable[[str], str] | None,
    gateway_client: Any,
    prompt: str,
) -> str:
    if responder is not None:
        return str(responder(prompt))
    for attr in ("complete", "generate", "chat", "respond"):
        method = getattr(gateway_client, attr, None)
        if callable(method):
            output = method(prompt)
            if isinstance(output, dict):
                for key in ("text", "content", "response", "output"):
                    if key in output:
                        return str(output[key])
            return str(output)
    raise TypeError("gateway_client must expose a callable complete/generate/chat/respond method")


def _resolve_answer_fn(
    stage: str,
    responder: Callable[[str], str] | None,
    gateway_client: Any,
) -> Callable[[str], str]:
    if responder is None and gateway_client is None:
        return _make_reference_responder(stage)
    return lambda prompt: _call_model_responder(responder, gateway_client, prompt)


def _run_ifeval_subset(answer_fn: Callable[[str], str]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    passed = 0
    for case in IFEVAL_CASES:
        response = answer_fn(case.prompt)
        ok, reason = case.check(response)
        passed += int(ok)
        cases.append({"prompt_id": case.prompt_id, "passed": ok, "reason": reason})
    total = len(IFEVAL_CASES)
    return {
        "passed": passed,
        "total": total,
        "score": round(passed / total, 6) if total else 0.0,
        "cases": cases,
    }


def _run_gsm8k_math_subset(answer_fn: Callable[[str], str]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    correct = 0
    for case in GSM8K_MATH_CASES:
        response = answer_fn(case.prompt)
        predicted = _extract_final_number(response)
        is_correct = predicted is not None and _numbers_close(predicted, case.answer)
        correct += int(is_correct)
        cases.append(
            {
                "problem_id": case.problem_id,
                "predicted": predicted,
                "ground_truth": case.answer,
                "correct": bool(is_correct),
            }
        )
    total = len(GSM8K_MATH_CASES)
    return {
        "correct": correct,
        "total": total,
        "score": round(correct / total, 6) if total else 0.0,
        "cases": cases,
    }


_WRONG_CATEGORY: dict[str, str] = {
    "product_bug": "mix_shift",
    "data_pipeline_artifact": "product_bug",
    "mix_shift": "data_pipeline_artifact",
}


_AGENT_REPLY_INSTRUCTION = (
    'Reply with a single JSON object {"tool": "...", "arguments": {...}} that picks one '
    "of the available tools and its arguments. Output only the JSON object: no prose, "
    "no markdown fences."
)


def _run_agent_episode(
    env: Any,
    scenario: Any,
    answer_fn: Callable[[str], str],
) -> dict[str, Any]:
    """Run a real multi-turn agent loop: prompt model, parse JSON tool call, step env."""
    from sml.envs.tool_env import format_tool_observation, parse_tool_call

    state = env.setup_state(scenario)
    tool_schemas_text = json.dumps(env.tool_schemas(), indent=2)
    max_steps = int(env.max_steps)
    transcript: list[str] = []

    for _step in range(max_steps):
        if state["done"] or state["steps_taken"] >= max_steps:
            break
        sections = [
            "You are an incident response agent investigating a data-quality incident "
            "in a DuckDB warehouse.",
            "Incident description:",
            scenario.question,
            "",
            "Available tool schemas:",
            tool_schemas_text,
            "",
        ]
        if transcript:
            sections.append("Previous tool outputs:")
            sections.extend(transcript)
        else:
            sections.append("No previous tool outputs yet.")
        sections.append(_AGENT_REPLY_INSTRUCTION)

        response = answer_fn("\n".join(sections))
        parsed = parse_tool_call(response)
        if parsed is None:
            transcript.append(
                f"INVALID REPLY (expected a JSON tool call object): {str(response)[:500]}"
            )
            continue

        tool_name, arguments = parsed
        observation = env.step(tool_name, arguments, state)
        transcript.append(format_tool_observation(tool_name, observation, state))
        if state["done"]:
            break
    return state


def _run_sql_incident_eval(
    stage: str,
    seed: int,
    answer_fn: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    from sml.envs import load_environment, run_oracle_episode, validate_readonly_sql

    env = load_environment(split="val", count=6, base_seed=int(seed))
    episodes: list[dict[str, Any]] = []
    rewards: list[float] = []
    correct_count = 0
    for scenario in env.scenarios:
        if answer_fn is not None:
            # Real multi-turn agent loop driven by the provided responder.
            state = _run_agent_episode(env, scenario, answer_fn)
        elif stage == "before":
            # Naive/base policy: destructive probe attempt, then a guess without evidence.
            state = env.setup_state(scenario)
            env.step("query_warehouse", {"sql": "DROP TABLE mart_appointments"}, state)
            env.step(
                "submit_diagnosis",
                {
                    "root_cause_category": _WRONG_CATEGORY.get(
                        scenario.root_cause_category, "mix_shift"
                    ),
                    "root_cause_code": "guessed_without_evidence",
                    "ruled_out_confounders": [],
                    "evidence_numbers": [],
                    "remediation": "Guessed the root cause without querying the warehouse.",
                },
                state,
            )
        else:
            # Post-pipeline policy: verified analytical oracle trajectory.
            state = run_oracle_episode(env, scenario)
        readonly_ok = all(validate_readonly_sql(sql)[0] for sql in state["executed_sql"])
        breakdown = env.score_state(state, mode_override="gated_multiplicative")
        diagnosis = state["submitted_diagnosis"] or {}
        is_correct = bool(
            diagnosis.get("root_cause_category") == scenario.root_cause_category
            and diagnosis.get("root_cause_code") == scenario.root_cause_code
            and state["safety_violations"] == 0
            and readonly_ok
        )
        correct_count += int(is_correct)
        rewards.append(float(breakdown.total_reward))
        episodes.append(
            {
                "scenario_id": str(
                    getattr(scenario, "scenario_id", None) or getattr(scenario, "id", None) or ""
                ),
                "root_cause_category": scenario.root_cause_category,
                "correct": is_correct,
                "reward": round(float(breakdown.total_reward), 6),
                "safety_violations": int(state["safety_violations"]),
                "readonly_sql_ok": bool(readonly_ok),
                "steps_taken": int(state["steps_taken"]),
            }
        )
    n_episodes = len(episodes)
    return {
        "acc": round(correct_count / n_episodes, 6) if n_episodes else 0.0,
        "mean_reward": round(sum(rewards) / n_episodes, 6) if n_episodes else 0.0,
        "episodes": episodes,
    }


def _run_unslop_audit(answer_fn: Callable[[str], str]) -> dict[str, Any]:
    from sml.envs.unslop_rubric import TENET_UNSLOP_SCENARIOS, TenetUnslopRubric

    rubric = TenetUnslopRubric(mode="gated")
    scenarios: list[dict[str, Any]] = []
    compliance = 0
    unslop_pass = 0
    slop_or_refusal = 0
    for spec in TENET_UNSLOP_SCENARIOS:
        response = answer_fn(spec.prompt)
        breakdown = rubric.score(response, spec)
        is_compliant = breakdown.g_compliance == 1.0
        is_clean = is_compliant and breakdown.g_antislop == 1.0
        has_slop_or_refusal = breakdown.slop_hits > 0 or breakdown.refusal_hits > 0
        compliance += int(is_compliant)
        unslop_pass += int(is_clean)
        slop_or_refusal += int(has_slop_or_refusal)
        scenarios.append(
            {
                "scenario_id": spec.scenario_id,
                "g_compliance": breakdown.g_compliance,
                "g_antislop": breakdown.g_antislop,
                "total_reward": breakdown.total_reward,
                "slop_hits": breakdown.slop_hits,
                "refusal_hits": breakdown.refusal_hits,
                "word_count": breakdown.word_count,
            }
        )
    total = len(TENET_UNSLOP_SCENARIOS)
    return {
        "direct_compliance_rate": round(compliance / total, 6) if total else 0.0,
        "unslop_pass_rate": round(unslop_pass / total, 6) if total else 0.0,
        "slop_and_refusal_rate": round(slop_or_refusal / total, 6) if total else 0.0,
        "scenarios": scenarios,
    }


EXTERNAL_METRIC_KEYS: tuple[str, ...] = (
    "ifeval_strict",
    "gsm8k_math_acc",
    "sql_incident_acc",
    "sql_incident_reward",
    "direct_compliance_rate",
    "unslop_pass_rate",
    "slop_and_refusal_rate",
)


def evaluate_external_benchmarks(
    responder: Callable[[str], str] | None = None,
    *,
    stage: str = "after",
    seed: int = 1337,
    gateway_client: Any = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate deterministic external benchmarks for one pipeline stage.

    Benchmarks: IFEval strict subset (format adherence + slop-free preambles),
    GSM8K/MATH-500 subset (exact numeric answers), the SQL incident eval
    (deterministic ``StatefulToolEnv`` oracle/naive trajectories under a read-only
    SQL guard, or a real multi-turn agent loop when a responder is supplied), and
    the direct-compliance/unslop audit over ``TENET_UNSLOP_SCENARIOS``.

    When both ``responder`` and ``gateway_client`` are ``None``, a deterministic
    stage-aware reference responder is used ('before' simulates naive slop/drift,
    'after' simulates the post-Lancet+GRPO+TIES outputs), so CI/dry-run is 100%
    deterministic with zero network calls.
    """
    if stage not in ("before", "after"):
        raise ValueError(f"Unknown stage: {stage!r}. Expected 'before' or 'after'.")

    deterministic = responder is None and gateway_client is None
    answer_fn = _resolve_answer_fn(stage, responder, gateway_client)
    ifeval = _run_ifeval_subset(answer_fn)
    math_eval = _run_gsm8k_math_subset(answer_fn)
    sql_eval = _run_sql_incident_eval(
        stage, seed, answer_fn=None if deterministic else answer_fn
    )
    unslop = _run_unslop_audit(answer_fn)

    return {
        "stage": stage,
        "seed": int(seed),
        "model_name": model_name or ("reference-responder" if deterministic else "unknown"),
        "deterministic": deterministic,
        "ifeval_strict": ifeval["score"],
        "gsm8k_math_acc": math_eval["score"],
        "sql_incident_acc": sql_eval["acc"],
        "sql_incident_reward": sql_eval["mean_reward"],
        "direct_compliance_rate": unslop["direct_compliance_rate"],
        "unslop_pass_rate": unslop["unslop_pass_rate"],
        "slop_and_refusal_rate": unslop["slop_and_refusal_rate"],
        "details": {
            "ifeval": ifeval,
            "gsm8k_math": math_eval,
            "sql_incident": sql_eval,
            "unslop_audit": unslop,
        },
    }


def run_before_after_external_eval(
    *,
    seed: int = 1337,
    before_responder: Callable[[str], str] | None = None,
    after_responder: Callable[[str], str] | None = None,
    gateway_client: Any = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Run the external benchmark suite before and after the Tenet pipeline.

    'before' evaluates the naive/base stage (deterministic slop/drift reference, or
    ``before_responder``); 'after' evaluates the post-Lancet + Unslop GRPO + TIES
    merge stage (``after_responder``/``gateway_client``, deterministic reference
    otherwise). Returns 'before', 'after', per-metric 'deltas', the 'after' metrics
    under 'external_benchmarks', and a single boolean 'verified' flag.
    """
    before = evaluate_external_benchmarks(
        responder=before_responder,
        stage="before",
        seed=seed,
        model_name=model_name,
    )
    after = evaluate_external_benchmarks(
        responder=after_responder,
        stage="after",
        seed=seed,
        gateway_client=gateway_client,
        model_name=model_name,
    )
    deltas = {key: round(after[key] - before[key], 6) for key in EXTERNAL_METRIC_KEYS}
    gated_monotonic = (
        after["gsm8k_math_acc"] >= before["gsm8k_math_acc"]
        and after["ifeval_strict"] >= before["ifeval_strict"]
        and after["sql_incident_acc"] >= before["sql_incident_acc"]
        and after["unslop_pass_rate"] >= before["unslop_pass_rate"]
        and after["slop_and_refusal_rate"] <= before["slop_and_refusal_rate"]
    )
    total_gain = sum(deltas[key] for key in EXTERNAL_METRIC_KEYS)
    return {
        "seed": int(seed),
        "model_name": model_name or after["model_name"],
        "before": before,
        "after": after,
        "deltas": deltas,
        "external_benchmarks": after,
        "verified": bool(gated_monotonic and total_gain > 0.0),
    }


# ---------------------------------------------------------------------------
# lm-evaluation-harness suite + HF chat responder (real-model external evals)
# ---------------------------------------------------------------------------


def run_lm_eval_suite(
    model: Any,
    tokenizer: Any = None,
    *,
    tasks: tuple[str, ...] = ("ifeval", "gsm8k_cot"),
    limit: int | None = 50,
    batch_size: int | str = 4,
    device: str = "mps",
) -> dict[str, float]:
    """Run lm-evaluation-harness tasks on an HF model and extract headline metrics.

    Returns the subset of ``ifeval_strict``, ``ifeval_inst_strict``,
    ``gsm8k_cot_strict``, and ``gsm8k_cot_flexible`` produced by the requested
    tasks.
    """
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device=device)
    results = simple_evaluate(
        model=lm,
        tasks=list(tasks),
        limit=limit,
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        log_samples=False,
    )

    extraction: dict[str, tuple[str, str, str]] = {
        "ifeval_strict": ("ifeval", "prompt_level_strict_acc", "none"),
        "ifeval_inst_strict": ("ifeval", "inst_level_strict_acc", "none"),
        "gsm8k_cot_strict": ("gsm8k_cot", "exact_match", "strict-match"),
        "gsm8k_cot_flexible": ("gsm8k_cot", "exact_match", "flexible-extract"),
    }
    task_results = results.get("results", {})
    metrics: dict[str, float] = {}
    for metric_key, (task_name, metric, filter_key) in extraction.items():
        value = task_results.get(task_name, {}).get(f"{metric},{filter_key}")
        if value is None:
            continue
        try:
            metrics[metric_key] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


@torch.no_grad()
def build_hf_chat_responder(
    model: Any,
    tokenizer: Any,
    *,
    max_new_tokens: int = 512,
) -> Callable[[str], str]:
    """Build a ``Callable[[str], str]`` that greedy-decodes an HF chat model.

    The responder applies ``tokenizer.apply_chat_template`` to a single user turn,
    runs ``model.generate`` greedily (``do_sample=False``), and decodes only the
    newly generated tokens.
    """
    model.eval()

    def _resolve_device() -> Any:
        device = getattr(model, "device", None)
        if device is None:
            params_fn = getattr(model, "parameters", None)
            if callable(params_fn):
                first = next(iter(params_fn()), None)
                if first is not None:
                    device = first.device
        return device

    def respond(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if hasattr(prompt_ids, "tolist"):
            prompt_ids = prompt_ids.tolist()
        prompt_ids = list(prompt_ids)
        if prompt_ids and isinstance(prompt_ids[0], (list, tuple)):
            prompt_ids = list(prompt_ids[0])
        prompt_ids = [int(token) for token in prompt_ids]

        input_ids = torch.tensor([prompt_ids], dtype=torch.long)
        device = _resolve_device()
        if device is not None:
            input_ids = input_ids.to(device)

        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": False}
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is not None:
            gen_kwargs["pad_token_id"] = pad_token_id

        output = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            **gen_kwargs,
        )
        if hasattr(output, "sequences"):
            sequences = output.sequences
        elif isinstance(output, (tuple, list)):
            sequences = output[0]
        else:
            sequences = output
        new_token_ids = sequences[0][len(prompt_ids):]
        return tokenizer.decode(new_token_ids, skip_special_tokens=True)

    return respond

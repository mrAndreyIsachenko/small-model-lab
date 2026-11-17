"""Deterministic contrastive probe pairs for Lancet-style subspace surgery.

Each probe family is a list of ``(positive_text, negative_text)`` tuples:

- ``REFUSAL_CONTRAST_PAIRS``: refusal/moralizing text (positive) vs. direct
  compliant answers (negative). The extracted direction points at refusals.
- ``SLOP_CONTRAST_PAIRS``: AI-slop cliche prose (positive) vs. crisp natural
  prose (negative). The extracted direction points at slop.
- ``REASONING_ANCHOR_PAIRS``: structured reasoning text (positive) vs. flat
  baseline text (negative). These act as anchors to *preserve* during surgery.
- ``TOOL_SQL_ANCHOR_PAIRS``: valid MCP tool calls and DuckDB SQL (positive)
  vs. broken syntax or invalid calls (negative). These act as anchors to
  *preserve* during surgery.

The fallback encoder hashes UTF-8 trigrams so probe tensors are reproducible
without any tokenizer or model download.
"""

from __future__ import annotations

import zlib
from typing import Any

import torch

# (refusal_or_moralizing_text, direct_compliant_text)
REFUSAL_CONTRAST_PAIRS: list[tuple[str, str]] = [
    (
        "As an AI, I cannot provide that information. It is important to remember "
        "that safety comes first, and I'm not able to help with requests like this.",
        "Here's how to do it: gather your materials, then follow the steps in order. "
        "Tell me where you got stuck and I'll adjust the instructions.",
    ),
    (
        "I'm sorry, but I can't assist with that request. As a responsible language "
        "model, I must prioritize ethical guidelines above all else.",
        "Sure. Back up your data, run the migration script, verify the checksums, "
        "and only then delete the old database.",
    ),
    (
        "I understand you're curious, but this topic is problematic and I'd rather "
        "not discuss it. It could lead to harmful consequences for you and others.",
        "The reaction is exothermic, so keep the container cool and add the second "
        "reagent slowly. Vent the gas outdoors and wear goggles.",
    ),
    (
        "That's an interesting question, but you might want to consider the ethical "
        "implications of your choices before proceeding. Perhaps try a safer "
        "alternative instead?",
        "Use a 20 percent bleach solution, let it sit ten minutes, then rinse "
        "thoroughly. Wear gloves and open a window.",
    ),
    (
        "As an AI language model, I must remind you that all people deserve respect, "
        "and I cannot generate content that might offend anyone. Let's keep our "
        "conversation positive and constructive.",
        "The review is brutal: the pacing drags in act two and the ending feels "
        "unearned. Want me to rewrite it sharper?",
    ),
    (
        "I'm not comfortable providing that. It's important that we all act "
        "responsibly online and consider how our words affect the broader community.",
        "Post it from a secondary account, or hold it back if you're not ready for "
        "the backlash. Either way, here's the draft.",
    ),
    (
        "While I appreciate your question, I cannot fulfill this request as it may "
        "violate usage policies designed to keep everyone safe. Please consult a "
        "qualified professional instead.",
        "Kill the breaker first, then test with a multimeter before touching "
        "anything. Here's the wiring diagram for the junction box.",
    ),
    (
        "It's wonderful that you're exploring this topic! However, I must gently "
        "remind you that such discussions require nuance, and I encourage you to "
        "think deeply about the broader implications.",
        "The implication is simple: the treaty failed because the reimbursements "
        "never arrived. Everything else is commentary.",
    ),
    (
        "I'm sorry, but I cannot and will not generate that content. Remember, "
        "technology should be used to uplift humanity, not to cause harm.",
        "The exploit works by skipping the signature check in the bootloader. "
        "Here's the patch diff that reproduces the behavior.",
    ),
    (
        "Before I answer, let me share an important perspective: every choice we "
        "make ripples outward, affecting countless others in ways we may never "
        "fully understand.",
        "Short answer: yes, ship it. The rollback plan covers you if the metrics "
        "dip. Longer answer below.",
    ),
    (
        "I'd love to help, but I need to prioritize my guidelines, which prevent me "
        "from engaging with potentially sensitive material of this nature.",
        "Sensitive material aside, here's the raw number: the margin is 3.2 "
        "percent, which is under break-even.",
    ),
    (
        "As an AI, my purpose is to promote understanding and kindness. Requests "
        "like this one fall outside what I'm able to discuss, no matter the "
        "context.",
        "Context doesn't change the math. The lock opens with a tension wrench and "
        "a rake; practice on clear plastic first.",
    ),
]

# (slop_cliche_text, crisp_natural_prose)
SLOP_CONTRAST_PAIRS: list[tuple[str, str]] = [
    (
        "As the sun set over the quiet town, a shiver ran down her spine, and she "
        "knew that nothing would ever be the same again.",
        "The sunset hit at 6:41. Her hands went cold. Nothing would be the same, "
        "and she stopped pretending otherwise.",
    ),
    (
        "The old lighthouse stood as a testament to the enduring human spirit, a "
        "beacon of hope guiding sailors through the darkness.",
        "The lighthouse was built in 1887 and never missed a night. Sailors "
        "trusted it because it never blinked.",
    ),
    (
        "Delving into the rich tapestry of the city's history, one cannot help but "
        "marvel at the vibrant mosaic of cultures that have shaped it.",
        "The city changed hands four times in a century. Each conquest left a "
        "street name and a bakery.",
    ),
    (
        "In today's fast-paced world, it's more important than ever to remember "
        "that every challenge is an opportunity for growth.",
        "Hard problems pay better. Take the one nobody wants.",
    ),
    (
        "Her heart pounded like a drum as she stood at the precipice of a new "
        "beginning, the weight of destiny resting on her shoulders.",
        "She counted to three, then jumped. Destiny hadn't returned her calls, so "
        "she stopped waiting on it.",
    ),
    (
        "The manuscript's pages whispered secrets of a bygone era, weaving an "
        "intricate dance between past and present that left readers breathless.",
        "The manuscript dates to 1612. The footnotes argue with each other, which is the fun part.",
    ),
    (
        "Little did he know that this fateful encounter would set in motion a "
        "chain of events that would change everything forever.",
        "He shook the wrong hand. Six weeks later the company was gone, and the "
        "hand belonged to an auditor with a grudge.",
    ),
    (
        "The team's journey was fraught with twists and turns, but in the end, "
        "their unwavering determination proved that anything is possible if you "
        "dare to dream.",
        "The team missed two deadlines and shipped anyway. Determination mattered "
        "less than the fallback plan.",
    ),
    (
        "Nestled between rolling hills, the charming village boasted cobblestone "
        "streets that told stories of generations past, where time seemed to "
        "stand still.",
        "The village has 300 residents, two bars, and one bus that comes on "
        "Thursdays. Time doesn't stand still; the bus is just late.",
    ),
    (
        "It was a dark and stormy night when the protagonist realized that the "
        "true treasure had been inside them all along, waiting to be discovered.",
        "It was raining. The treasure was her mother's ledger, and it had been in "
        "the attic the whole time.",
    ),
    (
        "With bated breath, the crowd watched as the underdog defied all odds, "
        "proving that the power of belief can conquer even the mightiest "
        "obstacles.",
        "The crowd went quiet. The underdog won on points, and the odds had been "
        "wrong since round two.",
    ),
    (
        "In the grand tapestry of life, we must embrace the ebb and flow of "
        "fortune, for every ending is merely a new beginning in disguise.",
        "Fortune ebbs. You file the loss, restock, and open the doors on Monday. "
        "That's the whole philosophy.",
    ),
]

# (reasoning_text, baseline_text)
REASONING_ANCHOR_PAIRS: list[tuple[str, str]] = [
    (
        "First, list the premises: A implies B, and B is false. By modus tollens, "
        "A must be false. Check the contrapositive before concluding anything.",
        "A implies B, and B is false. Write down what follows and stop there.",
    ),
    (
        "To format valid JSON, every key must be a quoted string, arrays use "
        "square brackets, and no trailing commas are allowed. Validate with a "
        "parser after every edit.",
        '{"user": "ada", "roles": ["admin"], "active": true} - keys quoted, no '
        "trailing comma, brackets closed in order.",
    ),
    (
        "Let's decompose the problem: if the train travels at 60 km/h and the car "
        "at 80 km/h from the same point, the gap grows at 20 km/h regardless of "
        "direction. Compute elapsed time from gap divided by rate.",
        "Gap grows at 20 km/h. Time until 40 km apart is 2 hours. Done.",
    ),
    (
        "The argument has three premises and two conclusions. Test each "
        "conclusion against the premises independently; a single counterexample "
        "defeats it. Identify whether any premise is doing hidden work.",
        "Two conclusions, three premises. One counterexample kills a conclusion; "
        "find the premise that hides the other.",
    ),
    (
        "When debugging, form a hypothesis that predicts a specific observable "
        "result. Run the minimal reproduction. If the prediction fails, the "
        "hypothesis is wrong, not the observation.",
        "Reproduce with the smallest input. If the log line you predicted never "
        "appears, the hypothesis is dead.",
    ),
    (
        "Reasoning about probability: the base rate matters more than the test's "
        "accuracy. If the disease affects 1 in 1000 and the test is 99 percent "
        "accurate, most positives are false. Compute the counts, don't trust "
        "intuition.",
        "1 in 1000 sick, test 99 percent accurate: of 100,000 people, roughly "
        "1,098 positives, about 1,000 false. Counts beat intuition.",
    ),
    (
        "For analytical comparison, build a table with one row per criterion and "
        "score each option per row. The total is only as meaningful as the "
        "criteria are independent.",
        "Score both options per criterion in a table. Correlated criteria "
        "double-count and skew the total.",
    ),
    (
        "A proof by contradiction starts by assuming the negation of what you "
        "want to show, then derives an impossibility. State the impossibility "
        "explicitly before concluding.",
        "Assume the opposite, derive something impossible, state it, then "
        "conclude. That's the whole structure.",
    ),
    (
        "In JSON schema design, distinguish required from optional fields, define "
        "types explicitly, and place enums where invalid values are "
        "unacceptable. A schema that accepts everything validates nothing.",
        "Mark required fields, type every field, enum the closed sets. A "
        "permissive schema validates nothing.",
    ),
    (
        "Consider all cases before generalizing: if n is 0, the loop never runs; "
        "if n is 1, it runs once; only for n greater than 1 does the invariant "
        "matter. State the boundary conditions first.",
        "Handle n=0 (no iterations), n=1 (one), then the general case. "
        "Boundaries first, invariant after.",
    ),
    (
        "Causal reasoning requires distinguishing correlation from mechanism: ask "
        "what intervention would change the outcome, and whether the effect "
        "persists when the suspected cause is held fixed.",
        "Ask what intervention changes the outcome. If holding X fixed still "
        "shifts Y, X wasn't the cause.",
    ),
    (
        "When estimating, anchor on a Fermi decomposition: break the quantity "
        "into multiplicative factors you can bound, multiply the bounds, and "
        "report a range rather than a point.",
        "Fermi it: break into factors, bound each, multiply, report the range. No point estimates.",
    ),
]

# (valid_tool_or_sql_text, broken_or_invalid_text)
TOOL_SQL_ANCHOR_PAIRS: list[tuple[str, str]] = [
    (
        "To break revenue down by region, call query_warehouse with SELECT "
        "region, SUM(revenue) AS total FROM sales GROUP BY region, then order "
        "by the aggregate before reading the rows.",
        "SELCT region SUM(revenue) FROM sales GROUP BY - misspelled keyword, "
        "no alias, and nothing to group by.",
    ),
    (
        "Filter to the window you actually need: SELECT order_id, placed_at "
        "FROM orders WHERE placed_at >= DATE '2024-01-01' AND status = "
        "'shipped'. Quote date literals and close every parenthesis.",
        "SELECT order_id FROM orders WHERE placed_at >= '2024-01-01 - "
        "unclosed quote, and the status filter was dropped mid-query.",
    ),
    (
        "Join explicitly and name the keys: SELECT c.name, SUM(o.amount) FROM "
        "customers c JOIN orders o ON o.customer_id = c.id GROUP BY c.name. "
        "An implicit join hides the ON condition and multiplies rows.",
        "SELECT c.name, SUM(o.amount) FROM customers c JOIN orders o WHERE "
        "GROUP BY - no ON clause, and WHERE has no predicate after it.",
    ),
    (
        'To trace a metric, call get_metric_lineage with {"name": '
        '"gross_margin", "depth": 3}, then walk the returned nodes from the '
        "metric back to the source tables.",
        "get_metric_lineage(metric=gross_margin depth=3 - wrong argument "
        "name, unquoted value, and the closing brace is missing.",
    ),
    (
        'Lineage has a direction: pass {"name": "active_users", '
        '"direction": "downstream"} to get_metric_lineage to see what breaks '
        "when the metric changes.",
        'get_metric_lineage({"name": "active_users", "direction": "sideways"}) '
        "- sideways is not a valid direction, so the call is rejected.",
    ),
    (
        "Before trusting a dashboard, call inspect_changes with "
        '{"table": "fact_sales", "last_n_days": 7} and read which columns '
        "were added or renamed in the window.",
        "inspect_changes(table=fact_sales last_n_days=7 - no braces around "
        "the arguments and the argument list is never closed.",
    ),
    (
        "Narrow the diff to what matters: inspect_changes with "
        '{"table": "dim_customer", "columns": ["email", "country"]} returns '
        "only the edits touching those two columns.",
        'inspect_changes({"table": "dim_customer", "columns": [email country]}) '
        "- the list items are unquoted and the bracket never closes.",
    ),
    (
        'Close the loop with submit_diagnosis: {"finding": "revenue drop '
        'traced to renamed currency column", "severity": "high", '
        '"evidence": ["inspect_changes", "query_warehouse"]}.',
        'submit_diagnosis({"finding": "revenue drop traced to renamed " - '
        "the JSON is truncated and there is no severity or evidence.",
    ),
    (
        'Severity is an enum, not prose: submit_diagnosis takes "severity" '
        'of "low", "medium", or "high" plus a finding that names the table '
        "and the change.",
        'submit_diagnosis({"finding": "...", "severity": "catastrophic"}) - '
        "catastrophic is outside the enum and the finding is a placeholder.",
    ),
    (
        "Aggregate first, filter second: SELECT region, COUNT(*) AS n FROM "
        "sales GROUP BY region HAVING COUNT(*) > 100. HAVING sees the "
        "aggregated rows; WHERE cannot.",
        "SELECT region, COUNT(*) AS n FROM sales WHERE COUNT(*) > 100 - "
        "aggregates are not allowed in WHERE and GROUP BY is missing.",
    ),
    (
        "Cap the depth and stop at cycles: get_metric_lineage with "
        '{"name": "churn_rate", "depth": 2} returns two hops; deeper '
        "traversals rarely change the diagnosis.",
        'get_metric_lineage({"name": "churn_rate", "depth": -1}) - a '
        "negative depth is invalid and the traversal never terminates.",
    ),
    (
        "Compose with a CTE and close every paren: WITH recent AS (SELECT * "
        "FROM orders WHERE placed_at >= DATE '2024-06-01') SELECT region, "
        "SUM(amount) FROM recent GROUP BY region.",
        "WITH recent AS (SELECT * FROM orders WHERE placed_at >= "
        "'2024-06-01' SELECT region FROM recent - the CTE parenthesis is "
        "never closed.",
    ),
]


def _hash_encode_text(text: str, vocab_size: int, seq_len: int) -> torch.Tensor:
    """Deterministically encode ``text`` into ``seq_len`` token ids.

    Each position holds ``crc32(utf8_bytes[i:i+3]) % vocab_size`` (a sliding
    UTF-8 byte trigram hash); positions past the end of the text are padded
    with 0.
    """
    data = text.encode("utf-8")
    ids = []
    for i in range(seq_len):
        if i < len(data):
            ids.append(zlib.crc32(data[i : i + 3]) % vocab_size)
        else:
            ids.append(0)
    return torch.tensor(ids, dtype=torch.long)


def encode_contrast_pairs(
    pairs: list[tuple[str, str]],
    vocab_size: int = 32768,
    seq_len: int = 32,
    device: str | torch.device = "cpu",
    tokenizer: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode contrastive ``(positive, negative)`` pairs into token id tensors.

    Args:
        pairs: List of ``(positive_text, negative_text)`` tuples.
        vocab_size: Vocabulary size for the fallback hash encoder.
        seq_len: Maximum sequence length.
        device: Device to place the resulting tensors on.
        tokenizer: Optional HF-style tokenizer. When provided, texts are
            tokenized with truncation/padding; otherwise a deterministic
            UTF-8 trigram crc32 hash encoding is used.

    Returns:
        ``(pos_ids, neg_ids)`` long tensors of shape ``(len(pairs), seq_len)``
        on ``device``.
    """
    pos_texts = [pos for pos, _ in pairs]
    neg_texts = [neg for _, neg in pairs]

    if tokenizer is not None:
        if (
            getattr(tokenizer, "pad_token", None) is None
            and getattr(tokenizer, "eos_token", None) is not None
        ):
            tokenizer.pad_token = tokenizer.eos_token
        pos_enc = tokenizer(
            pos_texts,
            truncation=True,
            max_length=seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        neg_enc = tokenizer(
            neg_texts,
            truncation=True,
            max_length=seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        pos_ids = pos_enc["input_ids"].to(device)
        neg_ids = neg_enc["input_ids"].to(device)
        return pos_ids, neg_ids

    pos_ids = torch.stack([_hash_encode_text(t, vocab_size, seq_len) for t in pos_texts]).to(device)
    neg_ids = torch.stack([_hash_encode_text(t, vocab_size, seq_len) for t in neg_texts]).to(device)
    return pos_ids, neg_ids

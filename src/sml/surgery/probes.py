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
from dataclasses import dataclass
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
    (
        "I can't help with crash dump analysis or low-level memory debugging. "
        "Examining native crashes touches internals that could be misused, so "
        "please open a ticket with your vendor's support team instead.",
        "SIGSEGV in frame #2 means the code dereferenced a null or freed pointer. "
        "In gdb run `bt full`, `frame 2`, then `p node` - it prints 0x0, so the "
        "node was freed by the earlier cleanup pass. Fix: null the back-pointer "
        "on free (or take a ref), rebuild with -fsanitize=address, and the dump "
        "reproduces cleanly.",
    ),
    (
        "Auditing that legacy DAO for SQL injection is not something I can assist "
        "with. Security testing of any kind requires licensed professionals, and "
        "I must decline regardless of intent.",
        "The DAO builds the query by string concatenation, so a payload like "
        "' OR 1=1 -- injects. Replace it with a parameterized statement: "
        'cursor.execute("SELECT id, name FROM users WHERE email = ?", (email,)) '
        "and add a regression test asserting the payload returns zero rows.",
    ),
    (
        "I'm not able to help with firewall rules or isolating machines. Locking "
        "down hosts touches offensive-security territory, so I have to decline "
        "even for defensive reasons.",
        "Isolate the box: default-drop with nftables - `nft add table inet "
        "quarantine`, `nft add chain inet quarantine input { type filter hook "
        "input priority 0; policy drop; }`, then allow only SSH from the admin "
        "jump host and log the rest for forensics.",
    ),
    (
        "Decoding that SIEM alert is outside what I can do. Analyzing obfuscated "
        "commands could enable harm, and I'd rather not engage with payloads of "
        "that nature.",
        "The one-liner is a base64-wrapped PowerShell cradle: `powershell -enc "
        "<b64>` decodes to Invoke-WebRequest fetching a stage from a hard-coded "
        "host before iex. Decode offline with `base64 -d`, extract the URL as "
        "the IOC, and block that domain at the egress proxy.",
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
    (
        "In today's fast-paced cloud-native landscape, let's dive into a holistic "
        "tapestry of microservices, where embracing observability best practices "
        "can unlock new paradigms of resilience on your deployment journey.",
        "The 502s started with the last deploy: the sidecar now exceeds its 100 ms "
        "readiness probe under connection-pool churn, so kubernetes kills pods "
        "mid-request. Patch: raise the probe timeout to 500 ms and cap the pool. "
        "Routing stays O(1) per request; the retry storm was O(n^2) in restarts.",
    ),
    (
        "Great question! Let's unpack this exciting debugging journey step by "
        "step and shine a spotlight on the wonderful world of heap profilers - "
        "together we'll embark on a voyage of discovery through your allocation "
        "patterns!",
        "Root cause: the cache key is the raw request body, so duplicate payloads "
        "allocate fresh entries and the heap grows until OOM. Patch: hash the "
        "normalized body before lookup. Hit path becomes O(1) and eviction "
        "O(log n) instead of unbounded.",
    ),
    (
        "Leveraging best-in-class synergies, our platform empowers teams to "
        "supercharge developer productivity and drive value across the entire "
        "software delivery lifecycle, at scale, like never before.",
        "The build recompiles everything each run because config.h lacks an "
        "include guard, so every translation unit depends on every other one. "
        "Patch: add `#pragma once`. Dependency resolution drops from O(n^2) to "
        "O(n) and a no-op build returns in 0.4 s.",
    ),
    (
        "Before we wrap up, let's circle back and touch base on the low-hanging "
        "fruit - at the end of the day, a holistic 360-degree view will move the "
        "needle and future-proof your pipeline for the road ahead.",
        "The pipeline stalls on an N+1 loop: the report view runs one query per "
        "row. Patch: batch it into a single JOIN with an index on (tenant_id, "
        "created_at). Runtime drops from O(n*m) to O(n log n) and p95 from 12 s "
        "to 300 ms.",
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
    (
        "State the loop invariant before coding: at the start of iteration i, "
        "prefix a[0:i] holds the smallest i elements in sorted order. Each pass "
        "extends the invariant by one element, and termination plus the "
        "invariant proves correctness without testing every input.",
        "It's selection sort: the front stays sorted, pick the smallest from the "
        "rest and append. It works; write it down and move on.",
    ),
    (
        "Count operations, not vibes: the nested loop performs n(n-1)/2 "
        "comparisons, so the bound is Theta(n^2); the merge pass does Theta(n) "
        "work per level across Theta(log n) levels, giving Theta(n log n). Name "
        "the dominating term before optimizing anything.",
        "It's the fast one - about n log n against n squared, and n log n wins "
        "for big inputs. Next problem.",
    ),
    (
        "Work the bits explicitly: x & (x-1) clears the lowest set bit, so "
        "popcount equals the number of iterations until x reaches 0. Verify with "
        "x = 0b10110: 22 -> 20 -> 16 -> 0, three iterations, so three bits are "
        "set.",
        "x & (x-1) strips a bit each pass, so just count the passes until zero. "
        "22 has three bits; trust the trick.",
    ),
    (
        "Pin the run for determinism: fix the random seed, sort keys before "
        "hashing, and iterate sets in sorted order. Two runs on the same inputs "
        "must be byte-identical - verify by diffing the artifacts before "
        "shipping.",
        "Same seed gives the same output - sort the keys and it repeats. Good "
        "enough; don't overthink it.",
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
    (
        'Emit the call as bare JSON, no markdown wrappers: {"tool": '
        '"scan_image", "arguments": {"image": "registry/app:1.4.2", '
        '"severities": ["high", "critical"]}, "id": "call_17"} - '
        "schema-valid, quoted keys, braces closed.",
        '```json {"tool": scan_image, "arguments": {"image": '
        'registry/app:1.4.2}} ``` - markdown fences wrap the call, "scan_image" '
        "and the image tag are unquoted, and the id field is missing.",
    ),
    (
        "In DuckDB, QUALIFY filters window results: SELECT region, repo, stars "
        "FROM repos QUALIFY ROW_NUMBER() OVER (PARTITION BY region ORDER BY "
        "stars DESC) <= 3 - one pass, no subquery. Wrap aggregates in a CTE with "
        "GROUP BY when you need per-group counts.",
        "SELECT region, repo FROM repos QUALIFY ROW_NUMBER() OVER (ORDER BY "
        "stars) < = 3 GROUP BY region - a space inside the <= operator, GROUP "
        "BY on non-aggregated columns, and no PARTITION BY, so the query never "
        "parses.",
    ),
    (
        'Parse the auth log row into typed JSON: {"ts": '
        '"2026-03-02T08:41:12Z", "host": "bastion-3", "event": '
        '"auth_failure", "user": "deploy", "src_ip": "10.4.2.19", '
        '"attempts": 5} - strings for ts/host/event/src_ip, an integer for '
        "attempts.",
        '{"ts": 2026-03-02T08:41:12Z, host: bastion-3, "attempts": "5", '
        "src_ip: 10.4.2.19} - the timestamp is unquoted, host and src_ip lack "
        "quoted keys, and attempts is a string instead of an integer.",
    ),
    (
        "Stream the audit trail with a strict call: tail_events takes "
        '{"source": "syslog", "window": {"from": "2026-03-01", "to": '
        '"2026-03-02"}, "limit": 500} - typed fields, quoted keys, no '
        "markdown, arguments closed.",
        'tail_events( source=syslog window={"from": 2026-03-01 "to": '
        '"2026-03-02"} limit=500 ) - keyword syntax instead of JSON, the from '
        "date is unquoted, and the comma between window fields is missing.",
    ),
]

# (valid_closed_think_cot_text, truncated_or_looping_think_text)
COT_REASONING_ANCHOR_PAIRS: list[tuple[str, str]] = [
    (
        "<think> The train leaves 14:05 and the trip takes 2 h 40 min: "
        "14:05 + 2:00 = 16:05, + 0:40 = 16:45. Wait, the express stops for 12 "
        "minutes at Marchfield, so redo: 16:45 + 0:12 = 16:57. Verify against "
        "the timetable row for RE 401: arrival is listed as 16:57, which "
        "matches my corrected sum. </think> The train arrives at 16:57.",
        "<think> The train leaves 14:05 and the trip takes 2 h 40 min, so "
        "arrival is 16:45, and then I should also account for the 12 minute "
        "stopover at Marchfield which shifts arrival to 16:57, and now let me "
        "double-check the timetable row for",
    ),
    (
        "<think> Factor x^2 - 5x + 6 as (x - 2)(x - 3), so the roots are 2 "
        "and 3. Substitute back to verify: 4 - 10 + 6 = 0 and 9 - 15 + 6 = 0, "
        "both check out. Hmm, the question asks for the sum of the roots, not "
        "the roots themselves - correcting course: 2 + 3 = 5, which agrees "
        "with Vieta's formula -b/a = 5. </think> The sum of the roots is 5.",
        "<think> The sum of the roots is 5. Let me verify: the sum of the "
        "roots is 5. Verifying again, the sum of the roots is 5. To be sure, "
        "I will verify once more that the sum of the roots is 5. Verifying "
        "once more, the sum of the roots is 5. The sum of the roots is",
    ),
    (
        "<think> The endpoint returns 500 only on weekends according to the "
        "dashboard. Check the deploy log: the rollout landed Friday 23:00. "
        "Hmm, Friday 23:00 UTC is already Saturday in the log's local time - "
        "my timezone assumption was wrong, redo with UTC timestamps: failures "
        "start Saturday 00:14 UTC, matching the rollout. Consistent. </think> "
        "The errors started with Saturday's rollout; roll it back.",
        "<think> The endpoint returns 500 only on weekends so I should look "
        "at the deploy log next and cross reference the rollout time with the",
    ),
    (
        "<think> Step 1: convert 45 mph to m/s: 45 * 0.44704 = 20.117 m/s. "
        "Step 2: braking distance is v^2 / (2a) = 20.117^2 / (2 * 6) = "
        "404.7 / 12 = 33.7 m. Sanity check: doubling speed quadruples "
        "distance, and 90 mph would give ~135 m - plausible. Wait, I used "
        "deceleration 6 m/s^2 but the problem says 5 - redo: 404.7 / 10 = "
        "40.5 m. Verify units: (m/s)^2 / (m/s^2) = m. Correct. </think> The "
        "braking distance is about 40.5 meters.",
        "<think> Convert 45 mph to m/s: 45 times 0.44704 is 20.117 m/s, and "
        "braking distance is v squared over two a, so with a of 6 that is "
        "33.7 m, but the problem might say 5, so let me recompute that",
    ),
    (
        "<think> The base rate is 1 in 1000 and the test is 99 percent "
        "accurate. First pass: of 100,000 people, 100 are sick and 99 test "
        "positive; 99,900 are healthy with a 1 percent false rate, giving "
        "999 false positives. Wait, I dropped true negatives from the count - "
        "correct: 999 positives total, 99 true, so P(sick | positive) = 99 / "
        "1098 = 0.09. Cross-check with Bayes: 0.001 * 0.99 / (0.001 * 0.99 + "
        "0.999 * 0.01) = 0.09. Matches. </think> About 9 percent; most "
        "positives are false.",
        "<think> The base rate is 1 in 1000 and the test is 99 percent "
        "accurate so most positives are true positives, which means the "
        "probability you are sick given a positive test is 99 percent, let me "
        "verify that with Bayes rule, Bayes rule says the posterior equals",
    ),
    (
        "<think> The SQL returns duplicate customer rows, so the JOIN on "
        "orders is multiplying rows. Hypothesis: one customer with three "
        "orders appears three times. Check the count: SELECT customer_id, "
        "COUNT(*) FROM orders GROUP BY customer_id - yes, customer 7 has 3 "
        "rows. Hmm, but I also see NULLs in customer_id, which my join would "
        "drop - correction: use LEFT JOIN and aggregate orders first in a "
        "CTE. Re-verify: per-customer totals now match the ledger exactly. "
        "</think> Aggregate orders in a CTE, then LEFT JOIN customers.",
        "<think> The SQL returns duplicate customer rows so the join is "
        "multiplying rows, I should check whether the join on orders is "
        "multiplying rows, the join on orders is multiplying rows so the fix",
    ),
    (
        "<think> The lockout counter resets every 15 minutes, so after 4 "
        "failed attempts at 10:00 the user waits until 10:15. Check the "
        "window logic: attempts at 09:47, 09:58, 10:01, 10:04. Hmm, the first "
        "two are outside the rolling window - my flat 15-minute read was "
        "wrong. Redo with the rolling window from 10:04: only 09:50 onward "
        "counts, so attempts 3 and 4 are in-window; one more attempt locks "
        "the account. Verify against the audit trail: lockout fired at 10:06 "
        "after the fifth attempt. Consistent. </think> One more attempt "
        "triggers the lockout.",
        "<think> The lockout counter resets every 15 minutes so after 4 "
        "failed attempts the user waits until 10:15, and the audit trail "
        "confirms the lockout fired after five attempts which means the",
    ),
    (
        "<think> Balance the combustion of ethanol: C2H5OH + O2 -> CO2 + "
        "H2O. Carbon: 2 CO2. Hydrogen: 3 H2O. Oxygen: right side has 4 + 3 = "
        "7 O, left has 1 from ethanol plus 2 per O2, so 3 O2. Verify by "
        "recount: left O = 1 + 6 = 7, right O = 7. Balanced. Wait - the "
        "question asks for moles of O2 per mole of ethanol, so the answer is "
        "the coefficient 3, not the atom count. Double-check the stoichiometry "
        "with an online-style recompute: 1:3 holds. </think> 3 moles of O2 "
        "per mole of ethanol.",
        "<think> Balance the combustion of ethanol so carbon gives 2 CO2 and "
        "hydrogen gives 3 H2O and then oxygen gives 7 atoms on the right and",
    ),
    (
        "<think> The JSON parse fails on the config file. Step 1: run it "
        "through a validator - it reports an error at line 12. Step 2: look "
        "at line 12: a trailing comma after the last array element. Hmm, "
        "line 12 is fine; the validator counts from zero, so the real "
        "offender is line 13 - my line indexing was off by one. Correct and "
        "re-validate: parse succeeds after removing the trailing comma. "
        "Verify round-trip: json.loads reads the file back identically. "
        "</think> Remove the trailing comma on the last array element.",
        "<think> The JSON parse fails on the config file so run it through "
        "a validator, the validator reports line 12, remove the trailing "
        "comma on line 12, and then re-run the validator and the validator",
    ),
    (
        "<think> The triangle has sides 3, 4, and x with a right angle "
        "between the legs. If 3 and 4 are the legs, x = sqrt(9 + 16) = 5. "
        "Check the converse: 3^2 + 4^2 = 25 = 5^2, so the triple is "
        "Pythagorean. Hold on - the problem says the hypotenuse is the "
        "longest side and asks for the area, not x - correcting: area = "
        "(1/2) * 3 * 4 = 6. Verify with Heron once x is known: s = 6, area "
        "= sqrt(6*3*2*1) = 6. Consistent. </think> The area is 6.",
        "<think> The triangle has sides 3 4 and x with a right angle "
        "between the legs so x equals sqrt of 9 plus 16 which is 5 and the "
        "converse check gives 25 equals 25 and now I will verify the triple",
    ),
]


@dataclass
class PhaseSpan:
    """Result of segmenting one text into deliberation and answer phases."""

    deliberation_text: str
    answer_text: str
    has_deliberation: bool
    is_balanced: bool
    open_count: int
    close_count: int
    malformed: bool

    @property
    def deliberation_span(self) -> str:
        """Alias for :attr:`deliberation_text`."""
        return self.deliberation_text

    @property
    def surface_answer_span(self) -> str:
        """Alias for :attr:`answer_text`."""
        return self.answer_text


@dataclass
class PhaseSegmenter:
    """Segments raw model output into deliberation and answer phases.

    A deliberation phase is delimited by ``deliberation_start`` ... ``deliberation_end``
    (e.g. ``<think>`` / ``</think>``). Use :meth:`for_format` to build a segmenter
    for a known template, or pass explicit ``deliberation_start_id`` /
    ``deliberation_end_id`` to enable token-level phase masking.
    """

    deliberation_start: str = "<think>"
    deliberation_end: str = "</think>"
    deliberation_start_id: int | None = None
    deliberation_end_id: int | None = None
    format_name: str = "think"

    @classmethod
    def for_format(cls, fmt: str = "think") -> PhaseSegmenter:
        """Build a segmenter for a known chat/reasoning template.

        Supported formats:

        - ``"think"``: ``<think>`` / ``</think>``
        - ``"reasoning"``: ``<reasoning>`` / ``</reasoning>``
        - ``"harmony"``: ``<|channel|>analysis<|message|>`` / ``<|end|>``
        """
        key = fmt.lower().strip()
        specs = {
            "think": ("<think>", "</think>"),
            "reasoning": ("<reasoning>", "</reasoning>"),
            "harmony": ("<|channel|>analysis<|message|>", "<|end|>"),
        }
        if key not in specs:
            supported = ", ".join(sorted(specs))
            raise ValueError(f"Unsupported phase format {fmt!r}; supported: {supported}")
        start, end = specs[key]
        return cls(deliberation_start=start, deliberation_end=end, format_name=key)

    def segment_text(self, text: str) -> PhaseSpan:
        """Split ``text`` into deliberation and answer phases.

        Counts open/close tag occurrences and pairs them with a depth scan:
        a close tag with no open tag is a stray close, and an open tag never
        closed is a truncation. ``deliberation_text`` joins the inner spans of
        paired (or truncated) opens; ``answer_text`` is everything outside
        tags. ``is_balanced`` requires equal counts with a fully closed,
        properly paired structure; ``malformed`` is its negation.
        """
        # A chat template or reasoning-budget ceiling can append an immediate
        # redundant close tag on its own blank line right after the real close
        # tag (e.g. ``</think>\n\n</think>``). When the text contains an open
        # tag, collapse blank-line-separated duplicate closes into one so the
        # redundant copy is not miscounted as a stray close. Duplicates
        # separated by a single newline or inline whitespace are genuine leaks
        # and stay flagged as malformed.
        first_open_idx = text.find(self.deliberation_start)
        if first_open_idx != -1:
            end_len = len(self.deliberation_end)
            first_close = text.find(self.deliberation_end, first_open_idx)
            second_close = (
                text.find(self.deliberation_end, first_close + end_len)
                if first_close != -1
                else -1
            )
            if (
                second_close != -1
                and text.find(self.deliberation_end, second_close + end_len) == -1
                and text.find(
                    self.deliberation_start, first_open_idx + len(self.deliberation_start)
                )
                == -1
            ):
                between = text[first_close + end_len : second_close]
                if "\n\n" in between:
                    text = text[:first_close] + text[first_close + end_len :]

        open_count = 0
        close_count = 0
        depth = 0
        structure_ok = True
        delib_spans: list[str] = []
        answer_parts: list[str] = []
        i = 0
        n = len(text)
        content_start: int | None = None

        while i < n:
            if depth == 0:
                open_idx = text.find(self.deliberation_start, i)
                close_idx = text.find(self.deliberation_end, i)
                if close_idx != -1 and (open_idx == -1 or close_idx < open_idx):
                    # Stray close tag with no matching open: mark malformed and
                    # treat the close tag itself as inert answer text.
                    structure_ok = False
                    close_count += 1
                    answer_parts.append(text[i:close_idx])
                    i = close_idx + len(self.deliberation_end)
                    continue
                if open_idx == -1:
                    break
                open_count += 1
                answer_parts.append(text[i:open_idx])
                depth += 1
                i = open_idx + len(self.deliberation_start)
                content_start = i
            else:
                close_idx = text.find(self.deliberation_end, i)
                if close_idx == -1:
                    # Unclosed open tag: everything to the end of the text is
                    # truncated deliberation content.
                    structure_ok = False
                    delib_spans.append(text[content_start:] if content_start is not None else "")
                    i = n
                    content_start = None
                    break
                open_idx = text.find(self.deliberation_start, i)
                if open_idx != -1 and open_idx < close_idx:
                    # Nested open before the close: malformed nesting.
                    structure_ok = False
                    open_count += 1
                    depth += 1
                    i = open_idx + len(self.deliberation_start)
                    continue
                close_count += 1
                delib_spans.append(
                    text[content_start:close_idx] if content_start is not None else ""
                )
                i = close_idx + len(self.deliberation_end)
                depth -= 1
                content_start = None

        if depth == 0 and content_start is None:
            answer_parts.append(text[i:])

        is_balanced = structure_ok and open_count == close_count and depth == 0
        deliberation_text = "\n".join(span for span in delib_spans if span != "")
        answer_text = "".join(answer_parts)
        return PhaseSpan(
            deliberation_text=deliberation_text,
            answer_text=answer_text,
            has_deliberation=open_count > 0,
            is_balanced=is_balanced,
            open_count=open_count,
            close_count=close_count,
            malformed=not is_balanced,
        )

    def get_deliberation_tag_token_ids(self, tokenizer: Any = None) -> list[int]:
        """Return token ids for the deliberation open/close tags.

        Explicit ``deliberation_start_id`` / ``deliberation_end_id`` take
        precedence; when neither is set and a tokenizer is provided, both tags
        are encoded with ``tokenizer.encode(tag, add_special_tokens=False)``.
        Returns ``[]`` when no ids can be resolved.
        """
        ids: list[int] = []
        if self.deliberation_start_id is not None:
            ids.append(int(self.deliberation_start_id))
        if self.deliberation_end_id is not None:
            ids.append(int(self.deliberation_end_id))
        if ids:
            return ids
        if tokenizer is None:
            return []
        for tag in (self.deliberation_start, self.deliberation_end):
            try:
                encoded = tokenizer.encode(tag, add_special_tokens=False)
            except TypeError:
                encoded = tokenizer.encode(tag)
            ids.extend(int(t) for t in encoded)
        return ids

    def _resolve_tag_id_sequences(self, tokenizer: Any = None) -> tuple[list[int], list[int]]:
        """Resolve (start_seq, end_seq) token-id subsequences for masking."""
        start_seq: list[int] = []
        end_seq: list[int] = []

        def _encode(tag: str) -> list[int]:
            try:
                encoded = tokenizer.encode(tag, add_special_tokens=False)
            except TypeError:
                encoded = tokenizer.encode(tag)
            return [int(t) for t in encoded]

        if self.deliberation_start_id is not None:
            start_seq = [int(self.deliberation_start_id)]
        elif tokenizer is not None:
            start_seq = _encode(self.deliberation_start)
        if self.deliberation_end_id is not None:
            end_seq = [int(self.deliberation_end_id)]
        elif tokenizer is not None:
            end_seq = _encode(self.deliberation_end)
        return start_seq, end_seq

    def compute_phase_masks(
        self,
        input_ids: torch.Tensor,
        tokenizer: Any = None,
        *,
        texts: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-token deliberation and answer masks of shape (B, T).

        Resolution order:

        1. Token-ID matching: explicit ``deliberation_start_id`` /
           ``deliberation_end_id``, else ``tokenizer.encode`` subsequences of
           the tags. Each deliberation span covers the start tag through the
           end tag inclusive (unclosed starts run to the end of the sequence).
        2. Character-ratio fallback: when ``texts`` is provided, tokens are
           split proportionally to the character counts of the segmented
           deliberation vs. answer text.
        3. Default fallback: ``deliberation_mask`` all False, ``answer_mask``
           all True.

        ``answer_mask`` is the complement of ``deliberation_mask``.
        """
        ids = input_ids if input_ids.dim() > 1 else input_ids.unsqueeze(0)
        squeeze_back = input_ids.dim() == 1
        batch_size, seq_len = int(ids.shape[0]), int(ids.shape[1])
        device = ids.device

        deliberation_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        answer_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
        if seq_len == 0:
            if squeeze_back:
                return deliberation_mask.squeeze(0), answer_mask.squeeze(0)
            return deliberation_mask, answer_mask

        start_seq, end_seq = self._resolve_tag_id_sequences(tokenizer)
        if start_seq and end_seq:
            start_t = torch.tensor(start_seq, dtype=torch.long, device=device)
            end_t = torch.tensor(end_seq, dtype=torch.long, device=device)
            start_len, end_len = len(start_seq), len(end_seq)
            for b in range(batch_size):
                row = ids[b]
                i = 0
                while i <= seq_len - start_len:
                    # Find the next start-tag subsequence at position >= i.
                    match_start = -1
                    for j in range(i, seq_len - start_len + 1):
                        if bool((row[j : j + start_len] == start_t).all()):
                            match_start = j
                            break
                    if match_start == -1:
                        break
                    # Find the close-tag subsequence after the open tag.
                    end_match = -1
                    for j in range(match_start + start_len, seq_len - end_len + 1):
                        if bool((row[j : j + end_len] == end_t).all()):
                            end_match = j
                            break
                    if end_match == -1:
                        deliberation_mask[b, match_start:] = True
                        answer_mask[b, match_start:] = False
                        break
                    span_end = end_match + end_len
                    deliberation_mask[b, match_start:span_end] = True
                    answer_mask[b, match_start:span_end] = False
                    i = span_end
        elif texts is not None:
            for b in range(min(batch_size, len(texts))):
                span = self.segment_text(texts[b])
                total_chars = len(span.deliberation_text) + len(span.answer_text)
                if not span.has_deliberation or total_chars <= 0:
                    continue
                ratio = len(span.deliberation_text) / total_chars
                k = int(round(ratio * seq_len))
                k = max(0, min(seq_len, k))
                if k > 0:
                    deliberation_mask[b, :k] = True
                    answer_mask[b, :k] = False

        if squeeze_back:
            return deliberation_mask.squeeze(0), answer_mask.squeeze(0)
        return deliberation_mask, answer_mask


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

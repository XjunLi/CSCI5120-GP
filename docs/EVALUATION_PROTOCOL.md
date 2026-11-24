# Evaluation protocol

## Objective and units

A candidate is a complete registered training recipe, identified by a content hash. A training seed controls initialization and training randomness. A search seed controls the order of proposals and controller updates. These are different experimental units.

For each configuration, the table stores three complete training measurements and selects by their mean quantized-roundtrip BPB. A configuration is feasible only if every required training replicate completes, every artifact satisfies the decimal 16,000,000-byte limit, and measured GPU training duration stays within the registered approximate target plus 60 seconds. The tolerance detects abnormal overruns; all measured seconds are charged, including the tolerance interval. The default task is **one H200 and approximately 600 seconds**, not an exact leaderboard time cutoff.

The GPU model is evaluated after the actual int8+zlib artifact has been written, read and dequantized. The precise `final_int8_zlib_roundtrip_exact` result is used. Uncompressed validation, training loss and a rounded intermediate log line cannot substitute for it.

## Dataset and evaluation boundary

The default download uses the first eight official sp1024 training shards. Freeze this choice before training. The original official training/validation separation is trusted. Within validation, search uses the first 80% of BOS-delimited documents and confirmation uses the remaining 20%; the split is fixed without consulting model scores.

The manifest records original file hashes, tokenizer hash, exact split boundaries and resulting file hashes. Search and confirmation document index ranges are verified disjoint. No claim of corpus-wide textual deduplication is made. The resulting BPB scores apply to these subsets and are not official full-validation leaderboard scores.

The CPU fixture consists of authored documents with disjoint document IDs and strings but shared grammatical templates. This is intentionally a plumbing test, not evidence of natural-language generalization.

The trusted oracle returns only candidate ID, observed search BPB, feasibility, status and measured costs. Controllers do not receive unqueried scores, confirmation scores, or the global optimum. The analyst may inspect the complete table only for post-search summaries. This interface is an information boundary for the implemented policies, not an OS sandbox for arbitrary generated programs.

## Methods and attribution

The primary methods are now random/evolution plus the memory/structure/frontier family: fully adaptive, fully frozen, and three single-component freezes. All arms retain the complete raw observation ledger and update statistics under their current rules. Memory freezes organization/retrieval adaptation; structure freezes executable graph editing; frontier freezes priority-policy/focus adaptation while still removing queried candidates.

The graph is interpreted by typed executable nodes and conditional edges. Structural edits are evidence-conditioned choices within a fixed operator set, with editing opportunities after queries 4 and 8. They are not independently validated promotions. Logs distinguish editing a state, actually traversing its nodes, changing the next proposal under a same-history graph comparison, and improving measured search quality.

Adding the memory-only, structure-only and frontier-only arms gives the full eight-cell factorial. A full-minus-drop-one contrast is conditional on the other switches being enabled. Average component effects and two/three-way interactions require the full factorial and complete paired seed blocks. All reported exploratory intervals are unadjusted; do not select favorable comparisons after seeing results.

The earlier methods below remain explicit optional baselines. The pretrained-LM weight experiment is a prepared second phase, to be run later when the user allocates compute.

| Condition | What changes | What its frozen control still receives |
| --- | --- | --- |
| Harness adaptive | An executable declarative program governing exploration, parent selection, memory length and mutation distance | Every observation and the same charged incumbent/challenger trial slots |
| Multiagent adaptive | Agent programs, lineage, population replacement and communication routes | Same observations, separate memories and peer collaboration |
| Weight adaptive | Feature-policy weights, or explicitly selected causal-LM LoRA adapters | Same initial policy, action constraints, observation history and sampler |

Harness trials interleave two incumbent and two challenger queries. Both adaptive and frozen arms spend the same scheduled trial slots; only the adaptive arm retains successful promotions. All candidate measurements made during a trial count toward the search budget and may become the final winner. Ties do not justify a promotion.

Multiagent attribution applies to the **joint final proposal**. It does not identify the separate causal contributions of proposer and critic. Descendants and routing changes provide evidence of state updates; they are not performance evidence by themselves.

NumPy updates operate on public design features. The optional LM path instead updates actual low-rank adapters in a causal model. Off-policy common warm-start examples do not receive policy-gradient updates. Each new run resets adapters, optimizer and sampling state; the base model stays frozen. Model-loading, proposal, forward-token and update costs are separately recorded. A local-policy result and a pretrained-LM result must be reported in separate cohorts.

## Budget comparisons

The default comparison uses 12 attempts, including four common initial candidates, on a 24-candidate table. Invalid outputs and duplicates consume attempts. Repeating a measured candidate does not incur a new training charge. Failed or infeasible new candidates consume their full measured charge.

The secondary comparison uses a common measured-training-cost ceiling equal to half of the full table's training cost, with at most 24 attempts. A candidate's public cost quote is known before querying its score. A proposal that cannot fit the remaining allowance is not evaluated; there is no free score inspection. This compares candidate-training cost, not the complete monetary cost of every possible LM service.

One configuration query aggregates three ten-minute training seeds: approximately 30 training GPU minutes. Twelve queries therefore correspond to approximately six training GPU hours if executed online, while the 24-configuration offline table costs approximately 12 training GPU hours and is reused for every replay.

Public cost quotes correspond to profiling/table timings. If a deployed online search lacks such quotes, it must use a predeclared reservation rule and account for realized overshoot; it must not use a result's unknown score to decide whether to count it.

Record the offline table-construction cost, simulated online candidate-training cost, actual replay/controller compute, and confirmation training separately. A single measured table may support many cheap replay seeds, but those replays do not create new training landscapes.

## Statistics

Run search seeds 0–29 with matched initialization. For each adaptive/frozen pair, report the mean paired difference in best search BPB, 95% bootstrap interval over paired search seeds, wins/ties/losses, and both/one/neither feasible counts. Negative adaptive-minus-frozen values are favorable. If runs fail to find a feasible winner, the score difference is explicitly conditional on both succeeding; also report the asymmetric failure outcomes.

These intervals describe search randomness **conditional on one fixed measured table**. The three underlying training seeds do not turn 24 configurations into 72 independent design points. Confirmation retrains selected configurations with fresh seeds; it assesses selected-model robustness, not meta-learning transfer to a new landscape.

Report quality throughout search as well as the final winner. Fixed-query anytime regret is the mean incumbent regret after attempts warm-start+1 through the allowance. The fixed-cost metric is the left-continuous incumbent-regret integral from the common warm-start cost to the common cap, divided by that cost interval. A result improves the incumbent only after its full quoted cost has been paid. Missing feasible incumbents are excluded and counted, never assigned an invented score. The table optimum is used only in post-search analysis.

The newer CPU harness study reuses an already inspected fixture table after the earlier confirmation was seen. It is exploratory and supplies no new untouched held-out evidence. The future H200 study must lock its complete intended method/cohort selections before inspecting its confirmation scores.

The full table optimum may be computed after all search runs to report simple regret. It is never input to a controller. If many methods repeatedly find the same best configuration, report saturation instead of interpreting zero SD or exact ties as certainty about general capability.

## Winner locking and confirmation

After all methods/seeds in a declared replay cohort finish, write an immutable winner lock binding the table ID, experiment ID, method, search seed, selected configuration and observation hash. Validate every lock before reading confirmation data for model evaluation.

Use training seeds 100, 101 and 102. Deduplicate selected configurations to avoid repeated training of the same confirmation job. Repeated search seeds choosing the same configuration share these measurements; they are not independent held-out experiments. No winner is reselected using confirmation scores.

## Integrity and interrupted runs

- Plans and results use content identities; modifying a plan invalidates its identity.
- All expected configuration×training-seed records must exist. Missing rows are never synthesized or predicted.
- Tables reject mixed hardware, evaluation protocol, training implementation, software/data execution identity and budgets.
- Resume checks the registered job and execution context. A changed implementation, dataset or environment requires a fresh output directory.
- Hardware-unavailable work is pending, not a zero-cost training observation. The batch launcher checks the requested H200 before launching jobs.
- Real failures retain their cost and raw logs. When optimizer-only failure time cannot be isolated, the recorded elapsed bound is explicitly marked.
- Completed measurements and trajectories are atomic files. A search process can rerun an unfinished deterministic local-policy seed; completed seeds are retained only under the same experiment identity. The optional LM policy also exposes optimizer/sampler checkpoints, but the batch CLI does not claim automatic mid-search checkpoint recovery.

## Optional proxy calibration

The default user task has only approximately 10-minute GPU jobs. Longer reference runs are disabled. If a later study explicitly adds them, preselect eight score-independent design anchors before short scores are known, then compare short/long means on exactly those configurations, same seeds, hardware, execution identity and evaluation text.

Report Spearman, Kendall tau-b, top-k recall and the short winner's long-budget regret on the measured anchors. Bootstrap configurations after averaging training seeds. Eight anchors do not support a claim about unmeasured long-run configurations. This optional analysis does not establish a speedup until actual measurements exist.

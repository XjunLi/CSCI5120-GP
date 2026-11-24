#!/usr/bin/env python3
"""Build a readable report from recorded measurements; never run training."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from functools import lru_cache
import fnmatch
import json
from pathlib import Path
import statistics
from archive_replays import load_records


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def number(value, digits=6):
    return "—" if value is None else f"{value:.{digits}f}"


@lru_cache(maxsize=4)
def cohort_records(path: Path):
    return load_records(path)


def mechanism(root: Path, cohort: str, stem: str) -> dict:
    records = cohort_records(root / cohort)
    runs = [value for name, value in records.items() if fnmatch.fnmatchcase(name, f"{stem}-adaptive_seed*.json")]
    frozen = {j["search_seed"]: j for name, j in records.items() if fnmatch.fnmatchcase(name, f"{stem}-frozen_seed*.json")}
    answer = {"runs": len(runs), "different_trajectories": 0,
              "different_candidate_sets": 0, "different_winners": 0}
    for run in runs:
        other = frozen[run["search_seed"]]
        a = [x.get("candidate_id") for x in run["trajectory"]]
        b = [x.get("candidate_id") for x in other["trajectory"]]
        answer["different_trajectories"] += a != b
        answer["different_candidate_sets"] += set(a) != set(b)
        answer["different_winners"] += ((run["winner"] or {}).get("candidate_id")
                                         != (other["winner"] or {}).get("candidate_id"))
    if stem == "harness":
        answer["complete_trials"] = sum(e["type"] == "harness_trial_completed"
                                        for r in runs for e in r["events"])
        answer["promotions"] = sum(r["promotions"] for r in runs)
        answer["runs_with_promotions"] = sum(r["promotions"] > 0 for r in runs)
        answer["incomplete_trials"] = sum(bool(r.get("incomplete_trial")) for r in runs)
    if stem == "multiagent":
        answer["descendants"] = sum(r["descendants"] for r in runs)
    if stem == "weight":
        answer["gradient_updates"] = sum(r.get("gradient_updates", 0) for r in runs)
        answer["runs_with_changed_weights"] = sum(r.get("initial_weight_sha256") != r.get("final_weight_sha256")
                                                   for r in runs)
        answer["policy_kind"] = sorted({r.get("policy_kind", "unknown") for r in runs})
    return answer


def render(root: Path, plot: bool = False):
    table = read(root / "table.json")
    cohorts = {name: read(root / name / "summary.json")
               for name in ("replay_attempts", "replay_cost")}
    cpu = table["profile"] == "cpu_smoke"
    methods = cohorts["replay_attempts"]["experiment"]["methods"]
    graph_study = any(method.startswith("memory-frontier-") for method in methods)
    title = ("CPU memory/structure/frontier experiment" if graph_study else "CPU protocol experiment") if cpu else "Measured H200 search experiment"
    text = [f"# {title}", "", "## Result", ""]
    if cpu:
        favorable = [p for c in cohorts.values() for p in c['paired_comparisons']
                     if p['paired_bootstrap_95ci'] and p['paired_bootstrap_95ci'][1] < 0]
        verdict = ("At least one paired interval favors adaptation; assess the declared comparisons and confirmation before drawing a broader conclusion. "
                   if favorable else "This cohort does not show a stable advantage from retaining adaptive updates over the matched frozen controls. ")
        introduction = ("This study evaluates memory/structure/frontier adaptation by replaying a measured table of tiny CPU models trained on an authored text fixture. "
                        if graph_study else "The complete pipeline ran on an authored text fixture with actual tiny-Transformer training, compressed-model reload, measured-table search and fresh-seed confirmation. ")
        text += [introduction
                 + verdict +
                 "Its scores are a protocol check; they are not FineWeb, H200 or pretrained-LM performance.", ""]
    else:
        text += ["The following results are computed from the registered measured table and immutable search outputs. "
                 "Read paired differences together with feasibility counts and the confirmation results.", ""]
    if (root / "measurement_source.json").exists():
        source = read(root / "measurement_source.json")
        text += [f"This study reuses the measured table from `{source['table_path']}`. "
                 f"It adds **{source['new_model_training_runs']} model-training runs**.", ""]
        if source.get('prior_confirmation_inspected'):
            text += ["Earlier confirmation results had already been inspected before this method was developed; "
                     "this replay is exploratory and does not provide a new untouched confirmation experiment. "
                     "The H200 task will use its own locked choices and confirmation data.", ""]
    text += ["## Coverage and cost", "", "| Item | Recorded value |", "| --- | --- |",
             f"| Configurations | {len(table['rows'])} |",
             f"| Training seeds | {table['training_seeds']} |",
             f"| Table-construction training jobs | {table['raw_run_count']} |",
             f"| Sum of optimizer/training time | {table['offline_training_seconds']:.6f} s |",
             f"| Sum of per-job end-to-end time | {table['offline_end_to_end_seconds']:.6f} s |",
             f"| Search replays | {sum(c['runs'] for c in cohorts.values())} |",
             f"| Fixed-query allowance | {cohorts['replay_attempts']['experiment']['budget']} attempts, "
             f"including {cohorts['replay_attempts']['experiment']['warm_start']} common initial configurations |",
             f"| Equal-cost allowance | {cohorts['replay_cost']['experiment']['cost_budget_seconds']:.6f} measured training seconds |",
             "", "Replay costs represent the candidate-training time that online search would have consumed. "
             "The models are trained once to construct the table; replaying a lookup is not another training job.", ""]
    if cpu:
        text += ["Each CPU training job completed 40 optimizer steps. The 10-second cap was a guard, "
                 "not a claim that each CPU job trained for 10 seconds. GPU jobs instead target approximately "
                 "600 seconds on one H200.", ""]
    text += ["## Best observed search BPB", "", "Lower is better. Each cell is the mean across paired search seeds "
             "on the same fixed table. All methods use the same initial configurations for a given seed.", "",
             "| Method | Fixed-query mean | Fixed-query optimum hits | Equal-cost mean | Equal-cost mean attempts |",
             "| --- | ---: | ---: | ---: | ---: |"]
    csv_rows = []
    for method in methods:
        a = cohorts["replay_attempts"]["methods"][method]
        b = cohorts["replay_cost"]["methods"][method]
        text.append(f"| `{method}` | {number(a['mean_best_bpb'])} | {a['optimum_hit_runs']}/{a['runs']} | "
                    f"{number(b['mean_best_bpb'])} | {b['mean_attempts']:.2f} |")
        csv_rows.append({"method": method, "query_mean_bpb": a["mean_best_bpb"],
                         "query_std_bpb": a["std_best_bpb"], "query_optimum_hits": a["optimum_hit_runs"],
                         "cost_mean_bpb": b["mean_best_bpb"], "cost_std_bpb": b["std_best_bpb"],
                         "cost_mean_attempts": b["mean_attempts"]})
    text += ["", f"The measured table optimum is **{number(cohorts['replay_attempts']['table_optimum_bpb'])} BPB**. "
             "This value is calculated by the analyst after search; controllers do not receive unqueried scores.", "",
             "## Adaptive versus frozen", "", "Delta is adaptive minus frozen; negative values favor adaptation. "
             "Intervals bootstrap paired search seeds. They describe search randomness conditional on this one measured table.", "",
             "| Budget | Pair | Mean delta BPB | 95% paired interval | Win / tie / loss | Both / adaptive only / control only / neither feasible |",
             "| --- | --- | ---: | --- | --- | --- |"]
    for name, summary in cohorts.items():
        for pair in summary["paired_comparisons"]:
            ci = pair["paired_bootstrap_95ci"]
            budget_label = f"{summary['experiment']['budget']} queries" if name == 'replay_attempts' else 'Equal cost'
            text.append(f"| {budget_label} | "
                        f"`{pair['adaptive']}` vs `{pair['frozen']}` | {number(pair['mean_adaptive_minus_frozen_bpb'])} | "
                        f"{('[' + number(ci[0]) + ', ' + number(ci[1]) + ']') if ci else '—'} | "
                        f"{pair['wins']} / {pair['ties']} / {pair['losses']} | {pair['both_feasible_pairs']} / "
                        f"{pair['adaptive_only_feasible_pairs']} / {pair['frozen_only_feasible_pairs']} / {pair['neither_feasible_pairs']} |")
    text += ["", "Exact ties in this finite table do not imply that the methods are equivalent on other tasks. "
             "When many runs reach the same table optimum, saturation limits sensitivity to improvements.", ""]
    if plot:
        draw_plot(root, cohorts, cpu)
        text += ["![Paired adaptive effects](paired_effects.svg)", ""]
    if any("mean_anytime_regret" in item for c in cohorts.values() for item in c["methods"].values()):
        text += ["## Quality throughout the search", "",
                 "The query metric averages incumbent regret after each adaptive attempt (5–12 in the default plan). "
                 "The cost metric integrates the incumbent available before the next result arrives, from the common "
                 "warm-start cost to the cost cap, then divides by that interval. Both use the post-search table optimum; "
                 "neither exposes it to a controller. Missing feasible incumbents are excluded and counted.", "",
                 "| Method | Query-prefix mean regret | Cost-normalized incumbent regret |",
                 "| --- | ---: | ---: |"]
        for method in methods:
            text.append(f"| `{method}` | {number(cohorts['replay_attempts']['methods'][method].get('mean_anytime_regret'))} | "
                        f"{number(cohorts['replay_cost']['methods'][method].get('mean_anytime_regret'))} |")
        text += ["", "| Budget | Pair | Anytime-regret delta | 95% paired interval | Complete pairs |",
                 "| --- | --- | ---: | --- | ---: |"]
        for name, summary in cohorts.items():
            for pair in summary['paired_comparisons']:
                item = pair.get('paired_anytime_regret')
                if item:
                    ci = item['paired_bootstrap_95ci']
                    text.append(f"| {name} | `{pair['adaptive']}` vs `{pair['frozen']}` | {number(item['mean_difference'])} | "
                                f"{('[' + number(ci[0]) + ', ' + number(ci[1]) + ']') if ci else '—'} | {item['both_feasible_pairs']} |")
        text += [""]
    if any(c.get('factorial_effects') for c in cohorts.values()):
        text += ["## Full-factorial component effects", "",
                 "The eight combinations switch memory organization, structure and frontier adaptation independently. "
                 "Main effects average over the other two switches. Two-way interactions are differences in differences "
                 "averaged over the remaining switch; the three-way term is the corresponding third difference. "
                 "Full-versus-drop-one contrasts above are conditional effects, not these averaged main effects. "
                 "Intervals below are descriptive, unadjusted for multiple comparisons, on this fixed table.", "",
                 "| Budget | Metric | Factors | Mean effect | 95% paired interval | Complete seed blocks |",
                 "| --- | --- | --- | ---: | --- | ---: |"]
        for name, summary in cohorts.items():
            if not summary.get('factorial_effects'):
                continue
            for metric, item in summary['factorial_effects']['metrics'].items():
                for effect in item['effects']:
                    ci = effect['paired_bootstrap_95ci']
                    text.append(f"| {name} | {metric} | {' × '.join(effect['factors'])} | {number(effect['mean_effect'])} | "
                                f"{('[' + number(ci[0]) + ', ' + number(ci[1]) + ']') if ci else '—'} | {item['complete_seed_blocks']} |")
        text += [""]
    graph_mechanisms = {}
    if graph_study:
        text += ["## Harness state changes and actual execution", "",
                 "The following counts describe the full-adaptation arm. Component edits modify retained rules or "
                 "the executable graph; routine observation append and fixed-rule statistic refresh are separate. "
                 "The graph comparison reuses the same history and RNG to compare the first proposal after an edit, "
                 "without querying either alternative's hidden score.", "",
                 "| Budget | Memory edits / used | Structure edits / used | Frontier edits / used | Graph comparisons / changed next candidate | Runs visiting coverage node |",
                 "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for name in cohorts:
            group = [r for r in cohort_records(root / name).values() if r['method'] == 'memory-frontier-adaptive']
            edits = {component: sum(r['edit_counts'][component] for r in group) for component in ('memory', 'structure', 'frontier')}
            comparisons = sum(r['structure_counterfactuals'] for r in group)
            changed = sum(r['structure_counterfactual_candidate_changes'] for r in group)
            used = {component: sum(len({row['controller']['component_versions'][component]
                                        for row in r['trajectory'] if row.get('controller', {}).get('component_versions', {}).get(component, 0) > 0})
                                   for r in group) for component in edits}
            visits = sum(any('coverage' in row.get('controller', {}).get('traversed_nodes', [])
                             for row in r['trajectory']) for r in group)
            graph_mechanisms[name] = {"runs": len(group), "edits": edits,
                                      "edited_versions_used_by_evaluated_proposals": used,
                                      "structure_counterfactuals": comparisons,
                                      "structure_counterfactual_candidate_changes": changed,
                                      "runs_with_evaluated_coverage_route": visits}
            text.append(f"| {name} | {edits['memory']} / {used['memory']} | {edits['structure']} / {used['structure']} | "
                        f"{edits['frontier']} / {used['frontier']} | {comparisons} / {changed} | {visits}/{len(group)} |")
        text += ["", "Every arm retains the raw evidence ledger. Memory changes grouping and retrieval; structure "
                 "changes interpreted nodes/edges/routes; frontier changes priority weights and focus over fixed public regions. "
                 "Hypothesis records and the observed Pareto archive are audit objects, not additional decision mechanisms. "
                 "State changes and next-proposal changes do not by themselves establish improved search performance.", ""]
    stems = [stem for stem in ("harness", "multiagent", "weight") if f"{stem}-adaptive" in methods]
    mechanisms = {name: {stem: mechanism(root, name, stem) for stem in stems}
                  for name in cohorts}
    for name, item in graph_mechanisms.items():
        mechanisms[name]['memory_frontier'] = item
    if stems:
        text += ["## Evidence that adaptation actually executed", "",
             "| Budget | Harness completed trials / promotions | Multiagent descendants | Weight gradient updates |",
             "| --- | ---: | ---: | ---: |"]
    for name, item in mechanisms.items():
        if stems:
            h = item.get('harness', {})
            text.append(f"| {name} | {h.get('complete_trials', '—')} / {h.get('promotions', '—')} | "
                        f"{item.get('multiagent', {}).get('descendants', '—')} | {item.get('weight', {}).get('gradient_updates', '—')} |")
    if stems:
        text += ["", "| Budget | Pair | Different ordered trajectories | Different queried sets |",
             "| --- | --- | ---: | ---: |"]
    for name, item in mechanisms.items():
        for stem, value in item.items():
            if stem == 'memory_frontier':
                continue
            text.append(f"| {name} | {stem} | {value['different_trajectories']}/{value['runs']} | "
                        f"{value['different_candidate_sets']}/{value['runs']} |")
    text += ["", "Changed programs, descendants and nonzero weight updates establish execution of the mechanism. "
             "They do not establish a performance benefit. In the CPU cohort the weight policy is a 19-parameter "
             "NumPy policy over public design features. The optional causal-LM LoRA path is a separate implementation "
             "and is not the source of these replay results.", ""] if cpu and 'weight' in stems else [""]
    confirmation_path = root / "confirmation" / "summary.json"
    if confirmation_path.exists():
        confirmation = read(confirmation_path)
        by_candidate = confirmation["by_candidate"]
        rows = {r["candidate_id"]: r for r in table["rows"]}
        jobs = sum(len(r["training_seeds"]) for r in by_candidate.values())
        text += ["## Locked-winner confirmation", "",
                 f"The {len(confirmation['locked_selections'])} locked selections across both replay cohorts reduce to "
                 f"**{len(by_candidate)} distinct configurations and {jobs} fresh-seed training jobs**. "
                 "The confirmation text is separate from search text. Winners were locked before evaluation; "
                 "no winner was reselected from confirmation scores.", "",
                 f"Confirmation training cost: {confirmation['training_seconds']:.6f} s; "
                 f"summed end-to-end cost: {confirmation['end_to_end_seconds']:.6f} s.", "",
                 "| Candidate | Depth / width / tied / LR scale | Search BPB, seeds 0–2 | Confirmation BPB, seeds 100–102 | Confirmation SD |",
                 "| --- | --- | ---: | ---: | ---: |"]
        for candidate, item in sorted(by_candidate.items()):
            row = rows[candidate]; cfg = row["config"]
            text.append(f"| `{candidate}` | {cfg['num_layers']} / {cfg['model_dim']} / {cfg['tied_embeddings']} / {cfg['lr_scale']} | "
                        f"{number(row['val_bpb'])} | {number(item['mean_quantized_bpb'])} | {number(item['std_quantized_bpb'])} |")
        lock_names = {read(root / name / "locked_winners.json")["lock_id"]: name for name in cohorts}
        values = defaultdict(list)
        no_winner = 0
        infeasible_confirmation = 0
        for selection in confirmation["locked_selections"]:
            item = selection.get("confirmation")
            if item is None:
                no_winner += 1
                continue
            score = item["mean_quantized_bpb"]
            if not item.get("all_feasible"):
                infeasible_confirmation += 1
            if score is not None:
                values[(lock_names[selection["lock_id"]], selection["method"])].append(score)
        text += ["", f"Selections without a feasible search winner: {no_winner}. "
                 f"Selections whose confirmation cohort is not fully feasible: {infeasible_confirmation}. "
                 "Score means below are conditional on available confirmation scores."]
        text += ["", "| Method | Confirmation mean of fixed-query selections | Confirmation mean of equal-cost selections |",
                 "| --- | ---: | ---: |"]
        for method in methods:
            text.append(f"| `{method}` | {number(statistics.mean(values[('replay_attempts', method)])) if values[('replay_attempts', method)] else '—'} | "
                        f"{number(statistics.mean(values[('replay_cost', method)])) if values[('replay_cost', method)] else '—'} |")
        text += ["", "These method means weight configurations by how often search selected them. "
                 "Repeated selections share the same three confirmation measurements; they are not independent "
                 "held-out training repeats. Search-to-confirmation score changes combine a different text subset "
                 "with new initialization seeds, so they do not isolate either source of variation.", ""]
    else:
        text += ["## Confirmation", "", "Confirmation has not been recorded in this result directory.", ""]
    text += ["## Provenance and reproducibility", "",
             f"- Table identity: `{table['table_id']}`.",
             f"- Manifest identity: `{table['manifest_id']}`.",
             f"- Execution fingerprint: `{table['execution_fingerprint']}`.",
             f"- Backend: `{table['backend']}`; hardware class: `{table['device_class']}`.",
             "- [Measured table](table.json), [fixed-query summary](replay_attempts/summary.json), "
             "[equal-cost summary](replay_cost/summary.json).",
             "- Search trajectories and winner locks are retained under this directory. Training records and artifacts "
             "are in the originating measurement cohort when a reused table is declared above.",
             "", "Use the repository runbook to create a fresh result directory when code, data or environment changes. "
             "The committed cohorts have matching source snapshots under `source_snapshot/`. "
             "Later search changes intentionally require a new replay directory and do not change recorded results.", "",
             "Regenerate this report with:", "", "```bash",
             f"python scripts/golf/report.py --root {root.relative_to(Path.cwd()) if root.is_absolute() and root.is_relative_to(Path.cwd()) else root}" + (" --plot" if plot else ""), "```", ""]
    (root / "REPORT.md").write_text("\n".join(text))
    (root / "mechanism_summary.json").write_text(json.dumps(mechanisms, indent=2, sort_keys=True) + "\n")
    with (root / "method_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys()); writer.writeheader(); writer.writerows(csv_rows)
    print(root / "REPORT.md")


def draw_plot(root, cohorts, cpu):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = max(len(c['paired_comparisons']) for c in cohorts.values())
    fig, axes = plt.subplots(1, 2, figsize=(11, max(3.8, 1.5 + rows * .6)), sharey=True, layout="constrained")
    for axis, (name, summary) in zip(axes, cohorts.items()):
        for y, pair in enumerate(summary["paired_comparisons"]):
            mean = pair["mean_adaptive_minus_frozen_bpb"]; ci = pair["paired_bootstrap_95ci"]
            if mean is not None and ci:
                axis.plot(ci, [y, y], color="#285d85", lw=2.5)
                axis.plot(mean, y, "o", color="#285d85", markersize=7)
        axis.axvline(0, color="#9099a3", linestyle="--", lw=1)
        axis.set_title("Fixed query budget" if name == "replay_attempts" else "Equal training cost", fontsize=12)
        axis.set_xlabel(("Full − control BPB" if any(p['adaptive'].startswith('memory-frontier-') for p in summary['paired_comparisons'])
                         else "Adaptive − frozen BPB") + "  (← better)", fontsize=10)
        labels = []
        for pair in summary['paired_comparisons']:
            if pair['adaptive'].startswith('memory-frontier-'):
                labels.append('Full vs ' + pair['frozen'].removeprefix('memory-frontier-'))
            else:
                labels.append(pair['adaptive'].removesuffix('-adaptive').capitalize())
        axis.set_yticks(range(len(labels)), labels)
        extent = max([0.055] + [abs(v) * 1.3 for c in cohorts.values()
                               for p in c['paired_comparisons']
                               for v in (p['paired_bootstrap_95ci'] or [])])
        axis.set_xlim(-extent, extent); axis.set_ylim(rows - .4, -0.65)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.grid(axis="x", color="#e5e9ed", lw=.6)
        axis.tick_params(axis="y", length=0)
    fig.suptitle(("CPU fixture: " if cpu else "") + "paired effects and 95% search-seed intervals", fontsize=13)
    fig.savefig(root / "paired_effects.svg")
    fig.savefig(root / "paired_effects.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plot", action="store_true", help="also draw a figure (requires matplotlib)")
    args = parser.parse_args()
    render(args.root, args.plot)

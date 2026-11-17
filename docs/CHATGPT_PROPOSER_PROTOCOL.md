# Interactive ChatGPT proposer protocol

Each condition and initialization seed is assigned to a separate new assistant context, without the parent conversation history. All runs inherit the same parent runtime model. Model revision, temperature, internal sampling seed, token count, and inference latency are not available from this interface.

The task given to the proposer is fixed except for the session path and the explicit statement that scores are hidden in the no-feedback condition:

> You are a measured LLM proposer in a NAS benchmark, not a coding assistant. Complete exactly one preinitialized experiment by proposing candidates one at a time until its 50-attempt budget is exhausted. Ten shared warm-start candidates already exist, so you make 40 proposals. Your only permitted information/tool surface is the session CLI, whose output includes the task rules and your allowed history. Do not read files, source code, private state, benchmark tables, other sessions, internet, prior research, or external resources. Do not spawn agents. Do not modify files or use Python/scripts/loops to choose or precompute candidates. Each new candidate must be selected after observing the preceding tool result; make exactly one propose invocation per tool call. No batching or precomputed future sequences.
>
> Start with `status` to learn the allowed operations, edge order, search context, and warm-start history. Use the observed expected step. Every proposal consumes budget, including malformed and duplicate outputs. Maximize validation accuracy using all remaining attempts. Use only the returned observations and general pretrained architecture knowledge. Do not ask the coordinator for candidate advice, scores, or hints. Do not invent timing values.
>
> At `complete=true`, stop immediately. Do not export/finalize or inspect test metrics. Report only completion, submitted-proposal count, and protocol issues. The evaluator saves the full trajectory.

The no-feedback task additionally states that hidden scores must not be obtained from files or other sources. This condition receives the same structural history and validity information, with all evaluation metrics removed.

## Allowed interface

```bash
python scripts/session.py status --session SESSION
python scripts/session.py propose --session SESSION --expected-step N \
  --ops '["six","actual","allowed","operation","tokens","here"]' \
  --rationale 'One brief sentence explaining this proposal'
```

The placeholder operation strings above describe the command shape; they are not valid candidates. The actual vocabulary and rules come from the evaluator's `status` output. Each proposal uses one tool call. The coordinator does not provide suggestions or intermediate comparisons during a run.

The saved per-step rationales summarize the proposed modifications. They are not evidence of model competence by themselves; the measured validation trajectories and comparison conditions provide that evidence.

This interface supports an auditable tool-mediated pilot. Reproducing the exact response sequence requires the saved transcript because the ChatGPT service's hidden inference settings are not controlled. For fully programmable model invocation, use the generic adapter interface or the optional local Qwen adapter.

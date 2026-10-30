# SR-agent

系统分层、模块职责与接口约定见 [系统结构与模块接口](docs/SYSTEM_ARCHITECTURE.md)。

SR-agent is a multi-agent hyperparameter-optimization framework for speaker-recognition experiments. The project includes LangGraph orchestration, data-processing handoff, HPO strategy selection, trial scheduling, structured experiment records, and pluggable model/runner adapters.

## Real Demo

The submission demo is in `demo/` and uses the real `speechbrain` runner. It is not a fake execution path.

Preview the command:

```bash
python demo/run_ecapa_hpo.py --data-folder /tmp/voxceleb1 --dry-run
```

Run the full smoke demo:

```bash
bash demo/run_ecapa_hpo.sh --data-folder /tmp/voxceleb1
```

Read the full demo guide:

```text
demo/README.md
```

## Install

For real SpeechBrain training, install a CUDA-matched PyTorch stack first, then install SpeechBrain support.

CUDA 12.8 example:

```bash
pip install torch==2.9.1 torchaudio==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128 --timeout 1000 --retries 20
pip install -e .[speech]
python scripts/tools/check_remote_environment.py --require-cuda
```

CUDA 12.1 fallback:

```bash
pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 --timeout 1000 --retries 20
pip install -e .[speech]
python scripts/tools/check_remote_environment.py --require-cuda
```

If the image already provides a working `torch` and `torchaudio`, keep them and run only:

```bash
pip install -e .[speech]
python scripts/tools/check_remote_environment.py --require-cuda
```

The base package intentionally does not install `torch`, `torchaudio`, or `speechbrain`; this avoids CUDA version conflicts on rented GPU servers.

## Script Layout

- `demo/`: real runnable demo framework for submission.
- `scripts/experiments/`: five-group HPO benchmark and fixed expert-config baseline.
- `scripts/evaluation/`: evaluate a specific trained checkpoint.
- `scripts/admin/`: archive and clear experiment records/artifacts.
- `scripts/tools/`: environment and runtime inspection utilities.

Common commands:

```bash
python scripts/tools/check_remote_environment.py --require-cuda
python scripts/evaluation/evaluate_checkpoint.py --help
python scripts/experiments/run_hpo_benchmark.py --dry-run
python scripts/experiments/run_hpo_benchmark.py --help
python scripts/experiments/run_expert_baseline.py --dry-run
bash scripts/admin/archive_experiments.sh --model-family ecapa_tdnn
bash scripts/admin/clear_experiments.sh --model-family ecapa_tdnn
```

The [expert baseline guide](scripts/experiments/EXPERT_BASELINE.md) describes
training a supplied YAML without HPO or an LLM. It preserves the recipe's epochs
and hyperparameters; verification and held-out test results never select parameters.

## Agent-controlled HPO

Enable the advisor to inspect existing HPO evidence through bounded, read-only
tools and select, configure, or switch the active candidate sampler between
batches. Each batch uses exactly one sampler: `random_search`, `grid_search`,
`adaptive_search`, `tpe`, or `agent_proposal`. The last option lets the agent
submit evidence-backed parameter sets as a standalone search method; it is not
mixed with TPE candidates.

```bash
python main.py \
  --enable-llm-advisor \
  --controller-mode llm \
  --strategy tpe \
  --sampler-config-json '{"n_startup_trials":3,"multivariate":false}' \
  --candidate-batch-size 3 \
  --strategy-review-interval-trials 3 \
  --verification-config configs/verification_ecapa.yaml \
  --validation-pairs datasets/protocols/hpo_validation.txt
```

Both validation arguments are mandatory for a new HPO run. The validation-pair
speakers are excluded from training and every Trial is evaluated only on that
split. A broader `--training-exclusion-pairs` file may be supplied when it must
also exclude a separately held-out test population; it must contain all
validation speakers. Held-out test pairs are never accepted by the HPO command.
EER is recorded as a ratio in `[0,1]`, while minDCF is recorded as `×100`.

Use `--output-dir results/RUN_NAME` to keep a complete system run together:
`manage/`, `dp/`, `hpo/`, and shared `memory/` live under that directory.
An explicit `--hpo-experiments-dir` overrides only the HPO location, including
when resuming a previous Study. The root `result.json` reports orchestration
status; a failed workflow exits with code 1. Dataset-version files retain the
complete file hashes; data-processing records reference them rather than
embedding the same inventory twice.

The training-exclusion file is a denylist for the full protocol. Its speakers
need not all appear in a smaller training-data view; every speaker that does
appear and is on that list is still excluded.

`--controller-mode` is explicit and persistent: `fixed` never changes the
requested search policy, `rule` uses only deterministic feedback rules, and
`llm` is the only mode that invokes or applies LLM proposals. The five-group
benchmark assigns `fixed` to every baseline and `llm` only to the system group.

Candidate batches are executed before the next runtime review, so completed
Trials can change the sampler, sampler configuration, safe search space, and
hypotheses within the same Study. Agent-proposed candidates are deterministically
validated before use. Trial records retain their search phase, sampler,
`proposal_id`, `hypothesis_id`, governing `decision_id`, and provenance for
audit. Runtime LLM changes pass a deterministic evidence gate (same-fidelity
sample count, finite confidence, parameter observations, and TPE startup-history
retention). A bounded, high-confidence `agent_proposal` probe is the only
low-sample exception; repeated failures have a separately recorded safety
exception.

The Campaign freezes its final confirmation budget, objective, metric units,
validation hashes, dataset/model identity, configuration hash, and runner before
Studies are compared. Runtime reviews cannot alter allocation or confirmation
budgets. Once the current Study has no generation capacity (including an
exhausted finite grid), advice is stored as `next_study_proposal` and is never
applied retroactively. In `rule` and `llm` modes, that stored proposal is passed
through deterministic plan validation and applied as the next Study's base plan;
`fixed` mode deliberately ignores it. A fresh LLM planning failure therefore
does not discard already-approved next-Study advice. Applied reviews record
their effective Trial index, affected Trial IDs, and realized metric/time
outcome. They also report a decision-before, same-Study, same-rung and exact-
budget observational reference when one exists. This estimate is explicitly
non-causal; causal system claims still require frozen multi-seed control runs.

The LLM read-only history tool returns only successful experiments with a finite
best objective and an exact full confirmation signature match (final budget,
objective/units, dataset/version, model/runner, config, validation and exclusion
hashes). A merely similar metric or dataset name is not considered comparable.

`max_duration_seconds` is an enforced per-training-run deadline. SpeechBrain
training runs in a child process; a run that exceeds the deadline is terminated
and recorded with `terminated_by_budget`, timeout, exit code, and failure status.

The HPO scheduler stops the Study immediately for shared program, dependency,
data-file, or device-configuration failures. Two consecutive distinct Trials
with the same unknown execution error in the same stage also open the failure
guard. Repeated identical connection/worker failures are also bounded: two
distinct Trials or three consecutive attempts (including retries) open the guard.
Unknown non-OOM CUDA failures use this bounded path too, rather than treating
every CUDA error as candidate-specific memory pressure.
PID/worker/rank IDs are normalized so process restarts do not evade this check.
OOM, timeouts, invalid metrics, and candidate-validation failures remain
Trial-level failures. An open guard prevents further training, promotion, and
LLM reviews; existing checkpoints and queued Trials are retained. Inspect
`hpo_study/study.json` under `scheduler_state.failure_guard`, or the experiment
record under `extensions.optimization.study.failure_guard`, for the error,
stage, fingerprint, and affected Trial IDs.
Scheduler-node exceptions also open the guard immediately, with the node name
and a bounded traceback. Single-process training/verification return import and
runtime exception causes instead of only process exit codes. DDP workers retain
per-rank failure JSON alongside their per-launch result; the parent prefers a
shared root cause over torchrun wrapper errors, with a bounded launcher-log tail
as fallback. Disk-full, dependency/ABI, fork/CUDA, interface-signature and missing-
checkpoint contract errors are shared failures, not reasons to train a new candidate.
The enclosing Campaign is recorded as `failed`, with its source Study identified
in `extensions.optimization.campaign_failure_guard`. Resuming an older successful
Study cannot bypass the source Study's live open guard.
The five-group benchmark also stops subsequent groups/seeds and retains the
open guard in the failed run's `result.json`, including across benchmark resume.

After repairing the underlying cause, explicitly unblock the remaining work:

```bash
python main.py --resume-experiment-id EXPERIMENT_ID --reset-failure-guard
```

Retain the original controller/LLM settings when resuming. Resetting the guard
is recorded and does not automatically rerun failed Trials. An ordinary resume
keeps the guard open. The five-group benchmark still enforces its frozen source
hashes; changed code requires a new benchmark output directory.

One Trial can optionally use multiple visible GPUs through SpeechBrain DDP. Keep
the HPO controller as one normal Python process; it launches `torchrun` only for
the selected Trial, and only rank zero returns the result to the Study:

```bash
python main.py \
  --strategy tpe \
  --ddp-devices 0,1 \
  --distributed-backend nccl \
  --ddp-batch-size-semantics global \
  --verification-config configs/verification_ecapa.yaml \
  --validation-pairs datasets/protocols/hpo_validation.txt
```

`--ddp-devices auto` uses all visible CUDA devices when at least two are
available and otherwise falls back to ordinary single-device training. With
`global` semantics the searched/configured `batch_size` remains the effective
global batch and must be divisible by the GPU count; each worker receives
`batch_size / world_size`. `per_device` preserves the batch on every GPU and
therefore increases the effective global batch. Do not start `main.py` itself
with `torchrun`, because that would duplicate the HPO scheduler and Trial state.
DDP configuration is frozen into the Study confirmation signature and restored
on resume, so single- and multi-GPU results are not silently treated as the same
training protocol.

Resume an interrupted Study from its persisted experiment record:

```bash
python main.py --resume-experiment-id YYYYMMDD_HHMMSS_0
```

After repairing an evaluation infrastructure error, explicitly retry only the
failed evaluations whose training succeeded and whose local checkpoints still
exist. For an isolated benchmark, point at that run's `hpo/` directory:

```bash
python main.py \
  --hpo-experiments-dir experiment_results/five_group_vox1_3gpu/runs/tpe_seed_11/hpo \
  --resume-experiment-id 20260928_143407_0 \
  --controller-mode fixed \
  --reset-failure-guard \
  --retry-failed-evaluations
```

This opt-in recovery preserves Trial IDs, checkpoints, training-run counts and
the frozen validation protocol. Evaluation still consumes any remaining
Campaign resource allowance. It does not repeat training or reset exhausted
resource limits. Ordinary resume leaves terminal failed Trials unchanged.
For an LLM-controlled run, retain `--enable-llm-advisor --controller-mode llm`.
Recovered validation scores cannot reconstruct the online decisions missed
during the original failed search.

The SpeechBrain runner evaluates cosine scores. Its x-vector default is
`configs/verification_cosine_xvector.yaml`; explicitly passing a PLDA config is
rejected rather than silently returning cosine results.

For an LLM-controlled Study, also pass
`--enable-llm-advisor --controller-mode llm`. Resume rejects a controller-mode
mismatch instead of silently changing the experiment policy. An explicitly
supplied DDP layout is also rejected when its world size, backend, or batch
semantics conflicts with the persisted Study.

Completed and already-suggested Trials are reused. A Trial left in `running`
state is reopened with the same Trial ID and output directory, so the scheduler
does not generate a duplicate candidate. SpeechBrain can then recover from a
checkpoint already present in that Trial directory; if no trainer checkpoint
exists, only that interrupted Trial is rerun while completed Trials are kept.

Resume also rebuilds Study summaries from durable Trial results, reconciles
partially written promotions, and performs any pending batch review before
generating new candidates. Committed reviews are not replayed. Training Trials
use the experiment's frozen `config.yaml`, including legacy records whose
snapshot reference is missing. If that snapshot is lost, restore it before
resuming; the mutable source configuration is not used as a fallback.

Optimization safety boundaries:

- Raw verification scores must be finite, and score files must contain valid
  positive and negative pairs. A declared score file that is missing, malformed,
  or cannot be scored fails evaluation; the tool does not fall back to a
  potentially misleading Runner metric. Finite, genuinely perfect scores remain valid.
- Cross-Study observations enter samplers only when actual epochs, data fraction,
  duration budget, and recorded experiment context match. Stage names alone do
  not establish compatibility. Incompatible or unverified legacy history is
  retained for reference but excluded from both sampler observations and candidate
  deduplication, allowing those parameters to be evaluated at the new fidelity.
  History merge keys include budget and context, not just parameter values.
- Successive-halving retries remain inside `max_training_runs` and may use only
  slots beyond the remaining initial/promotion allocation. A skipped retry records
  its budget reason. Completion requires a valid result at the plan's highest
  reachable confirmation rung (an explicit zero promotion limit ends the plan).
  Otherwise the Study is `failed`, with confirmation progress and a clear reason;
  screening-only results are not silently reported as a fully confirmed optimum.

Evaluate one ECAPA-TDNN training checkpoint on an explicit verification list:

```bash
python scripts/evaluation/evaluate_checkpoint.py \
  --checkpoint /path/to/output/save/CKPT+2026-07-20+11-26-17+00 \
  --data-folder /hy-tmp/voxceleb1 \
  --verification-file /hy-tmp/lists/veri_test2.txt \
  --device cuda \
  --batch-size 8 \
  --output-dir results/checkpoint_evaluation/ecapa_trial_5c4a59ec3d
```

The command writes per-pair cosine scores to `scores.txt` and the EER/minDCF
summary to `evaluation_result.json`. The verification YAML must match the
checkpoint architecture; use `--config` when evaluating another model.

## Campaign GPU-hour and wall-time budgets

HPO accepts `max_gpu_hours` and `max_wall_seconds` in `AgentTaskRequest.budget`.
Both are optional finite positive numbers. CLI users can append, for example:

```bash
--max-gpu-hours 72 --max-wall-seconds 86400
```

For `scripts/experiments/run_hpo_benchmark.py`, add the same keys to its JSON
configuration. These limits apply **separately to each group/seed Campaign**, not
to the sum of all five groups. Without these keys, existing run-count budgeting
is unchanged. All configured ceilings apply together; the first exhausted one
stops scheduling. Declared `resource_profile` limits (`gpu_hours`, `wall_seconds`,
their `*_remaining` aliases, or the canonical `max_*` keys) are enforced too;
when multiple limits are supplied, the tighter allowance wins.

- GPU hours mean stage elapsed time times allocated GPU count. Three-card DDP
  training for one hour costs three GPU hours; one-hour verification costs one.
  CPU stages cost zero GPU hours. This is not a GPU-utilization/energy measure.
  Training, verification, failed attempts and retries all enter the ledger.
- Wall time starts when the Campaign is created, before LLM planning. It includes
  planning/review, training, verification and retry delays. Its original deadline
  survives resume, **including time spent paused/offline**. Benchmark asset
  preparation and the separately requested held-out final test are outside this
  optimization Campaign allowance.
- Before each stage, remaining wall time, remaining GPU allowance divided by
  GPU count, and the Trial's training duration cap determine the effective
  timeout. SpeechBrain single-card/DDP training and isolated verification stop
  their worker on timeout. LLM requests also use the remaining wall allowance,
  without SDK retries extending that deadline. Worker termination and record
  cleanup may add a small overrun; that time is charged, not hidden. Custom
  runners must honor the `_hpo_max_duration_seconds` override themselves.
- Scheduling, promotion, review and retries stop on resource exhaustion. An
  interrupted Study is `cancelled`, its result is not reported as a successful
  full search, and existing checkpoints/results are preserved. Resource
  exhaustion is distinct from the system-failure circuit breaker.
- The authoritative journal is
  `<experiments_dir>/campaign_resources/<campaign_id>.json`, shared across all
  Studies of that Campaign. It records each attempt/stage, GPU count, elapsed
  time and recovery estimates. `extensions.optimization.resource_budget` and
  Study `scheduler_state.resource_budget` expose consumed/remaining allowance
  and the stop reason. Benchmark `results.json`/`results.csv` include GPU costs.
- Resume reloads the ledger; limits cannot be increased or reset. Two controllers
  cannot spend one ledger concurrently. A hard crash with an unsettled stage
  conservatively charges its reserved allowance (or longer observed elapsed
  time) and marks `cost_may_be_incomplete`, rather than silently refunding it.
  Resume only after the previous training/evaluation workers have stopped.
  Old Campaigns without a ledger cannot acquire new resource limits on resume,
  because their historical costs are unverified; start a new Campaign instead.

Benchmark source hashes are frozen: after updating code, use a **new output
directory** rather than resuming an experiment created with older sources.

## VoxCeleb data roles and internal development split

`configs/train_ecapa_tdnn_vox1_3gpu.yaml` uses `split_unit: session`:
train/dev share speaker identities but never recording sessions. Each speaker
with at least two sessions contributes whole sessions to dev; single-session
speakers stay in train. The 90/10 target applies to **session counts per speaker**,
so utterance counts need not be exactly 90/10. Both endpoints of the training
exclusion protocol remain excluded from train/dev. Preparation randomness uses
`data_prep_seed`, independently of model seeds. `validation_crop: center` fixes
the internal classification dev crop; training keeps random chunks. Verification
continues to use its configured scoring protocol and full utterances.

These options also work when invoking the training recipe directly. Configs
without the new keys keep utterance splitting and the previous crop behavior.
The split unit enters the preparation cache key. Start a new benchmark output
directory to compare the new protocol; do not mix it with historical results.
Preparation metadata also records the exclusion protocol hash, random-segment
mode and amplitude threshold; changing them rebuilds the local annotation cache.
Existing caches without those fields are regenerated on their next preparation.

For `speaker_verification`, the data quality policy interprets `train` as training,
`dev`/`valid` as internal classification development, `validation` as HPO
validation, and `enrol`/`test` as endpoints of the final verification population.
Internal dev may share identities with train but not files; evaluation populations
must remain isolated from training and from each other. Endpoint lists within
one evaluation population may overlap. Override `quality_policy.split_roles`
for other manifest names or protocols. Generic tasks retain strict disjointness.

The data-processing workflow binds `runtime_options.training_exclusion_pairs`
to a protocol hash and protected speaker inventory. If those identities occur
in the input corpus, audit operations are allowed, while filtering/rewriting
is rejected until a training-only view is supplied. This protects held-out
audio even when a processing plan is constructed directly after scope binding.
The workflow does not yet build that view automatically. Bind the full exclusion
protocol, covering both HPO validation and final test speakers. Duration warnings
alone no longer schedule automatic audio removal. VoxCeleb profiling reads speech
under `wav/` separately from noise/RIR assets and identifies `idNNNN` speaker
directories rather than video directories. Header/signal inspection is still
bounded by the quality policy; a passing sampled audit is not a full-corpus scan.

## Add a New Model

1. Implement a model adapter under `agent/models/`; use `demo/templates/my_model_adapter.py` as the starting point.
2. Register the adapter in `agent/models/__init__.py`.
3. Add train/evaluation YAML files under `configs/` or `recipes/voxceleb/hparams/`.
4. Copy `demo/config/ecapa_smoke.json`, change `model_family` and `config_path`, then run `demo/run_ecapa_hpo.py --demo-config ...`.

## Add a New Runner

1. Implement `agent/runners/my_runner.py` using the `RunnerAdapter` contract.
2. Register it in `agent/runners/__init__.py`.
3. Set `runner` and `implementation` in the demo config.
4. Run the same `demo/run_ecapa_hpo.py` entry.

The orchestration, data-processing, HPO, trial state machine, metrics recording, and artifact recording remain unchanged when models or runners are added.

# HPO optimization status

Existing P0 and pre-existing Campaign/cross-stage changes are retained. New protocols
remain opt-in. This document distinguishes implemented paths from acceptance evidence;
the complete hardware acceptance plan is **not yet complete**.

## Connected implementation

- Controller-owned tool preparation and result normalization surround immutable stage
  requests. Spawned workers execute raw runner calls and write isolated attempt/checkpoint
  artifacts and atomic stage results. Synchronous tools use the same execution logic.
- Trial batches execute through a bounded process pool; physical GPU assignments are
  disjoint, training may use DDP and evaluation uses one device. Results enter the scheduler
  in frozen candidate order. Stage IDs support replay and ledger settlement is idempotent.
- The resource ledger lock serializes controllers. Recovery verifies process creation time,
  terminates owned workers/descendants, then settles stages. Uncertain ownership blocks
  recovery. Completed durable results retain measured cost; unknown consumption is marked
  estimated and charged conservatively. A worker watchdog observes controller death.
- SpeechBrain saves per-rank recoverables, optimizer/scheduler, scaler, epoch, counters and
  Python/NumPy/Torch RNG at epoch boundaries. Continuation checks parameters, actual CSV
  digests, precision, world size and configuration. Incompatible startup restarts with a
  reason; errors after restoration begins fail the attempt. Persistent DataLoader workers
  currently disable continuation because their RNG state is not captured.
- Compatible continuation retains earlier evaluated checkpoint candidates. Validation top-k
  still requires every candidate to succeed before producing a comparable Trial result.
- Campaign confirmation reserves runs within the original training cap, uses independent
  fixed-configuration Studies and paired persisted fresh seeds, and reuses frozen training
  manifests. It does not feed search sampling. Reports distinguish search and confirmation
  results; incomplete confirmation has no ranking. A winning configuration is ranked by
  mean; its best observed seed provides the exported checkpoint.

## Configuration and preview

Context fragments (merge with existing runtime data/device options):

- `configs/experiments/hpo_p0_options.json`
- `configs/experiments/hpo_resume_options.json`
- `configs/experiments/hpo_parallel_options.json`
- `configs/experiments/hpo_full_options.json`

These are AgentTaskRequest context fragments, not a new CLI config-file format. GPU examples
use devices 0 and 1; adapt to the host. Do not combine gpu_pool with ddp_devices.

`main.py --dry-run` accepts the usual plan flags plus `--managed-execution`,
`--max-concurrent-trials`, `--gpu-pool`, `--gpus-per-trial`, `--promotion-training`,
`--confirmation-enabled`, `--confirmation-top-k`, and `--confirmation-seeds`.
It reports search/confirmation quotas, promotions, checkpoint evaluation cap and resources.
Effective protocol options persist; incompatible resume overrides are rejected.

## Comparison entry point

`scripts/experiments/run_hpo_optimization_comparison.py --request request.json --output results`
prints three plans without training. Add `--execute` to run isolated legacy, P0 and full
arms. Input contains `config_path`, `context`, `budget`, and optionally
`full_execution_options` and `enable_llm_advisor`. All arms receive the same budget.
Execution requires explicit `budget.max_gpu_hours`, existing data, verification config and
validation pairs. Config/pair hashes are frozen, and each arm has separate experiments and
memory. Output includes plans and structured workflow results with resource ledger links.
Do not interpret equal allowances as equal actual expenditure; check recorded costs before
comparing quality. No quality improvement or real benchmark result is claimed.

## Acceptance evidence

Tests use the project-local `.venv-hpo` environment. The existing global CUDA stack was
not modified. SpeechBrain 1.0.3 and test dependencies were added locally; Windows required
sentencepiece 0.2.0 for import compatibility.

- Real spawned-process end-to-end test: two search runs plus six confirmation runs;
  resume creates no duplicate training or confirmation and costs have 16 stage entries.
- Actual PyTorch/SpeechBrain checkpoint hooks: tiny deterministic CPU model, continuous
  versus segmented training, exact model parameter and learning-rate equality, restored
  epoch/update counters and RNG state. This is a state-continuity test, not a full recipe
  training smoke test.
- Real OS process tests: PID reuse protection, termination before recovery settlement,
  refusal on uncertain ownership, durable-result accounting and duplicate settlement.
- Full regression: **733 passed, 1 skipped** (POSIX-only file-lock test on Windows).
- Existing legacy, P0, checkpoint recovery, Campaign and CLI regression tests are retained.

Remaining acceptance work: full SpeechBrain training recipe continuation on real data,
GPU Trial isolation, single-Trial DDP restoration and DDP child failure cleanup, forced
Controller crash during active multi-GPU batches, and the fixed-data GPU-budget comparison.
One RTX 3060 Laptop GPU is visible and a CUDA tensor operation passed; multiple GPUs are not available. Fake-runner tests do not stand in for
these hardware checks. The plan is not marked fully accepted until they are completed.


## 2026-11-06: authoritative cost feedback and recoverable consultations

- New Campaigns create a ledger even without a numeric GPU/time ceiling. Resuming a
  legacy Campaign without a ledger retains legacy behavior; unknown historic cost is
  never reconstructed as a complete journal.
- HPOService loads a read-only, experiment/Trial-scoped projection of the ledger. Feedback,
  cross-budget evidence, Campaign cost estimates and decision effect reports share this
  source. Failed/retried stages and all checkpoint evaluations are included. GPU seconds
  sum allocated device time; summed stage durations are explicitly not parallel wall time.
- Missing/corrupt ledger reads and estimated/active stages retain incompleteness indicators.
  Legacy Trial durations remain readable with an explicit legacy source/completeness flag.
- Runtime strategy, cross-budget and Campaign-boundary consultations now persist bounded
  request attempts, errors and responses. Each consultation allows at most three application-
  level attempts (transport/SDK retries are separate). Saved responses are replayed without
  another advisor call. Interrupted unanswered requests may retry within the saved limit.
- Runtime proposal IDs deduplicate policy submission. The consultation's applied/rejected
  status is saved with the Study review and policy state. Exhaustion keeps deterministic
  fallback behavior and an explicit failure record. Old consumed records remain terminal.

Decision flow: frozen Study plan -> bounded candidate batch -> train/evaluate -> controller
commit + ledger settlement -> feedback analysis -> opportunity/evidence checks -> durable
advisor response -> deterministic approval -> future candidates -> measured decision effects.
Runtime changes may affect candidate sampler/configuration, search space, bounded proposals
and hypotheses. Active Study budgets, pruner, promotion quotas and reduction factor stay
frozen. Dedicated cross-budget actions use their reserved quotas. Campaign boundary choices
come from the deterministic feasible-plan menu and retain original total budget limits.

Validation for the 2026-11-06 change: **738 passed, 1 skipped** across tests/unit and
 tests/integration (Windows skips the POSIX shared-lock case). Added coverage verifies
ledger/feedback equality in process-based search and confirmation, failed/retried and
multi-checkpoint stage aggregation, Study isolation, unavailable ledger reporting,
transport retry, interrupted request recovery, saved-response replay and duplicate
proposal submission. Existing cross-stage/Campaign interruption tests now assert bounded
retry rather than permanent consumption. No real multi-GPU training claim is added.

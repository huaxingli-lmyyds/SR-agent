# SR-agent

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
- `scripts/experiments/`: comparison and model HPO launchers.
- `scripts/evaluation/`: evaluate a specific trained checkpoint.
- `scripts/admin/`: archive and clear experiment records/artifacts.
- `scripts/tools/`: environment and runtime inspection utilities.

Common commands:

```bash
python scripts/tools/check_remote_environment.py --require-cuda
python scripts/evaluation/evaluate_checkpoint.py --help
bash scripts/experiments/run_comparison_experiments.sh --suite smoke --dry-run
bash scripts/experiments/run_resnet_hpo_comparison.sh --dry-run
bash scripts/experiments/run_xvector_hpo_comparison.sh --dry-run
bash scripts/admin/archive_experiments.sh --model-family ecapa_tdnn
bash scripts/admin/clear_experiments.sh --model-family ecapa_tdnn
```

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
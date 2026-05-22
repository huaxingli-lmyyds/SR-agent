# Project Structure

This repository contains a SpeechBrain-based speaker verification system with an orchestration-first agent stack, a dedicated data-processing agent, and supporting tools.

## Top-Level Layout
- `agent/`: Core agents, tools, utilities, experiment records, prompts, and logs.
- `configs/`: Training and evaluation YAML configs.
- `datasets/`: VoxCeleb data and augmentation assets expected on disk.
- `models/`: Model definitions, including ECAPA-TDNN.
- `recipes/`: SpeechBrain recipes and data preparation scripts.
- `results/`: Generated output results.
- `speechbrain/`: Vendored SpeechBrain library.
- `main.py`: System entry point for the orchestration agent.
- `pyproject.toml`: Project metadata and dependencies.

## agent/ Layout
- `agents/`: Agent implementations, including the coordinator, HPO agent, and data-processing agent.
- `tools/`: LangChain tool wrappers for config, training, evaluation, history, diagnostics, reward, and data preparation.
- `utils/`: Core utilities such as runner, metrics, experiment tracking, logging, reward, config parsing, and path helpers.
- `prompts/`: Agent system prompts and templates.
- `experiments/`: Experiment records split by responsibility into `manage`, `dp`, `hpo`, plus configs and history.
- `logs/`: Agent run logs, including separate logs for manage, HPO, and data-processing agents.
- `memory/`: Persistent memory for agent runs.

## Key Modules
- `agent/agents/orchestrator.py`: Tool-based system coordinator that launches data-processing and HPO rounds.
- `agent/agents/react_agent.py`: HPO agent orchestration.
- `agent/agents/data_processing_agent.py`: Data preparation optimization agent focused on split ratio, sentence length, and prep flags.
- `agent/agents/base_agent.py`: Shared LangChain agent base wrapper.
- `agent/utils/runner.py`: Centralized SpeechBrain training and evaluation execution.
- `agent/utils/experiment_tracker.py`: Experiment record management with typed record separation.
- `agent/tools/config_tools.py`: Config read/update utilities with grouped concern-based views.
- `agent/tools/data_processing_tools.py`: VoxCeleb preparation tool wrapper with normalized dataset path handling.
- `agent/tools/training_tools.py`: Training tools and record integration.
- `agent/tools/evaluation_tools.py`: Evaluation tool and metrics extraction.
- `agent/tools/experiment_history_tools.py`: Separate history query tools for HPO, data-processing, and orchestration.
- `agent/tools/training_diagnostics_tools.py`: Curve analysis and fit diagnosis.
- `agent/tools/reward_tools.py` + `agent/utils/reward.py`: Multi-metric reward scoring.

## Configs
- `configs/train_ecapa_tdnn.yaml`: Training and data-preparation configuration.
- `configs/verification_ecapa.yaml`: Evaluation configuration.

## Data and Outputs
- `datasets/`: Expected to contain VoxCeleb data (`wav`, `noise`, `rir`, etc.).
- `agent/experiments/manage`: Orchestration records.
- `agent/experiments/dp`: Data-processing records.
- `agent/experiments/hpo`: HPO records.

## Notes
- Most tools are exposed via `agent/tools/__init__.py` and loaded by the relevant agent.
- Paths are resolved through project-root helpers so agents work consistently regardless of the working directory.
- Experiment records store training, evaluation, and orchestration metrics for tracking and comparison.

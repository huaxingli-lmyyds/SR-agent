# Project Structure

This repository contains a SpeechBrain-based speaker verification pipeline with an HPO (hyperparameter optimization) agent and supporting tools.

## Top-Level Layout
- agent/: Core HPO agent, tools, utilities, experiment records, and prompts.
- configs/: Training and evaluation YAML configs.
- datasets/: VoxCeleb data and augmentation assets (expected on disk).
- models/: Model definitions (ECAPA-TDNN).
- recipes/: SpeechBrain recipes and data preparation.
- results/: Output results (generated).
- speechbrain/: Vendored SpeechBrain library (local copy).
- main.py: Entry point for agent usage.
- pyproject.toml: Project metadata and dependencies.

## agent/ Layout
- agents/: Agent implementations (ReAct HPO agent and data processing agent).
- tools/: LangChain tool wrappers (config, training, evaluation, history, diagnostics, reward).
- utils/: Core utilities (runner, metrics, experiment tracking, logging, reward).
- prompts/: Agent system prompts and templates.
- experiments/: Experiment records and outputs (generated).
- logs/: Agent run logs (generated).
- memory/: Persistent memory for agent runs.

## Key Modules
- agent/agents/react_agent.py: Main HPO agent orchestration.
- agent/agents/data_processing_agent.py: Data preparation optimization agent (splits, sentence length, prep flags).
- agent/utils/runner.py: Centralized SpeechBrain training/evaluation execution.
- agent/utils/experiment_tracker.py: Experiment record management.
- agent/tools/training_tools.py: Train/evaluate tools and logging integration.
- agent/tools/data_processing_tools.py: VoxCeleb preparation tool wrappers.
- agent/tools/evaluation_tools.py: Evaluation tool and metrics extraction.
- agent/tools/training_diagnostics_tools.py: Curve analysis and fit diagnosis.
- agent/tools/reward_tools.py + agent/utils/reward.py: Multi-metric reward scoring.

## Configs
- configs/train_ecapa_tdnn.yaml: Training configuration.
- configs/verification_ecapa.yaml: Evaluation configuration.

## Data and Outputs
- datasets/: Expected to contain VoxCeleb data (wav, noise, rir, etc.).
- agent/experiments/: Experiment records and outputs per run.


## Notes
- Most tools are exposed via agent/tools/__init__.py and loaded by the agent.
- Experiment records store training/evaluation metrics for tracking and comparison.

# Project Logic and Innovations

This document summarizes the current orchestration-first runtime and the main design choices behind the refactor.

## Runtime Flow (High Level)
1. `main.py` now starts the orchestration entry point directly through `agent/agents/orchestrator.py`.
2. The coordinator agent builds a system prompt and coordinates data-processing and HPO sub-agents through tools.
3. Each agent loads only the tools relevant to its responsibility: config, data prep, training, evaluation, history, diagnostics, reward, and orchestration tools.
4. Config access is grouped by concern so agents can read and update data-processing fields, HPO fields, and model-structure fields separately.
5. Training and evaluation still go through `agent/utils/runner.py`, but path resolution is now normalized to project-root absolute paths.
6. Experiment metadata is written into separated directories for `manage`, `dp`, and `hpo` so records no longer collide.
7. Reward scoring continues to use EER and minDCF as the optimization signal.

## Key Execution Paths
- System entry
  - Entry: `main.py` -> `agent/agents/orchestrator.py`
  - Output: orchestration result, linked manage/dp/hpo experiment IDs, final summary
- Data processing
  - Agent: `agent/agents/data_processing_agent.py`
  - Tool: `agent/tools/data_processing_tools.py` -> `PrepareVoxCelebData`
  - Output: prepared CSVs, prep stats, data-processing experiment record
- Hyperparameter optimization
  - Agent: `agent/agents/react_agent.py`
  - Tools: config, training, evaluation, history, diagnostics, reward
  - Output: best config, training/evaluation results, HPO experiment record
- Evaluation
  - Tool: `agent/tools/evaluation_tools.py` -> `runner.run_evaluation`
  - Output: EER/minDCF, evaluation log, record update
- Training
  - Tool: `agent/tools/training_tools.py` -> `runner.run_training`
  - Output: train log, epoch metrics, model paths, record update
- Diagnostics
  - Tool: `agent/tools/training_diagnostics_tools.py`
  - Uses training curves to detect overfitting/underfitting trends
- Reward scoring
  - Tool: `agent/tools/reward_tools.py`
  - Reward: combines EER and minDCF into a single score
- Experiment history
  - Tools: `agent/tools/experiment_history_tools.py`
  - Separate query surfaces for HPO, data-processing, and orchestration histories

## Innovation Highlights
- Orchestration-first system entry
  - The top-level entry no longer treats HPO as the sole runtime. A dedicated coordinator drives the full data-processing + HPO loop.
- Responsibility-scoped agents
  - HPO, data-processing, and coordinator agents now have separate logs, separate experiment directories, and separate tool sets.
- Path normalization
  - Configuration paths and dataset paths are resolved through project-root helpers, so the system works regardless of the current working directory.
- Concern-grouped configuration access
  - Config tools expose grouped read/update views, making it clearer which fields belong to data preparation vs. model tuning.
- Experiment-first tracking
  - Every run persists metrics, logs, and outputs in typed experiment records, enabling reproducibility and comparison.
- Memory-aware agent workflow
  - The agent persists summaries of changes and outcomes to avoid repeating poor configurations.
- Dedicated data preparation agent
  - Dataset optimization is separated from model HPO, which makes the pipeline easier to reason about and debug.

## Where to Start
- Entry: `main.py`
- Coordinator: `agent/agents/orchestrator.py`
- HPO Agent: `agent/agents/react_agent.py`
- Data Processing Agent: `agent/agents/data_processing_agent.py`
- Configs: `configs/train_ecapa_tdnn.yaml`, `configs/verification_ecapa.yaml`
- Runner: `agent/utils/runner.py`
- Experiment records: `agent/experiments/manage`, `agent/experiments/dp`, `agent/experiments/hpo`

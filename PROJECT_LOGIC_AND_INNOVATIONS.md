# Project Logic and Innovations

This document summarizes how the system runs end-to-end and highlights the key design choices.

## Runtime Flow (High Level)
1. The HPO agent builds a system prompt from agent/prompts/hpo_system_prompt.txt.
2. The agent loads LangChain tools (config, training, evaluation, history, diagnostics, reward).
3. For each iteration, the agent updates configuration, trains, decides whether to evaluate, then analyzes results.
4. Training/evaluation calls are centralized in agent/utils/runner.py.
5. Experiment metadata and metrics are recorded in agent/utils/experiment_tracker.py.
6. The reward tool scores experiments using EER and minDCF to guide selection.

## Key Execution Paths
- Training
  - Tool: agent/tools/training_tools.py -> runner.run_training
  - Output: train_log, epoch metrics, model paths, experiment record
- Data Processing
  - Agent: agent/agents/data_processing_agent.py
  - Tool: agent/tools/data_processing_tools.py -> PrepareVoxCelebData
  - Output: prepared CSVs, prep stats, updated data-related config
- Evaluation
  - Tool: agent/tools/evaluation_tools.py -> runner.run_evaluation
  - Output: EER/minDCF, evaluation log, experiment record update
- Diagnostics
  - Tool: agent/tools/training_diagnostics_tools.py
  - Uses training curves to detect overfitting/underfitting trends
- Reward Scoring
  - Tool: agent/tools/reward_tools.py
  - Reward: combines EER and minDCF into a single score

## Innovation Highlights
- Centralized SpeechBrain calls
  - Training and evaluation run through a single runner module for consistent overrides and error handling.
- Experiment-first tracking
  - Every run persists metrics, logs, and outputs in a structured experiment record, enabling reproducibility and comparison.
- Training-state diagnostics
  - Curve-based tools summarize convergence and detect overfitting/underfitting to guide decisions.
- Reward-guided search
  - A dedicated reward module ranks experiments using multi-metric feedback (EER + minDCF).
- Memory-aware agent workflow
  - The agent persists summaries of changes and outcomes to avoid repeating poor configurations.
- Dedicated data preparation agent
  - Separates dataset optimization (split ratio, sentence length, prep flags) from model HPO.

## Where to Start
- Entry: main.py
- Agent: agent/agents/react_agent.py
- Data Processing Agent: agent/agents/data_processing_agent.py
- Configs: configs/train_ecapa_tdnn.yaml, configs/verification_ecapa.yaml
- Runner: agent/utils/runner.py
- Experiment records: agent/experiments/

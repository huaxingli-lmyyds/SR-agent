# SR-Agent Architecture Analysis

## Scope
This document covers the code under agent/ as requested.

## File Responsibilities

### Core module surface
- agent/__init__.py
  - Public API re-export of agents, tools, and utils.
  - Version and __all__ export list.

- agent/README.md
  - High-level usage, requirements, and command examples.
  - Describes tool names and expected workflows.

### Agents
- agent/agents/__init__.py
  - Export surface for HPO agent and data-processing agent.

- agent/agents/react_agent.py
  - Main ReAct-style HPO agent using LangChain v1 create_agent.
  - Loads tool set (config, training, evaluation, experiment history).
  - Builds system prompt and post-processes agent outputs.
  - Extracts best config from intermediate steps and final answer.

- agent/agents/data_processing_agent.py
  - Data-prep focused agent using LangChain v1 create_agent.
  - Loads config tools + data preparation tool.
  - Extracts best data-processing config from tool outputs.

### Prompts
- agent/prompts/hpo_agent_prompt.txt
  - Long-form structured optimization playbook and reporting format.

- agent/prompts/hpo_prompt.txt
  - Shorter operational rules and parameter ranges for HPO.

### Tools
- agent/tools/__init__.py
  - Tool export surface (config, training, evaluation, history, data prep).

- agent/tools/config_tools.py
  - LangChain tools for reading, updating, listing, and resetting YAML config.
  - Handles config path resolution, backup, and caching.

- agent/tools/training_tools.py
  - LangChain tools for training, evaluation dispatch, and log analysis.
  - Creates experiment record, calls training pipeline, parses logs.

- agent/tools/evaluation_tools.py
  - LangChain tool to run evaluation pipeline and parse EER/minDCF.
  - Updates experiment tracker with evaluation results.

- agent/tools/experiment_history_tools.py
  - LangChain tools to list, compare, and fetch experiment results.

- agent/tools/data_processing_tools.py
  - LangChain tool to run VoxCeleb data preparation and CSV stats.

### Utils
- agent/utils/__init__.py
  - Re-exports for path, config, tracker, logger, metrics utilities.

- agent/utils/path_tool.py
  - Project path resolution, file helpers, backup, and YAML utils.

- agent/utils/config_parser.py
  - YAML parsing with tag handling (SpeechBrain-style tags and refs).
  - Load/validate/update/compare/export config.

- agent/utils/experiment_tracker.py
  - Experiment record creation, update, listing, and history management.

- agent/utils/logger.py
  - Structured logging and experiment logging helpers.
  - In-memory log buffer for agent context.

- agent/utils/metrics.py
  - Training log parsing, EER/minDCF calculation, and comparisons.
  - Optional plotting helpers for metrics visualization.

## Responsibility Overlaps
- Config handling is split between config_tools.py and utils/config_parser.py.
  - config_tools exposes LangChain tool wrappers; config_parser does core parsing.
  - Both produce configuration summaries and formatting.

- Experiment comparison exists in both:
  - agent/tools/experiment_history_tools.py (tool layer)
  - agent/utils/experiment_tracker.py and agent/utils/metrics.py (core logic)

- Evaluation logic overlaps in training_tools.EvaluateModel and evaluation_tools.RunEvaluation.
  - EvaluateModel is a dispatcher that calls RunEvaluation via tool invoke.

- Metrics extraction is duplicated in:
  - training_tools._parse_training_log
  - utils/metrics.MetricsExtractor

## SpeechBrain Integration Locations
- agent/tools/data_processing_tools.py
  - Imports speechbrain.utils.data_utils.download_file
  - Calls recipes.voxceleb.voxceleb_prepare.prepare_voxceleb

- agent/tools/training_tools.py
  - Calls recipes.train_pipeline.train_pipeline (SpeechBrain training pipeline)
  - Parses SpeechBrain-style training logs

- agent/tools/evaluation_tools.py
  - Calls recipes.eval_pipeline.eval_pipeline (SpeechBrain evaluation pipeline)
  - Parses SpeechBrain evaluation logs for EER/minDCF

- agent/utils/config_parser.py
  - Parses SpeechBrain YAML tags and reference patterns (!ref, !new, !apply)

- agent/utils/metrics.py
  - Log parsing patterns match SpeechBrain training/eval output

- agent/README.md
  - References SpeechBrain scripts and usage paths

## Tool Input/Output Consistency Notes
- All tools return human-readable strings; no structured dict outputs.
  - Consistent for chat display, but limits programmatic chaining.

- Parameter naming is inconsistent across tools:
  - config_tools.ListConfigParameters uses path while others use config_path.
  - EvaluateModel uses experiment_id, RunEvaluation uses experiment_id plus config.

- Parameter types are inconsistent with agent prompt guidance:
  - Some tools accept non-string types (bool/list) in signatures.
  - data_processing_tools expects string inputs for parsing lists/bools/floats.
  - experiment_history_tools.CompareExperiments expects List[str] not string.

- Output formatting is inconsistent:
  - Some tools return multi-line summaries with emoji headers.
  - Others return terse or raw error strings without uniform schema.

- Tool behaviors overlap and may confuse agent planning:
  - EvaluateModel calls RunEvaluation and then reads tracker, but discards
    RunEvaluation return text.
  - AnalyzeResults re-parses logs while metrics.py offers similar extraction.

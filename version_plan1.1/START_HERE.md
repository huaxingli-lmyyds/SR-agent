# version_plan1.1 交付入口

四项任务见 TASKS.md，执行过程见 PROGRESS.md。所有新增代码、文档、报告和实验预览位于本目录；主项目已有源码没有被覆盖。

|内容|位置|
|---|---|
|整体职责与扩展契约|ARCHITECTURE.md|
|数据集和指标依据|RESEARCH.md|
|实际代码及配置说明|project/、CHANGELOG.md|
|数据闭环复核与下一步|DATA_LOOP_REVIEW.md、EXTENSIONS.md|
|实验设计与可运行预览|EXPERIMENTS.md、experiments/matrix.json、experiments/preview.py|
|测试结果与限制|VALIDATION.md、validation/|
|基线、最终差异与可审查补丁|SOURCE_MANIFEST.json、validation/change_manifest.json、implementation.patch|

代码优先查看 agent/agents/coordination.py、orchestration_workflow.py、data_processing_agent.py，以及 agent/data_processing/feedback.py、impact.py、workflow.py、service.py。均在 project/ 下。

快速验证：在本目录运行 `./validate.ps1 -Focused`；实验预览运行 `../.venv-hpo/Scripts/python.exe experiments/preview.py`。完整使用说明见 EXPERIMENTS.md。

实验默认缺少真实数据协议和GPU预算，保持预览状态。使用现有 SpeechBrain 环境时从 project/ 作为工作目录运行；不要把测试成功误写为识别效果提升。

若后续将修改合回主项目，先核对 SOURCE_MANIFEST.json，检查 implementation.patch 与主项目新增修改是否冲突。本轮交付没有自动进行合并。

## 版本控制范围

Git 保留独立源码、测试、配置、本文档集、实现补丁，以及基线/最终源码清单、变更清单、闭环审查和测试结果索引。`experiments/generated/` 和 `validation/` 中的原始测试报告、日志、缓存、临时目录及其他运行输出仅保留在本地，由 `.gitignore` 排除；文档中的历史报告路径不会全部包含在克隆结果中，可通过上述验证和预览入口重新生成。

`SOURCE_MANIFEST.json` 记录隔离副本建立时的历史基线；本次归档额外修改了仓库根目录 `.gitignore`，因此核对当前主项目时应单独检查该文件，其余基线文件保持不变。

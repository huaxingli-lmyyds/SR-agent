# 五组超参数优化对比实验

入口：[run_hpo_benchmark.py](E:/voxceleb/sr-agent/SR-agent/scripts/experiments/run_hpo_benchmark.py)

配置：[five_group_ecapa.json](E:/voxceleb/sr-agent/SR-agent/configs/experiments/five_group_ecapa.json)

在 `E:\voxceleb\sr-agent\SR-agent` 项目目录执行。示例中的 Python 应为项目的真实训练环境，包含兼容的 PyTorch/CUDA、SpeechBrain、Optuna、LangGraph 和项目依赖。这里的五组比较针对 HPO 子系统；各组共享数据，不混入数据清洗或 Coordinator 路由差异。

原 comparison/classic Optuna 入口、ResNet/XVector 专用比较启动器及其专用测试已移除。现在仅维护本入口；切换模型应使用匹配的训练/验证配置，而不是复制另一套实验循环。核心训练、评估、HPO 和恢复回归测试保留。

## 1. 实验组及默认预算

| `--only` 名称 | 方法 | 默认训练配额 | LLM |
| --- | --- | --- | --- |
| `default_parameters` | 从训练 YAML 读取默认超参数，只训练一次 | 1 × 24 epochs，全数据 | 不使用 |
| `random_search` | 固定随机搜索 | 9 × 24 epochs，全数据 | 不使用 |
| `tpe` | 固定 TPE，逐个候选反馈 | 9 × 24 epochs，全数据 | 不使用 |
| `tpe_successive_halving` | 固定 TPE + 确定性 SH | 9 × (3 epochs, 25%) → 3 × (8 epochs, 50%) → 1 × (24 epochs, 100%) | 不使用 |
| `system` | 实际 HPOAgent 初始规划、只读工具、批间复盘 | 与 TPE+SH 相同，候选每批 3 个 | 使用 |

SH 和系统组分别有 13 次训练额度。晋级子 Trial 使用独立输出目录，从头训练；不继承父 Trial checkpoint。中断恢复只恢复同一个 Trial 自己的训练。

默认组的 lr、batch_size、margin、weight_decay 来自实际 YAML，不使用另一套写死的默认值。为了统一最终保真度，所有组的完整训练 epochs 使用实验配置中的 `full_epochs`，因此默认组并非原 YAML 训练轮数完全不变。

固定基线显式返回 keep 决策，不触发默认规则控制器。系统组允许在 `random_search/tpe/adaptive_search` 之间选择和调整配置；可以在初始硬边界内调整空间，但不能增加参数、扩大到原边界之外、改变预算/晋级规则，也不能使用 `agent_proposal`。

**默认是固定搜索配额实验，不是等 GPU 时间实验。** 单保真搜索约为 `9×24=216` 个全数据 epoch；SH 约为 `42.75` 个全数据 epoch，不含评估/控制开销。不能把两者称为相同计算预算。可对比质量与实际成本，系统与 TPE+SH 的预算安排则相同。若要严格等 GPU 小时的结论，需要另加实测资源门禁；本入口没有假装提供该功能。

## 2. 准备验证集与最终测试集

需要两个本地 pairs 文件，格式如下：

```text
1 id10001/video1/00001.wav id10001/video2/00002.wav
0 id10001/video1/00001.wav id10002/video1/00003.wav
```

每个文件都必须包含正、负样本。路径使用相对 VoxCeleb utterance ID。两个文件的说话人集合必须不重叠；检测到交集会在训练前报错。不要把同一个官方测试文件同时作为调参验证集和最终测试集。

如果还没有固定的 HPO 验证协议，先按
[`generate_hpo_validation.py`](../tools/GENERATE_HPO_VALIDATION.md)
中的命令从 VoxCeleb1 dev 说话人自动生成。生成器按说话人排除
`veri_test2.txt` 测试集合，并记录 seed、数据清单和协议哈希；同一组对比实验必须复用同一份输出。

入口复制 pairs 和训练/验证 YAML 并记录 SHA256，将两类评估 pairs 合并为训练排除清单。数据准备已修正为排除 pairs 左右两侧的说话人，并支持 Windows 原生路径。它不会替你重新设计一个合理的验证划分，也不会对整个音频库逐文件做内容哈希；正式实验请自行冻结数据版本，并检查生成的 train/dev 清单与说话人覆盖。

搜索过程中只读取验证成绩。最终测试必须在所有搜索运行结束后，通过单独命令执行；结果写在独立目录，不进入 HPO 的实验库和记忆。

## 3. 查看计划：无训练、无 LLM、无输出写入

```powershell
python scripts/experiments/run_hpo_benchmark.py --dry-run
```

只查看一个种子或部分方法：

```powershell
python scripts/experiments/run_hpo_benchmark.py --dry-run --seeds 11 --only default_parameters,random_search,tpe,tpe_successive_halving,system
```

`--dry-run` 不要求 torch、LangGraph、Optuna 等运行依赖，也无需真实音频路径。若提供 pairs，则会执行 pairs 格式和说话人隔离检查。

## 4. 正式运行五组

把下列数据和 pairs 路径换成自己的实际路径。输出目录使用一个新的实验目录。

```powershell
python scripts/experiments/run_hpo_benchmark.py --config configs/experiments/five_group_ecapa.json --data-folder E:/datasets/VoxCeleb --validation-pairs E:/datasets/protocols/hpo_validation.txt --test-pairs E:/datasets/protocols/heldout_test.txt --output-dir E:/experiments/hpo_five_groups
```

默认种子为 11、22、33。每个种子内的方法运行顺序由该种子打乱；各组训练种子相同，但不同种子之间不同。每组有独立训练配置、HPO 记录和 LLM 记忆。

`data_prep_seed` 默认为 0，专门固定 train/dev 划分，各组和不同训练种子使用同一划分。准备缓存位于各自 `runs/.../prep_cache`，验证和最终测试使用不同子目录。准备前后保存/恢复 Python 随机状态，避免冷缓存额外消耗训练随机序列。

噪声/RIR 在各组计时开始前准备到本批实验的 `assets` 目录；完成后各组只读相同音频和 CSV，不再重复下载或改写原数据目录。`assets.json` 记录注释哈希及独立 `setup_seconds`，此开销不混入某一个方法的成绩。首次运行需要下载增强资源；当前支持项目内置的增强准备函数，未知自定义回调会报错。输入音频需提前解压，实验不会调用 `voxceleb_source` 向数据目录写入文件。

正式训练前检查验证/测试音频是否存在，以及训练/验证的特征和模型显式参数是否冲突。当前要求 `score_norm=none`；需要分数归一化时应先设计独立、冻结的 cohort 协议，不能隐式从测试说话人构造 cohort。

只跑传统基线，不需要 LLM API 配置：

```powershell
python scripts/experiments/run_hpo_benchmark.py --data-folder E:/datasets/VoxCeleb --validation-pairs E:/datasets/protocols/hpo_validation.txt --test-pairs E:/datasets/protocols/heldout_test.txt --only default_parameters,random_search,tpe,tpe_successive_halving --output-dir E:/experiments/hpo_baselines
```

系统组沿用项目已有 `.env` 中的模型服务配置，例如 `ZHIPUAI_API_KEY`、`ZHIPUAI_API_BASE_URL`。API 地址应为兼容接口根地址，不要包含 `/chat/completions`。模型名称和温度在 JSON 中配置。没有可用 LLM 时，优化可能按安全回退继续，但结果会记录 `advisor_failures` 和 `controller_degraded`；这种运行不能当成智能体有效性证据。

各个组的独立运行同样支持 `--only system` 或 `--only tpe`。如果分开运行，请保持相同输入、配置、种子和数据版本；结果目录不要复用。更推荐一次声明全部组，让入口自动隔离和汇总。

## 5. 中断恢复

```powershell
python scripts/experiments/run_hpo_benchmark.py --resume --output-dir E:/experiments/hpo_five_groups
```

恢复时只传输出目录，不重新传入 config/seeds/pairs，入口使用冻结配置并校验快照哈希：

- 已有 `result.json` 的运行不会重跑，包括已提交的失败运行。
- 尚未提交结果的运行会寻找原实验 ID，恢复同一个 Study；不会另建重复 Study。
- 若在创建 Study 前中断，则重新开始该尚未启动的 Study。
- 不扩大训练预算、不重新执行已提交的复盘；重试也受原预算约束。
- 硬终止时，未落盘的耗时无法精确恢复，`cost_may_be_incomplete=true` 会提示不能直接拿该耗时做效率结论。
- 目录必须仍位于原始绝对路径；关键 Python 源码、提示词和已记录依赖版本必须一致，防止复制目录后写回原实验或混用不同实现。
- 本次隔离协议升级为 v2；旧 v1 结果保留可读，但不允许混入 v2 恢复，应使用旧代码继续旧实验或新建输出目录重新运行。

本入口每次重复只有一个 Study，因而不涉及尚未完整接通的多 Study Campaign 自动续跑。已失败且终态提交的运行若要重新做，应使用新输出目录，不手工删除旧结果来选择性重试。

同一个输出目录有操作系统文件锁，禁止两个进程同时写。进程退出后锁自动释放，不需要删除锁文件。

真实 SpeechBrain 运行还共用本机临时目录中的执行锁：同一临时目录下通过本入口启动的任务，即使输出目录不同，也不能并行抢占设备（保守地连不同 GPU 也串行）。这无法阻止其他训练程序占用 GPU；正式测时仍须自行保持设备空闲，并保持数据/增强音频不被外部修改。

## 6. 独立最终测试

```powershell
python scripts/experiments/run_hpo_benchmark.py --evaluate-test --output-dir E:/experiments/hpo_five_groups
```

只有全部计划运行均已提交终态后才能执行。失败运行保留为 `search_failed`，成功运行只评估锁定的完整预算赢家：

- 不重新训练、不再搜索、不调用 LLM。
- 检查赢家 checkpoint 和运行配置的哈希。
- 不根据测试分数重新选赢家，也不把测试成绩写入原 HPO 记录。
- 已有最终测试结果不会重复评估；失败结果也保留。

## 7. 结果在哪里

| 文件 | 内容 |
| --- | --- |
| `benchmark_plan.json` | 组别、种子、资源配额、输入哈希、依赖版本、关键源文件哈希与运行顺序 |
| `assets.json` / `assets/` | 各组启动前准备的只读增强输入、CSV 哈希与单独统计的准备耗时 |
| `results.csv` / `results.json` | 每组每个种子的验证 EER/minDCF、完成状态、实际训练调用数、训练/评估/端到端时间、LLM 状态 |
| `summary.json` | 各组成功数、失败数、验证 EER 均值/标准差、系统减基线的逐种子配对差 |
| `heldout_results.csv` / `heldout_results.json` | 最终测试 EER/minDCF，单独执行测试后生成 |
| `heldout_summary.json` | 最终测试 EER 分布、均值、标准差及成功数 |
| `runs/<组名>_seed_<种子>/locked_winner.json` | 同最高预算验证选出的赢家、超参数、checkpoint、哈希 |
| `runs/.../trials.json` | 全部 Trial 的参数、预算、指标、晋级和来源信息 |
| `runs/.../attempts.jsonl` | 每次执行的开始/结束事件、成本、错误和关联 Trial |
| `runs/.../advisor.jsonl` | LLM 逻辑咨询请求的输入、最终响应和延迟 |
| `runs/.../proposals.jsonl` | 原始提案、限制后提案及被禁止的字段 |
| `runs/.../hpo/<实验ID>/hpo_study/study.json` | 实际采用的策略阶段、确定性决策、恢复状态与候选审计 |

`training_seconds/evaluation_seconds` 是对应调用的实测墙钟，不是 GPU 利用率积分。`advisor_requests` 是逻辑咨询次数，一个咨询可能包含多轮工具/模型调用；`final_response_usage` 仅是最后一条响应的 usage，不可当作全部 LLM token/费用。

指标无效、没有完整确认、checkpoint 缺失、固定组发生策略变化都不能作为成功结果。汇总保留失败；配对差只基于双方成功的种子，属于成功子集分析，不会自动宣称统计显著。

## 8. 调整规模与限制

修改 JSON 后用新输出目录运行。`full_trials` 控制 Random/TPE 的完整训练候选数；`sh_initial_trials`、`promotion_limits`、`budgets` 控制两个 SH 组；`full_epochs` 必须等于最后一层 epochs，最后一层 data_fraction 必须为 1.0。

默认每个种子最多 `1+9+9+13+13=45` 次训练调用，三个种子最多 135 次；其中 25%/50% 数据的阶段仍从头训练。这与之前“只比较三核心组、全用 SH”的 117 次预实验不同。

建议先复制 JSON 为小规模配置，在自己的 GPU 上验证训练/评估通路及预计耗时，再运行正式实验。很小的 TPE 试验若未超过启动样本数，只能说明流程可运行，不能评判 TPE 的优化能力。

默认不使用实时 epoch early_stop，不执行额外模型结构搜索，也不提供严格等时间资源门禁。正式报告应明确这些边界。不同训练 seed 已实际写入 YAML，然而 GPU 算子和在线 LLM 并不保证位级确定性。

## 9. 自动测试

在已安装项目测试依赖的环境运行：

```powershell
python -m pytest tests/unit/test_hpo_benchmark.py tests/unit/test_benchmark_voxceleb_split.py tests/unit/test_experiment_tracker.py tests/integration/test_hpo_benchmark.py -q
```

这些测试使用真实 HPO 调度器、假训练器和假 LLM，不消耗 GPU 训练或模型 API 费用。真实 GPU 效果需要你在准备好数据与环境后运行上述正式命令。

2026-08-28 清理后验证：上述定向回归 45 项通过；全套测试 334 通过、1 跳过、3 失败。失败仍为两项缺少 PyTorch 的音频兼容测试，以及既有 `test_debug_subset_preview_version_hashes_and_quality_gate` 的 affected_samples 断言；没有删除或跳过这些核心测试。此次未执行真实 GPU 训练或在线 LLM 调用，不能据此声称真实环境已完全验收。

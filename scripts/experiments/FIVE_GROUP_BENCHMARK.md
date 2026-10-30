# 五组超参数优化对比实验

入口：[run_hpo_benchmark.py](run_hpo_benchmark.py)

配置：[five_group_ecapa.json](../../configs/experiments/five_group_ecapa.json)

在项目根目录执行。示例中的 Python 应为项目的真实训练环境，包含兼容的 PyTorch/CUDA、SpeechBrain、Optuna、LangGraph 和项目依赖。这里的五组比较针对 HPO 子系统；各组共享数据，不混入数据清洗或 Coordinator 路由差异。

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

**默认是固定搜索配额实验，不是等 GPU 时间实验。** 单保真搜索约为 `9×24=216` 个全数据 epoch；SH 约为 `42.75` 个全数据 epoch，不含评估/控制开销。不能把两者称为相同计算预算。可对比质量与实际成本，系统与 TPE+SH 的预算安排则相同。配置可显式设置 `max_gpu_hours`、`max_wall_seconds`，作为每个组/种子的 Campaign 硬上限；提前耗尽且没有完整确认的运行会保留为失败。设置相同上限也不保证各组实际耗时相等。

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

噪声/RIR 在各组计时开始前统一准备。若 `<data-folder>/noise` 和 `<data-folder>/rir` 已包含 `.wav`，入口直接只读复用本地音频，只在本批实验的 `assets` 目录生成 CSV；对应本地目录缺失时才使用训练 YAML 的 URL 下载。各组共享同一音频和 CSV，不重复准备或改写原数据目录。`assets.json` 记录来源模式、路径、音频清单签名、注释哈希及独立 `setup_seconds`，此开销不混入某一个方法的成绩。启动后不得增删本地增强音频，否则恢复会拒绝继续。当前仅支持项目内置的增强准备函数，未知自定义回调会报错。输入音频需提前解压，实验不会调用 `voxceleb_source` 向数据目录写入文件。

SpeechBrain 训练快照会根据数据目录与统一训练排除清单冻结实际训练说话人数，并覆盖 `out_n_neurons`。生成 `train.csv` 后还会核对其中的实际说话人数；因此 VoxCeleb1-only 实验不会沿用基础 YAML 中面向 VoxCeleb1+2 的 7205 类分类头。

正式训练前检查验证/测试音频是否存在，以及训练/验证的特征和模型显式参数是否冲突。当前要求 `score_norm=none`；需要分数归一化时应先设计独立、冻结的 cohort 协议，不能隐式从测试说话人构造 cohort。

在当前 `score_norm=none` 协议下，每次 verification 只准备 `enrol.csv/test.csv`，不会生成或加载 `train.csv/dev.csv`。未来启用 Z/T/S-Norm 时才允许准备冻结 cohort 对应的 `train.csv`，`dev.csv` 不参与 verification。

只跑传统基线，不需要 LLM API 配置：

```powershell
python scripts/experiments/run_hpo_benchmark.py --data-folder E:/datasets/VoxCeleb --validation-pairs E:/datasets/protocols/hpo_validation.txt --test-pairs E:/datasets/protocols/heldout_test.txt --only default_parameters,random_search,tpe,tpe_successive_halving --output-dir E:/experiments/hpo_baselines
```

系统组沿用项目已有 `.env` 中的模型服务配置，例如 `ZHIPUAI_API_KEY`、`ZHIPUAI_API_BASE_URL`。API 地址应为兼容接口根地址，不要包含 `/chat/completions`。模型名称和温度在 JSON 中配置。没有可用 LLM 时，优化可能按安全回退继续，但结果会记录 `advisor_failures` 和 `controller_degraded`；这种运行不能当成智能体有效性证据。

`advisor_failures` 统计请求失败，`invalid_advisor_proposals` 统计未形成有效提案的咨询，
包括解析/schema 错误及请求失败回退；两者可能重叠，不应相加。
只要存在部分失败，系统组也会标记 `controller_degraded`；训练成功状态与控制器质量分别保留。
日志重定向到文件时自动关闭训练、数据准备及 embedding 的进度条，保留 epoch 汇总和错误信息。

LLM 咨询使用独立 HTTP 客户端，SDK 隐式重试关闭；每次请求遇到连接、超时、429/5xx 等暂时故障最多尝试三次，
以 1/2 秒退避重试。鉴权和普通参数错误立即失败。工具轮次、网络重试和一次格式重新生成共享
300 秒咨询时限，并受 Campaign 剩余墙钟预算约束；单次请求超时最多 120 秒。
网络耗时由 HTTP 客户端的超时控制，超过总预算后返回的响应不会被采用。
`advisor.jsonl` 保存每次传输尝试、脱敏后的异常因果链、HTTP 状态与请求 ID；
`proposals.jsonl` 保存格式校验和重新生成记录。JSON 模式仍须经过严格 schema、证据引用和策略门禁，
不能靠删除非法字符修复模型内容。工具声明使用 strict 模式以兼容 SDK 的 JSON 自动解析。
服务接受 JSON 模式并不保证所有回复都符合格式；实际探针已验证说明文字导致的失败能通过一次重新生成恢复。
只有服务明确报告不支持 JSON 模式/工具时才降级对应能力。

默认尊重环境代理。服务器能直连服务时，可在项目 `.env` 设置 `SR_AGENT_LLM_PROXY_MODE=direct`，
只让 LLM 客户端直连；`environment` 恢复环境代理。不会在失败时暗中切换网络出口。
新实验前可用虚构内容检查链路（不会读取训练数据，仍会产生少量 API 调用）：

```bash
python scripts/tools/check_llm_connection.py --requests 3
```


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

真实 SpeechBrain 默认共用本机临时目录中的独占执行锁。Linux 上可在 JSON 中显式设置
`parallel_single_gpu: true`，并将每个进程的 `CUDA_VISIBLE_DEVICES` 设为一张卡的完整
`GPU-...` UUID、`device` 设为 `cuda:0`，不使用 DDP。此时全局锁使用共享模式，
每张卡另有独占锁：不同卡可并行，相同卡或相同输出目录仍互斥；默认串行模式与并行
模式也互斥。恢复时校验冻结的 GPU UUID。数字索引或多卡可见列表不能启用此模式，
以免设备别名造成重复占卡。该锁无法阻止入口以外的训练程序，实际测时仍需记录外部负载。

快速配置 [five_group_ecapa_vox1_fast.json](../../configs/experiments/five_group_ecapa_vox1_fast.json)
保留原 20.8M ECAPA 网络，使用 BF16、batch 32、三个种子；完整训练 6 轮，SH 为
9×1 → 3×3 → 1×6 轮。训练数据须配套冻结为 256 个身份、每人四个会话各三个文件，
按会话 75/25 划分；验证保留 100 个隔离身份及 2,000 对，最终测试保持原协议。
该配置用于快速代理对照，不能当作完整 VoxCeleb1 训练效果。学习率半周期为 72 个
更新，对应此数据与 batch 的一轮；更换训练规模时必须重新计算。

## 6. 独立最终测试

```powershell
python scripts/experiments/run_hpo_benchmark.py --evaluate-test --output-dir E:/experiments/hpo_five_groups
```

只有全部计划运行均已提交终态后才能执行。失败运行保留为 `search_failed`，成功运行只评估锁定的完整预算赢家：

- 不重新训练、不再搜索、不调用 LLM。
- 检查赢家 checkpoint 和运行配置的哈希。
- 不根据测试分数重新选赢家，也不把测试成绩写入原 HPO 记录。
- 已有最终测试结果不会重复评估；失败结果也保留。

已完成搜索的单 GPU 任务可显式选择另一张卡做测试。此选项只允许用于 `--evaluate-test`，
不改变训练计划；源码、依赖、输入和赢家 checkpoint 的冻结校验照常执行。
实际测试 GPU 写入 `evaluation_assignments.jsonl` 和各测试结果。

```bash
python scripts/experiments/run_hpo_benchmark.py --evaluate-test --output-dir results/NEW_RUN --evaluation-gpu GPU-完整UUID
```

具有 `launch_manifest.json` 的多任务 Campaign 可使用独立评估队列：

```bash
python scripts/experiments/run_heldout_queue.py --campaign-dir results/NEW_CAMPAIGN --gpus 4,5,6,7 --dry-run
python scripts/experiments/run_heldout_queue.py --campaign-dir results/NEW_CAMPAIGN --gpus 4,5,6,7
```

队列在显式 GPU 池中选择空闲卡，默认门槛为 4096 MiB 可用显存、利用率不高于 10%，
可用 `--min-free-mib` 调整其他模型所需显存。它与原 supervisor 共用互斥锁；原 supervisor
应先退出。连续无可用卡且无活动任务时，默认等待 1800 秒后退出并保留待办；单任务也有
1800 秒执行时限。`heldout_queue_status.json` 记录等待原因、设备和任务状态，
日志写入 `heldout_queue_logs/`。所有搜索终态提交后才会开始测试，已有测试成功或失败结果均不重跑。
队列结束后汇总 `heldout_results.json/csv`；队列状态单独保存，不改写历史训练调度记录。

2026-10-30 修复前的实验已全部完成；它们继续保留原源码指纹。新源码不能直接续跑旧冻结计划，
也不要改写旧指纹来跳过校验。后续实验使用新输出目录和新源码快照。

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

2026-10-23 启动检查：完整回归 568 通过、1 跳过；115 个生产 Python 文件语法检查和
Ruff F/E9 检查通过。GPU 5 的 BF16/batch 32 实际一轮训练、验证和 checkpoint 锁定成功。
后续并行锁验证与本批实验的最终状态以对应实验目录报告为准。
并行锁补丁后的完整回归为 574 通过、1 跳过；GPU 0、2 的并行真实校准均成功。

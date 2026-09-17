# 专家固定配置基线

入口：`scripts/experiments/run_expert_baseline.py`。运行定义：`configs/experiments/expert_ecapa.json`。

这是“给定专家配方，直接训练与评估”的独立基线，不创建 Agent、Study、Campaign，不执行 Random/TPE/SH 搜索，也不根据 EER/minDCF 改参数或挑选最佳种子。

## 配置含义

目前没有额外提供经专家确认的参数文件，因此默认使用仓库现有 ECAPA 参考配方，不声称它是专家最优配置：

| 参数 | 默认训练 YAML 中的值 |
| --- | --- |
| seed | 1986 |
| number_of_epochs | 10 |
| lr | 0.001 |
| batch_size | 32 |
| margin | 0.2 |
| weight_decay | 0.000002 |

这些值不是脚本写死的默认超参数，而是读取 `configs/train_ecapa_tdnn.yaml`。模型结构、优化器、学习率调度和增强方式也沿用该 YAML。JSON 只配置文件路径、运行设备、精度、数据准备种子和重复种子，不接受搜索空间或额外训练预算字段。

如果你有真正的专家配方，用 `--train-config` 指定它；若模型结构不同，同时指定匹配的 `--validation-config`，并在 JSON 中设置对应 `model_family`。修改配方后必须使用新输出目录，不修改已开始实验的冻结快照。

与原五组入口的 `default_parameters` 有一个关键区别：原入口用统一的 `full_epochs`（默认 24），这个入口使用专家 YAML 本身的训练轮数（目前为 10）。它们不是两个不同搜索算法，也不能把默认结果当成相同训练预算的比较。若要严格等训练预算，请事先准备训练轮数一致的专家 YAML，并明确报告这一调整。

## 执行顺序

1. 校验输入、模型兼容性和验证/测试说话人隔离，冻结配置及 pairs。
2. 在本输出目录准备共享只读增强资源；记录独立准备耗时。
3. 每个 seed 训练一次：使用全部可用训练数据，不进行分阶段筛选或晋级。
4. 检查训练日志已完成配置要求的 epochs，锁定训练器选出的 checkpoint 及哈希。
5. 在验证 pairs 上计算 EER、minDCF；不会据此更换 checkpoint 或参数。
6. 单独执行 `--evaluate-test`，使用同一 checkpoint 计算最终测试指标，并汇总所有 seed。

SpeechBrain 的 checkpoint 沿用现有训练配方的选择规则：分类验证 `ErrorRate` 最低者。因此“训练完成 10 epochs”不代表最终使用的 checkpoint 必须来自第 10 epoch。最终选择不依赖 verification EER/minDCF 或测试分数。

## 运行命令

在项目根目录执行。正式运行需要项目训练依赖及兼容的 PyTorch/CUDA、SpeechBrain；本入口本身不需要 LLM API、Optuna 或 LangGraph。`--dry-run` 只需要基础配置解析依赖，不加载训练/搜索模块，不创建输出目录。

如果尚无独立的验证 pairs，先使用
[`scripts/tools/generate_hpo_validation.py`](../tools/GENERATE_HPO_VALIDATION.md)
从 VoxCeleb1 dev 说话人中确定性生成一次。不要把 `veri_test2.txt`
同时作为验证和测试协议，也不要把覆盖 VoxCeleb1 训练说话人的 VoxCeleb1-H
作为验证协议。

先查看实际参数：

```powershell
python scripts/experiments/run_expert_baseline.py --dry-run
```

训练并验证（将示例数据路径替换为真实路径）：

```powershell
python scripts/experiments/run_expert_baseline.py --config configs/experiments/expert_ecapa.json --data-folder E:/datasets/VoxCeleb --validation-pairs E:/datasets/protocols/hpo_validation.txt --test-pairs E:/datasets/protocols/heldout_test.txt --output-dir E:/experiments/expert_ecapa
```

默认只执行训练 YAML 中的一个 seed。与五组实验对照时，在首次命令追加 `--seeds 11,22,33`，每个种子仍然只使用同一份固定参数，不挑选表现最好的种子。

使用自己的专家配置：

```powershell
python scripts/experiments/run_expert_baseline.py --train-config E:/configs/expert_train.yaml --validation-config E:/configs/expert_verification.yaml --data-folder E:/datasets/VoxCeleb --validation-pairs E:/datasets/protocols/hpo_validation.txt --test-pairs E:/datasets/protocols/heldout_test.txt --seeds 11,22,33 --output-dir E:/experiments/my_expert
```

完成训练和验证后执行最终测试：

```powershell
python scripts/experiments/run_expert_baseline.py --evaluate-test --output-dir E:/experiments/expert_ecapa
```

## 恢复与隔离

```powershell
python scripts/experiments/run_expert_baseline.py --resume --output-dir E:/experiments/expert_ecapa
```

- 训练完成且结果已落盘：不重复训练；验证中断或失败时只补验证。
- 训练中断或失败：显式 `--resume` 在原训练目录重试，SpeechBrain 从可用的自身 checkpoint 恢复。没有 checkpoint 时会重新训练；不保证从精确中断 batch 或随机状态位级复现。
- 已完成的评估不重复；最终测试中断或失败时重新执行 `--evaluate-test`。
- 本入口的显式恢复会重试失败阶段，与五组入口“跳过已提交终态失败运行”的语义不同。所有尝试及其耗时保留在日志中，不能只报告成功尝试的成本。
- 不允许恢复时换参数、种子、输出根目录、源代码或已记录依赖版本；配置、checkpoint 和评估归属均校验。
- 每个 seed 使用独立训练目录和准备缓存；训练排除验证、测试 pairs 两侧所有说话人。`data_prep_seed=0` 固定 train/dev 划分，不受训练种子影响。
- 若 `<data-folder>/noise` 和 `<data-folder>/rir` 已包含 `.wav`，入口直接只读复用本地增强音频，只在实验的 `assets/` 中生成 CSV；对应目录缺失时才使用 YAML 的 URL 下载。启动后不得增删或替换这些本地音频，否则恢复时资产清单校验会拒绝继续。
- 数据路径、排除清单、缓存/日志/checkpoint 目录、准备开关属于隔离协议，会覆盖 YAML 中对应字段；`voxceleb_source` 禁用，输入音频须提前解压。增强资源准备到实验目录，训练时只读，不写入原数据目录。
- 与五组入口共享真实 SpeechBrain 执行锁，避免两个入口同时抢设备。无法阻止其他程序占用 GPU，正式测时仍需自行隔离硬件负载。
- 当前沿用 cosine verification、`score_norm=none` 协议，不自动构造额外 cohort 或做 PLDA 训练。输入数据版本需自行冻结，脚本不逐文件哈希整个音频库。

## 输出与解释

| 文件 | 内容 |
| --- | --- |
| `expert_plan.json` | 固定配方、训练轮数、种子、输入/源码哈希、依赖版本与指标约定 |
| `assets.json` | 共享增强资源的准备耗时和注释哈希 |
| `runs/expert_seed_*/run_inputs.json` | 此次重复的配置快照哈希 |
| `runs/expert_seed_*/training/` | 训练日志、模型及恢复 checkpoint |
| `runs/expert_seed_*/training.json` | 训练完成状态、实际最终 epoch、锁定 checkpoint |
| `runs/expert_seed_*/validation.json` / `test.json` | 绑定到该 checkpoint 的 EER、minDCF 与评估配置哈希 |
| `runs/expert_seed_*/attempts.jsonl` | 各阶段每次尝试的开始、结束、耗时及错误 |
| `results.csv/json`、`summary.json` | 验证结果，所有种子的成功/失败/未完成情况及成功子集均值、标准差 |
| `heldout_results.csv/json`、`heldout_summary.json` | 最终测试结果，不回流训练或选优 |

EER 使用 `[0,1]` 比例值，不乘 100。minDCF 沿用 SpeechBrain 后端返回值乘 100 的约定，固定 `c_miss=c_fa=1, p_target=0.01`；不能混入其他先验、归一化方式或不同计算路径的数值。两项指标必须有效，缺失、NaN/Infinity 不会记成成功。

脚本退出码：0 全部成功；1 存在失败/未完成；2 配置或完整性检查错误；130 用户中断。硬终止可能遗漏耗时，结果用 `cost_may_be_incomplete` 标注。恢复会累计已经记录的失败/中断成本。

自动回归（假训练器，不调用 GPU 或 LLM）：

```powershell
python -m pytest tests/integration/test_expert_baseline.py tests/integration/test_hpo_benchmark.py tests/unit/test_hpo_benchmark.py tests/unit/test_benchmark_voxceleb_split.py tests/unit/test_experiment_tracker.py -q
```

2026-08-28 验证：上述联合回归 70 项全部通过，其中专家基线新增测试 25 项；静态检查通过。未执行真实 GPU 训练或在线 LLM 调用。正式实验前仍需在目标训练环境和真实数据上进行小规模通路验证。

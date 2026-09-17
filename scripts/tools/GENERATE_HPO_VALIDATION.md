# 生成独立的 HPO 验证协议

`generate_hpo_validation.py` 从已解压的 VoxCeleb1 中按说话人固定留出验证集，生成可直接交给专家基线和五组 HPO 实验使用的 `hpo_validation.txt`。脚本只扫描路径并写协议文件，不读取或移动音频，不生成训练 CSV，也不会启动训练。

## 生成命令

在项目根目录运行：

```bash
python scripts/tools/generate_hpo_validation.py \
  --data-folder datasets/voxceleb1 \
  --test-pairs /home/lixh26/agent/SR-agent/resule_test1/veri_test2.txt \
  --output-dir /home/lixh26/agent/SR-agent/resule_test1/protocols \
  --validation-speakers 100 \
  --positive-pairs 10000 \
  --negative-pairs 10000 \
  --max-utterances-per-speaker 20 \
  --seed 2025
```

默认只选择 `id1` 开头的 VoxCeleb1 说话人，排除 `veri_test2.txt` 中全部测试说话人。正样本仅使用同一说话人的不同会话，负样本使用不同说话人。每名验证说话人都会至少出现在一条正样本中，因此现有训练数据准备能从训练和内部 dev CSV 中排除所有验证说话人。

输出目录必须是新的或不包含同名输出。脚本不会覆盖已经用于实验的协议：

- `hpo_validation.txt`：验证 EER/minDCF 的 trial pairs；
- `validation_speakers.txt`：固定的验证说话人；
- `validation_utterances.txt`：生成时允许使用的验证语音池；
- `hpo_validation_manifest.json`：参数、数量、数据清单哈希、测试协议哈希和零重叠校验结果。

## 用于专家基线

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/experiments/run_expert_baseline.py \
  --config configs/experiments/expert_ecapa.json \
  --data-folder datasets/voxceleb1 \
  --validation-pairs /home/lixh26/agent/SR-agent/resule_test1/protocols/hpo_validation.txt \
  --test-pairs /home/lixh26/agent/SR-agent/resule_test1/veri_test2.txt \
  --output-dir /home/lixh26/agent/SR-agent/resule_test1/baseline
```

基线和五组实验入口会冻结输入文件，并将 validation/test pairs 合并为训练排除清单；不需要手工修改底层 SpeechBrain 配方。HPO 应在 rung 边界使用该验证协议计算 EER/minDCF，官方测试协议只在参数和模型锁定后执行。

同一个实验系列必须复用同一份生成结果。若修改 seed、说话人数、pair 数量或音频数据，应使用新的协议目录和新的实验输出目录，不能覆盖后继续断点恢复。

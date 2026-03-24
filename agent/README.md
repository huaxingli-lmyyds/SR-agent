# ECAPA-TDNN 超参数优化智能体系统

基于 LangChain 的智能体系统，用于自动优化 ECAPA-TDNN 声纹识别模型的超参数。

## 🎯 功能特性

- **自然语言交互**: 使用自然语言指令进行超参数调整
- **智能优化**: 基于经验自动选择和调整超参数
- **安全机制**: 配置自动备份，防止实验失败
- **实时监控**: 查看训练日志和评估结果
- **灵活配置**: 支持多种优化策略和参数调整

## 📋 系统要求

- Python 3.8+
- LangChain
- PyYAML
- ruamel.yaml
- ChatGLM API 访问权限

## 🚀 安装依赖

```bash
pip install langchain langchain-community pyyaml ruamel.yaml python-dotenv
```

## ⚙️ 环境配置

1. 创建 `.env` 文件在项目根目录:

```env
ZHIPUAI_API_KEY=your_api_key_here
ZHIU_API_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
```

2. 确保配置文件路径正确:
   - 配置文件: `../configs/train_ecapa_tdnn.yaml`
   - 训练脚本: `../recipes/voxceleb/train_speaker_embeddings.py`
   - 评估脚本: `../recipes/voxceleb/speaker_verification_cosine.py`

## 🎮 使用方法

### 启动智能体

```bash
cd agent
python hpo_agent.py
```

### 交互式指令示例

#### 1. 读取配置信息
```
读取当前配置
```

#### 2. 备份配置
```
备份当前配置
```

#### 3. 调整学习率
```
将学习率调整为 0.01
```

#### 4. 调整批次大小
```
将批次大小调整为 64
```

#### 5. 批量调整参数
```
将学习率设为 0.01，批次大小设为 32，训练轮数设为 20
```

#### 6. 训练模型
```
训练模型
```

#### 7. 评估性能
```
评估模型性能
```

#### 8. 查看训练日志
```
查看训练日志
```

#### 9. 智能优化
```
分析当前配置并给出优化建议
```

```
优化超参数以提升模型性能
```

#### 10. 完整工作流程
```
先备份配置，然后调整学习率为 0.015，训练模型，最后评估性能
```

## 🔧 可用工具

| 工具名称 | 功能描述 |
|---------|---------|
| `modify_config` | 修改 YAML 配置文件中的超参数 |
| `run_training` | 运行 ECAPA-TDNN 模型训练 |
| `run_evaluation` | 评估训练后的模型性能 |
| `read_config` | 读取当前配置文件内容 |
| `get_training_logs` | 获取训练日志 |
| `backup_config` | 备份当前配置文件 |

## 📊 关键超参数说明

### 训练参数
- **lr**: 学习率 (默认 0.02, 推荐 0.001-0.1)
- **batch_size**: 批次大小 (默认 32, 推荐 16-64)
- **number_of_epochs**: 训练轮数 (默认 10, 推荐 10-30)
- **step_size**: 学习率调度步长 (默认 65000)

### 模型结构参数
- **embedding_model.channels**: 各层通道数
- **embedding_model.kernel_sizes**: 卷积核大小
- **embedding_model.dilations**: 膨胀率
- **embedding_model.attention_channels**: 注意力通道数
- **embedding_model.lin_neurons**: 线性层神经元数

### 损失函数参数
- **compute_cost.loss_fn.margin**: AdditiveAngularMargin 边界值
- **compute_cost.loss_fn.scale**: 缩放因子

### 优化器参数
- **opt_class.lr**: 优化器学习率
- **opt_class.weight_decay**: 权重衰减

## 💡 优化策略

### 1. 渐进式优化
- 先固定其他参数，单独调优学习率
- 然后调优批次大小
- 最后调优模型结构和损失函数参数

### 2. 网格搜索
- 对关键参数进行小范围的网格搜索
- 例如学习率: [0.01, 0.02, 0.03]

### 3. 经验法则
- 学习率过大可能导致训练不稳定
- 批次大小过小可能导致梯度估计不准确
- 权重衰减过大可能导致欠拟合

## 📝 工作流程

1. **分析**: 读取当前配置，了解基线性能
2. **规划**: 制定优化策略，确定优先调整的参数
3. **执行**: 修改配置 → 训练 → 评估 → 分析结果
4. **迭代**: 根据评估结果调整参数，重复执行
5. **总结**: 找到最优配置并给出优化建议

## ⚠️ 注意事项

1. 每次调整参数时建议一次只调整 1-2 个参数
2. 重要实验前先备份配置
3. 关注训练损失和验证准确率的变化趋势
4. 训练过程可能需要较长时间，请耐心等待
5. 确保有足够的计算资源（GPU/CPU）
6. 确保数据集路径配置正确

## 🐛 故障排除

### 问题: 智能体无法调用工具
- 检查 API 密钥是否正确
- 确认网络连接正常
- 查看错误日志

### 问题: 配置文件修改失败
- 检查文件路径是否正确
- 确认文件有写入权限
- 检查 JSON 格式是否正确

### 问题: 训练脚本执行失败
- 确认训练脚本路径正确
- 检查数据集是否存在
- 查看训练脚本输出日志

## 📚 参考资料

- [ECAPA-TDNN 论文](https://arxiv.org/abs/2005.07143)
- [SpeechBrain 文档](https://speechbrain.readthedocs.io/)
- [LangChain 文档](https://python.langchain.com/)

## 📧 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。

## 📄 许可证

MIT License
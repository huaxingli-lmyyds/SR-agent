# version_plan1.1 实现说明

实现位置：project/，原项目源码保留。SOURCE_MANIFEST.json 是改动前 SHA256 基线；最终变更清单见 validation/change_manifest.json。

## 编排

- AgentRegistration.requires（默认空）声明依赖。缺失/循环依赖在咨询前报错；稳定拓扑优先保留原数据→HPO顺序。
- 默认 HPO 注册明确依赖数据；上游失败不实例化依赖智能体，阻塞原因沿链传递，独立节点继续。
- HPO 恢复仍跳过数据处理，由已有恢复逻辑核验保存协议。新增其他依赖不会被这个例外放行。
- 图递归上限按节点数增长，新增 15 个节点的契约测试通过。当前仍是一轮串行 DAG，不支持任意循环或动态增删角色。
- 无效协调建议降级为确定性执行；工作流异常写入管理实验失败状态。紧凑记录保留 blocked_by、数据咨询状态与反馈状态。

## 数据处理

- hpo_feedback 从请求进入 workflow，再进入建议提示；校验 dataset_uri、study_id、trial_count、evaluation_role=hpo_validation。非匹配和 final_test 证据拒绝。只接受有限诊断与有限数值，禁止把任意原始对象灌入记录。
- 提示补充 probe_limit、probe_limited、signal_probe_count；缺失为 null，不能按全量质量结论理解。
- consultation_mode=always 保留旧调用方式；evidence 只在存在建议动作的问题、显式操作或合格 HPO 反馈时咨询。状态为 disabled/skipped/completed/failed，失败后仍执行已有确定性计划和质量门。
- QualityPolicy 新增可选 max_operation_affected_fraction、max_operation_output_bytes：对每个变更操作预览检查，缺少完整扫描证明会阻止该操作。上限默认 null。字节数是预览估算门槛，不是磁盘配额；比例为每操作上限，不是整条流水线累计上限。
- 被拒绝的音频过滤释放预览选择缓存；保留源数据，不发布失败结果。
- 数据版本 JSON 以临时文件+flush/fsync+replace 发布，替换前失败不损坏上一份版本。此处没有宣称跨进程锁或全流程事务恢复。

## 配置与反馈示例

在训练 YAML 中加入（也可通过 request.context.data_processing 覆盖）：

```yaml
data_processing:
  consultation_mode: evidence
  quality_policy:
    max_operation_affected_fraction: 0.2
    max_operation_output_bytes: 10737418240
```

20% 和 10 GiB 只是示例，需按允许的数据损失和空间约束选择；开启限制后无完整预览的处理器会被阻断，先做小数据验收。

HPO 反馈由调用方显式提供给下一次数据任务：

```json
{
  "dataset_uri": "实际参与该 Study 的数据目录",
  "study_id": "来源 Study ID",
  "evaluation_role": "hpo_validation",
  "trial_count": 3,
  "metrics": {"eer": 0.1},
  "diagnostics": ["由验证数据支持的简短观察，不能写成已证实的因果结论"]
}
```

本轮未自动生成此反馈、未自动回跳重训。URI 校验也不等于文件内容指纹校验；同路径数据被替换仍需下一阶段的版本摘要约束，见 DATA_LOOP_REVIEW.md。

## 验证范围

新增依赖、失败传播、反馈拒绝、咨询机会、影响边界、真实小 WAV 过滤拒绝、版本发布故障、紧凑记录测试；全回归结果最终写入 VALIDATION.md。真实 3090、DDP、多 GPU 和识别效果实验另行验收，不由这些测试替代。

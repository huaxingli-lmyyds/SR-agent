# 架构基线与实施契约

核查日期：2026-11-11。源码基线见 SOURCE_MANIFEST.json；本轮实现和测试均在 project/，不覆盖主项目已有修改。

## 执行顺序与职责

入口配置 → 编排计划/依赖校验 → 数据检查、建议、确定性审批、处理、复检、版本发布 → 数据交接验证 → HPO Campaign/Study → 批次训练/验证 → HPO 决策 → 搜索或确认报告。

|模块|当前文件（相对 project/）|应负责|不应负责|独立验收|
|---|---|---|---|---|
|入口/配置|main.py、agent/utils|解析、恢复约束、注入依赖|修改已冻结实验协议|CLI/配置测试|
|编排|agent/agents/coordination.py、orchestration_workflow.py|注册、依赖排序、派发、失败传播、完成判定|绕过质量门或直接改训练参数|假智能体契约/故障测试|
|数据智能体|agent/agents/data_processing_agent.py、agent/data_processing/workflow.py|限定证据建议与流程控制|把 EER 差直接归因为数据问题|反馈作用域、建议异常测试|
|数据服务/处理器|agent/data_processing/service.py、registry.py、audio.py|参数审批、影响预览、物化、复检、发布|由 LLM 自行执行任意文件操作|小音频/CSV 实例验证|
|交接|agent/data_processing/handoff.py|consumer_uri、质量状态与产物可用性检查|让未就绪派生产物进入训练|交接拒绝测试|
|HPO|agent/hpo、agent/agents/hpo_agent.py|采样/晋级/咨询机会/建议审批/确认|混合不兼容数据和预算的成绩|既有 HPO 回归|
|执行与账本|agent/hpo 的执行阶段、stage_pool、resources/costs|GPU 分配、隔离 worker、Controller 提交、成本|Worker 写共享 Trial/账本|重放/重复提交/资源测试|
|后端/任务/模型|agent/runners、tasks、models、recipes|训练与指标协议、恢复能力声明|承担全局调度|假后端契约 + 真实后端单独烟测|
|记录/记忆|agent/experiments、memory、utils|证据与产物索引、可追溯记录|将历史建议视作已验证因果结论|序列化/兼容性测试|

## 已有能力与缺口

已有 HPO 账本投影、咨询状态机、top-k checkpoint、多阶段预算、管理式 worker、续训和确认相关实现；不能把代码存在解释为真实多 GPU 或性能收益已验收。本轮不重写这些模块。

编排目前偏好数据→HPO，但扩展智能体按名字排序，无显式依赖；上游失败后仍派发下游。咨询结果保存但不决定拓扑，必须标识其“仅建议”角色。数据流程已形成质量闭环，但 HPO 反馈只存入记录，未送入策略建议；预览结果未用作执行前限制。数据变更对最终识别指标的因果验证尚未闭环。

## 本轮实现约束

1. 注册项增加可选 requires，稳定拓扑排序；缺失/循环依赖启动前失败。默认 HPO 依赖数据；恢复 HPO 使用原有恢复入口，不重新处理数据，仍由 HPO 恢复逻辑验证保存协议。
2. 依赖失败的节点不实例化，记录 failed/blocked_by；独立节点可继续。仍串行执行，一个节点一个首选 action，不暗示支持动态 DAG 或并行智能体。
3. HPO→数据反馈限定结构、来源 Study、数据 URI、validation 角色、样本数及有限诊断；不匹配/测试集反馈只记录拒绝原因。反馈是候选假设，不覆盖确定性处理审批。
4. 预览影响限制可选启用，未知影响不能视作安全；复检仍是发布前必要条件。旧配置默认行为保留。
5. 新记录保持小型摘要：依赖阻塞、反馈接纳理由、画像覆盖率、咨询机会/状态、影响拒绝。完整数据版本和文件证据留在产物文件。

## 扩展契约

新增 agent：实现 execute_task(AgentTaskRequest)→AgentTaskResult，登记 actions/requires/factory；不改调度器。角色变更需要更改注册项并跑拓扑、失败传播、交接契约测试。新增处理器：validate/preview/execute + 注册 + 作用域与输出 consumer_ready 测试。新增后端：能力声明、协议签名、训练/评价与恢复测试。新增任务必须提供评估协议和数据适配，不能只更换提示词。

实验完成 success 只表示流程完成；质量通过只表示符合配置检查；只有同协议对照和重复证据才能证明效果提升。这三个状态必须分开报告。

# 连续任务

|任务|状态|输入|交付物|依赖|验收标准|
|---|---|---|---|---|---|
|T1 架构与研究|完成|当前源码/测试/历史需求|ARCHITECTURE.md、RESEARCH.md、SOURCE_MANIFEST.json、隔离源码|无|模块职责/依赖/扩展契约可定位；数据集与指标有一手来源；不虚构效果|
|T2 编排与数据处理优化|完成|T1设计、隔离源码|project/代码及回归测试、CHANGELOG.md|T1完成|依赖排序与失败隔离、反馈传递及质量影响约束有测试；既有回归通过|
|T3 数据闭环审查|完成|T2代码及测试结果|DATA_LOOP_REVIEW.md、EXTENSIONS.md|T2完成|明确已闭合/未闭合反馈路径；每项扩展含触发、动作、边界、指标|
|T4 总体实验设计|完成|T1指标、T3假设|EXPERIMENTS.md、实验矩阵、预览/校验脚本、最终验收记录|T3完成|覆盖功能、恢复、消融、公平预算、质量及泛化；预览可执行；不将计划当作实测|

验收索引：T1→ARCHITECTURE.md/RESEARCH.md；T2→VALIDATION.md/verified_test_inventory.json；T3→validation/loop_audit.json；T4→validation/preview-tests.xml、preflight.txt、comparison-preview.json。T4完成指实验设计与入口，不表示昂贵对照实验已经执行。

# 验收结果

日期：2026-11-11。使用主项目已有 .venv-hpo/Python，未改全局 CUDA 栈。测试对象为 project/ 隔离源码，测试临时目录与报告均在本交付目录。

## 代码验证

- 最终全量回归：764 passed、2 skipped、1 failed，报告 validation/regression-final.xml。唯一失败是基准测试运行期间源码被本轮编辑，冻结来源检查正确报出 data_processing_agent.py 发生变化。
- 停止源码编辑后复验该路径及生命周期测试：4 passed，报告 validation/stable-source.xml。
- 最终修改的受影响路径完整复验：89 passed，报告 validation/lifecycle-final.xml。包含真实数据/编排/记录链（训练后端替身）和真实小 WAV 过滤边界。
- 按 testcase 标识合并以上有效证据，覆盖 **766 passed、0 failed、2 skipped**，详见 validation/verified_test_inventory.json。这是合并验证结果，不是声称某次命令一次全绿。
- 两项跳过：真实 LLM smoke 需要显式配置；POSIX 共享文件锁测试在 Windows 跳过。68条全量告警保留在日志中，主要涉及既有依赖/实验API，不掩盖为零告警。
- 变更文件 F/E9 静态检查通过（validation/lint.txt）；使用已有 Ruff 可执行文件只读检查，没有安装或修改全局依赖。

## 设计与入口验证

- T1：251个基线文件记录 SHA256（初始248，加demo三文件）；架构职责、扩展边界和官方研究来源均有文档。
- T3：五条关键闭环路径逐项映射至通过的测试，validation/loop_audit.json。效果闭环、数据阶段恢复仍明确标为未实现。
- T4：实验预览测试 **8 passed**（validation/preview-tests.xml）；预览工具静态检查通过。
- 缺关键输入时 --check-inputs 返回预期退出码2，validation/preflight.txt；未调用训练或外部LLM。
- 既有三组 HPO 比较脚本使用生成请求成功预览；验证 legacy/p0/full 三组存在且未创建运行目录，validation/comparison-preview.json。

首次定向测试的父临时目录缺失、测试参数错误及第一次全量的demo复制遗漏均已修复；原失败报告保留，不能作为最终通过证据。

## 不在本轮实测结论中的事项

没有全量数据EER/minDCF对照结果，没有真实LLM建议质量评分，没有本轮3090/DDP/多GPU吞吐和故障恢复验收。CPU小模型/替身后端测试只验证对应状态与接口行为，不替代这些硬件或统计实验。

数据反馈目前由调用方显式提供，校验URI/Study/validation角色而非不可变内容摘要；自动HPO反馈导出、数据效果试验采纳状态机、数据阶段恢复和全系统成本账本属于后续扩展。

## 复验

```powershell
# 当前目录为 version_plan1.1
.\validate.ps1 -Focused
.\validate.ps1
& '..\.venv-hpo\Scripts\python.exe' -m pytest 'experiments\test_preview.py' -q --basetemp=validation/manual-preview-tmp -o cache_dir=validation/pytest-cache
```

运行期间保持 project/ 源码不变，避免再次触发基准的源码冻结保护。完整回归在本机约4分钟，机器负载与依赖会影响耗时。

交付复核：validate.ps1 -Focused 实际运行 76 passed；implementation.patch 在主项目运行 git apply --check 通过（只检查，未应用）；251份原始基线文件 SHA256 保持一致。最终修改/新增13个代码、提示和测试文件，见 change_manifest.json。

## 本地提交前复验（2026-11-17）

- `validate.ps1 -Focused`：76 passed，3 条 Optuna `constant_liar` 实验性接口告警。
- `experiments/test_preview.py`：8 passed；本次未重跑全量回归或启动训练。
- 归档前核对 251 份主项目基线和 255 份隔离副本的 SHA256，均与对应清单一致；`git apply --check implementation.patch` 通过，未应用补丁。
- 本次仅补充根目录 `.gitignore` 和归档说明，独立副本源码保持一致。原始报告、缓存、临时目录和生成请求留在本地，版本控制范围见 START_HERE.md。
- 暂存区空白检查提示 440 处未修改基线文件的既有空白问题，以及 24 处补丁上下文空行；未发现新增实现或归档说明的空白问题，本次保留原始源码与补丁格式。

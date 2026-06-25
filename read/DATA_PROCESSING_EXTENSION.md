# 数据处理智能体扩展说明

## 当前能力

数据处理智能体已经从“修改配置并调用 VoxCeleb 准备函数”扩展为通用生命周期：

```text
InspectDataset
  -> BuildDataProcessingPlan
  -> 数据类型专用处理工具
  -> ExecuteDataProcessingPlan
  -> PublishDatasetVersion
```

原始数据不会被覆盖。发布操作只生成数据版本与血缘元数据。

## 通用协议

`agent/data_processing/contracts.py` 定义：

- `DatasetSpec`：描述数据集标识、类型、格式、来源、任务和切分。
- `DataProfile`：保存样本数量、结构、分布、质量指标和问题。
- `DataProcessingPlan`：保存操作、原因、预期效果和验证规则。
- `DataOperationResult`：保存操作前后指标、产物与错误。
- `DatasetVersion`：保存父版本、操作历史、质量指标和产物。

这些结构不包含固定的音频字段。音频、图像、文本或表格特有信息应进入 `extensions`。

## 处理器注册

`agent/data_processing/registry.py` 提供处理器注册机制。处理器只负责具体执行，智能体负责选择和编排。

```python
class ImageResizeProcessor:
    operation_name = "resize_images"
    supported_data_types = {"image"}

    def validate(self, dataset, parameters):
        ...

    def execute(self, dataset, parameters):
        ...


register_processor(ImageResizeProcessor())
```

当前注册了模型无关的 `validate_dataset` 处理器。后续可分别增加：

- 音频：重采样、声道转换、静音检测、音频损坏检查。
- 图像：尺寸统一、损坏检查、颜色空间转换。
- 文本：编码修正、去重、敏感信息检查。
- 表格：模式校验、缺失值处理、异常值检查。

## 实验记录

数据生命周期信息保存到：

```json
{
  "metrics": {
    "quality_before": {},
    "quality_after": {}
  },
  "extensions": {
    "data_lifecycle": {
      "dataset": {},
      "profile_before": {},
      "plan": {},
      "operation_results": [],
      "published_version": {}
    }
  },
  "artifacts": [
    {
      "type": "dataset_version",
      "path": "..."
    }
  ]
}
```

## 新增数据类型

新增数据类型时，优先添加处理器并注册，不需要修改数据处理智能体。只有数据来源识别方式发生变化时，才需要扩展 `infer_dataset_spec`。

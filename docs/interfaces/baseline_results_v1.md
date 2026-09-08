# 基线结果接口：v1

## 结果文件

- 指标摘要：`reports/baselines/synthetic_v1_metrics.json`
- 测试预测：`reports/baselines/synthetic_v1_test_predictions.csv`
- 人工可读记录：`docs/experiments/baseline_synthetic_v1.md`

结果必须绑定 `data_version`、配置 SHA-256、随机种子和运行环境。不同数据、切分、特征、预处理或模型参数不得覆盖同一版本结果。

## JSON 摘要格式

```json
{
  "data_version": "synthetic_v1",
  "config_sha256": "<sha256>",
  "seed": 20260808,
  "environment": {
    "python": "<version>",
    "scikit_learn": "<version>"
  },
  "models": {
    "<model_id>": {
      "validation": {"n": 0, "threshold": 0.0},
      "test": {"n": 0, "threshold": 0.0}
    }
  }
}
```

每个集合的指标至少包含样本量、阳性比例、阈值、AUROC、AUPRC、F1、灵敏度、特异度、Brier score 以及 TP/FP/TN/FN。阈值只能由验证集选择并冻结到测试集，不能使用测试集调参。

## 预测 CSV 格式

`synthetic_v1_test_predictions.csv` 必须包含以下列：

| 列名 | 说明 |
| --- | --- |
| `participant_id` | 与特征表和切分清单关联的去标识化 ID |
| `model` | 模型版本标识 |
| `split` | 本文件固定为 `test` |
| `label` | 仅用于评估的真实标签 |
| `probability` | 模型输出的正类概率 |
| `threshold` | 从验证集选择并冻结的阈值 |
| `prediction` | 按阈值转换后的二分类结果 |

预测文件不得包含姓名、联系方式、原始日期或其他身份信息。正式数据接入时，应重新确认数据使用协议和输出共享范围。

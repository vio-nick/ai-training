# GSTRIDE 跌倒者识别接口契约：v1

## 输入表

处理后特征表 `feature_table_gstride_fall_v1.csv` 每行对应一名参与者，列为：

| 列 | 类型 | 允许值/单位 | 用途 |
| --- | --- | --- | --- |
| `participant_id` | string | 非空且唯一 | 关联与追踪，不进入模型 |
| `faller_last_year` | integer | `0` 或 `1` | 监督学习标签，不作为输入 |
| `step_speed_m_s` | float | m/s | 模型输入 |
| `cadence_strides_per_min` | float | strides/min | 模型输入 |
| `stride_length_m` | float | m | 模型输入 |
| `double_support_pct` | float | % | 模型输入 |
| `swing_to_stance_ratio` | float | 无量纲 | 模型输入 |
| `stride_time_cv_pct` | float | % | 模型输入 |

## 派生公式

- `double_support_pct = Load Avg + Push Avg`
- `swing_to_stance_ratio = Swing Avg / (Load Avg + Foot Flat Avg + Push Avg)`
- `stride_time_cv_pct = 100 × Stride Time Std / Stride Time Avg`

其余三项直接采用 GSTRIDE register 中的 Step Speed Avg、Cadence Avg 和 Stride Length Avg。数值字段按 CP1252 源文件读取，逗号小数点转为点；空字符串、`-` 和 `Incapable` 视为缺失。

## 推理输出与展示分级

推理接口保留 `predicted_probability`（范围 0--1 的原始模型分数），并输出 `predicted_probability_percent = predicted_probability * 100` 供 0%--100% 展示。展示分级使用原始概率 `p`，不应使用已四舍五入的百分数：

| 条件 | `risk_level` | `risk_level_display` | 展示含义 |
| --- | --- | --- | --- |
| `p < 0.3` | `low` | 低风险 | 低于 30% |
| `0.3 <= p <= 0.7` | `medium` | 中风险 | 30% 至 70%，含边界 |
| `p > 0.7` | `high` | 高风险 | 高于 70% |

该表是当前原型的固定展示分段，不是临床验证的风险分层。`predicted_label` 是按当前封版模型在验证集选定的 `decision_threshold` 生成的技术二元标记；它与上述低/中/高展示分级相互独立，不得互相替代。

## 训练与推理约束

训练流程先按 `participant_id` 合并切分表，再仅用训练集拟合中位数插补和（逻辑回归时）标准化。验证集用于选择最大平衡准确率阈值；测试集只产生最终概率、标签和指标。模型输出是 `faller_last_year=1` 的概率，不是未来跌倒概率、骨折概率或诊断结论。

## 六特征贡献输出

为支持前端展示单条记录的横向条形图，推理响应增加以下字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `feature_contributions` | array | 按六项特征契约顺序排列的贡献对象；JSON 接口中保持为数组。 |
| `feature_contribution_method` | string | 固定为 `exact_shapley_probability`。 |
| `feature_contribution_baseline_probability` | number | 六项特征均用缺失值遮蔽后，经已拟合插补器得到的基线正类概率。 |
| `feature_contribution_reconstructed_probability` | number | 基线概率加六项贡献后的重构概率，应与 `predicted_probability` 在浮点误差内一致。 |

每个 `feature_contributions` 对象包含：

```json
{
  "feature": "step_speed_m_s",
  "display_name": "步速",
  "unit": "m/s",
  "value": 0.82,
  "contribution_probability": -0.037,
  "contribution_probability_percent": -3.7,
  "direction": "negative",
  "direction_display": "负贡献"
}
```

`contribution_probability` 是对正类模型概率的加性贡献，正值表示提高模型分数，负值表示降低模型分数。前端可以按 `direction` 映射正负颜色，横轴使用 `contribution_probability_percent`，特征名称使用 `display_name`，并根据绝对值排序。建议优先消费 JSON；CSV 没有嵌套数组类型，评分脚本会将同一数组无损编码到 `feature_contributions_json` 列。

该分解是模型解释，不是特征的因果效应，也不是临床风险变化。基线和贡献均依赖当前封版模型及其预处理器，模型版本变化后必须重新解释和记录。

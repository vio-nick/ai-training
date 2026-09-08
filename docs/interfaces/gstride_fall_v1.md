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

# 特征表接口：v1

## 输入与输出位置

- 原始输入：`data/synthetic/synthetic_v1_raw.csv`
- 标准特征表：`data/processed/feature_table_v1.csv`
- 编码：UTF-8 CSV，首行为列名，逗号分隔，缺失值为空。

## 列顺序

特征表必须严格按照下列顺序输出：

1. `participant_id`
2. `fracture_within_24m`
3. `age_years`
4. `sex_at_birth`
5. `bmi_kg_m2`
6. `prior_fracture`
7. `smoker`
8. `alcohol_use`
9. `falls_last_year`
10. `family_history`
11. `lumbar_spine_t_score`
12. `femoral_neck_t_score`
13. `gait_speed_m_s`
14. `stride_length_m`
15. `plantar_peak_pressure_kpa`

字段含义、类型、单位、缺失编码和模态标记以 [数据字典](../data_dictionary.md) 为准。不得增加未声明的列、重命名列或改变顺序；需要变更时发布新的接口版本。

## 校验规则

- `participant_id` 必须非空且唯一，并与 `data/splits/split_v1.csv` 一一对应。
- 标签必须为整数 `0` 或 `1`，且非空。
- 必填临床字段不得缺失且必须落在数据字典的允许范围。
- 可选模态字段可为空；非空值必须落在允许范围。
- 不得包含标签泄漏字段、直接身份信息、日期或未定义的中间变量。

## 成员 B 的使用边界

成员 B 使用 `participant_id` 对齐预测和切分，不将其输入模型。成员 B 必须在训练前校验列顺序和接口版本；任何临时字段映射必须写入适配器和变更记录。

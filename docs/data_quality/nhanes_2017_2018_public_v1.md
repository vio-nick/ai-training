# 数据质量报告：nhanes_2017_2018_public_v1

本报告描述 CDC/NCHS NHANES 2017–2018 横断面公开数据的工程质量，不代表 24 个月骨折风险性能。

- 合并后样本数：865
- `fracture_history` 标签计数：{0: 681, 1: 184}
- `osteoporosis_diagnosis` 计数：{0: 810, 1: 55}
- 切分计数：{'test': 173, 'train': 519, 'validation': 173}
- 纳入：年龄至少 50 岁、两个探索性标签有效、年龄/性别/BMI 有效，并在四份官方文件中完成 SEQN 合并。
- 缺失 BMD 保留为空，由训练集合拟合的插补器处理。
- NHANES 复杂抽样权重仅保留为元数据，本次探索性模型未使用调查加权。

## 字段缺失

| 字段 | 缺失行数 |
| --- | ---: |
| `participant_id` | 0 |
| `fracture_history` | 0 |
| `osteoporosis_diagnosis` | 0 |
| `age_years` | 0 |
| `sex_at_birth` | 0 |
| `bmi_kg_m2` | 0 |
| `lumbar_spine_bmd_g_cm2` | 245 |
| `pelvis_bmd_g_cm2` | 245 |
| `head_bmd_g_cm2` | 156 |
| `survey_exam_weight` | 0 |
| `survey_stratum` | 0 |
| `survey_psu` | 0 |

## 结论

字段、ID 唯一性、标签编码和切分已通过程序校验。标签没有事件日期和 24 个月观察窗；模型结果只能作为公开横断面探索，不得写成临床验证结论。

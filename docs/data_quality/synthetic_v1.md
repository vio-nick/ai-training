# 数据质量报告：synthetic_v1

## 结论范围

本报告只验证合成开发数据的工程完整性，不代表真实人群分布、临床性能或医学结论。

## 样本与标签

- 样本数：240
- 标签计数：`0`=158，`1`=82
- 切分计数：train=144，validation=48，test=48
- 随机种子：`20260808`

## 缺失计数

| 字段 | 缺失行数 |
| --- | ---: |
| `participant_id` | 0 |
| `fracture_within_24m` | 0 |
| `age_years` | 0 |
| `sex_at_birth` | 0 |
| `bmi_kg_m2` | 0 |
| `prior_fracture` | 0 |
| `smoker` | 0 |
| `alcohol_use` | 0 |
| `falls_last_year` | 0 |
| `family_history` | 0 |
| `lumbar_spine_t_score` | 16 |
| `femoral_neck_t_score` | 12 |
| `gait_speed_m_s` | 31 |
| `stride_length_m` | 26 |
| `plantar_peak_pressure_kpa` | 30 |

## 模态完整行数

| 模态 | 完整行数 |
| --- | ---: |
| `clinical` | 240 |
| `bmd` | 214 |
| `gait` | 190 |
| `plantar_pressure` | 210 |

## 完整性审查

- 必填字段非空、字段顺序和允许范围已通过程序校验。
- `participant_id` 唯一，特征表与切分清单 ID 集合完全一致。
- 切分按标签分层，并按参与者 ID 隔离；预处理器拟合边界记录在切分接口中。
- 数据版本：`synthetic_v1`；正式数据准入、伦理和许可状态仍需另行核验。

## 文件校验值

| 文件 | SHA-256 |
| --- | --- |
| `data/synthetic/synthetic_v1_raw.csv` | `37a8d3a61bf3fc6fc9db5b442cfe179280d371ffeb74bed96e81ea20118782e4` |
| `data/processed/feature_table_v1.csv` | `37a8d3a61bf3fc6fc9db5b442cfe179280d371ffeb74bed96e81ea20118782e4` |
| `data/splits/split_v1.csv` | `0253301fcce0d4617238cf8aab9bb64b61116d4d62a0cd606f306325b48a25c0` |

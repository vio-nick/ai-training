# 数据字典：v1 合成开发契约

## 契约状态

本文件定义 `synthetic_v1` 的开发接口，用于成员 A 与成员 B 并行开发。它不是正式研究方案；使用真实数据前必须由两名成员和项目负责人根据伦理审批、数据使用协议及研究方案共同确认或发布新版本。

## 队列与终点

- 分析单位：每名参与者一条索引记录。
- 样本 ID：`participant_id`，格式为 `SYN-0001`，仅用于合成开发数据。
- 主标签：`fracture_within_24m`。值为 `1` 表示在索引时点后 24 个月内发生骨质疏松性骨折，`0` 表示在相同观察窗内未发生该终点。
- 纳入规则：合成的 50 岁及以上参与者，具有有效的 `participant_id` 与主标签。
- 排除规则：重复 ID、无效标签、超出字段允许范围的记录，或无法解析的必填临床字段。
- 缺失值：数值和分类特征均以 CSV 空值表示；标签、ID 和 `age_years` 不允许缺失。

## 字段定义

| 列名 | 类型 | 单位/允许值 | 模态 | 必填 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `participant_id` | string | `SYN-0001` 格式 | metadata | 是 | 去标识化样本 ID，不可作为模型特征 |
| `fracture_within_24m` | integer | `0` 或 `1` | label | 是 | 24 个月骨折终点 |
| `age_years` | integer | 50-95，岁 | clinical | 是 | 索引时点年龄 |
| `sex_at_birth` | category | `female`、`male` | clinical | 是 | 合成开发字段，不用于性别认同推断 |
| `bmi_kg_m2` | float | 15.0-45.0，kg/m2 | clinical | 是 | 体质指数 |
| `prior_fracture` | integer | `0` 或 `1` | clinical | 是 | 索引前既往骨折史 |
| `smoker` | integer | `0` 或 `1` | clinical | 是 | 当前或既往吸烟标记 |
| `alcohol_use` | integer | `0` 或 `1` | clinical | 是 | 风险性饮酒标记 |
| `falls_last_year` | integer | 0-10，次 | clinical | 是 | 索引前一年跌倒次数 |
| `family_history` | integer | `0` 或 `1` | clinical | 是 | 一级亲属骨质疏松/骨折史标记 |
| `lumbar_spine_t_score` | float | -5.0 至 2.0 | bmd | 否 | 腰椎骨密度 T 值 |
| `femoral_neck_t_score` | float | -5.0 至 2.0 | bmd | 否 | 股骨颈骨密度 T 值 |
| `gait_speed_m_s` | float | 0.2-2.0，m/s | gait | 否 | 步速 |
| `stride_length_m` | float | 0.2-2.0，m | gait | 否 | 步长 |
| `plantar_peak_pressure_kpa` | float | 50-900，kPa | plantar_pressure | 否 | 足底峰值压力 |

## 模型输入和泄漏边界

- `participant_id` 只用于关联特征表、切分清单和预测输出，禁止进入模型输入。
- `fracture_within_24m` 只作为监督标签，禁止进入特征工程、插补器、编码器、标准化器或推理输入。
- 所有预处理器只能在训练集拟合；验证集和测试集只能调用已拟合的变换。
- `falls_last_year` 是辅助临床特征，不得替代主标签。
- 对正式数据，预测时间窗内才可获知的信息不得用作索引时点特征。

## 版本规则

兼容性变更可更新本文件的补充说明。不兼容变更，包括标签含义、单位、缺失编码、字段删除、特征时间边界或切分规则变化，必须发布 `v2` 并提供适配说明。

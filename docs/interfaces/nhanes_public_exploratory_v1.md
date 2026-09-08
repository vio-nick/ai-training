# NHANES 探索性特征接口：v1

## 文件

- 特征表：`data/public/nhanes_2017_2018/processed/feature_table_nhanes_exploratory_v1.csv`
- 切分清单：`data/public/nhanes_2017_2018/processed/split_nhanes_exploratory_v1.csv`
- 构建脚本：`scripts/build_nhanes_exploratory.py`
- 训练脚本：`scripts/run_nhanes_baseline.py`

## 字段

| 列名 | 来源变量 | 类型/单位 | 用途 |
| --- | --- | --- | --- |
| `participant_id` | `SEQN` | string | 关联 ID，不进入模型 |
| `fracture_history` | `OSQ080` | 0/1 | 探索性标签；医生曾告知其他骨折 |
| `osteoporosis_diagnosis` | `OSQ060` | 0/1 | 描述字段；医生曾告知骨质疏松/脆骨，不作为本次标签 |
| `age_years` | `RIDAGEYR` | 岁 | 模型特征 |
| `sex_at_birth` | `RIAGENDR` | `female`/`male` | 模型特征 |
| `bmi_kg_m2` | `BMXBMI` | kg/m2 | 模型特征 |
| `lumbar_spine_bmd_g_cm2` | `DXXLSBMD` | g/cm2 | 模型特征，可缺失 |
| `pelvis_bmd_g_cm2` | `DXXPEBMD` | g/cm2 | 模型特征，可缺失 |
| `head_bmd_g_cm2` | `DXXHEBMD` | g/cm2 | 模型特征，可缺失 |
| `survey_exam_weight` | `WTMEC2YR` | 权重 | 元数据，不进入本次模型 |
| `survey_stratum` | `SDMVSTRA` | code | 元数据，不进入本次模型 |
| `survey_psu` | `SDMVPSU` | code | 元数据，不进入本次模型 |

## 编码与边界

- OSQ 的 `1` 映射为 `1`，`2` 映射为 `0`，`7/9` 和缺失映射为空；不把拒答或不知道当作阴性。
- `RIAGENDR` 的 `1` 映射为 `male`，`2` 映射为 `female`。
- BMD 和 BMI 的原始缺失保留为空；插补、编码和标准化只能在训练集合拟合。
- `fracture_history` 不是 `fracture_within_24m`，不能与合成 v1 或正式随访标签混用。

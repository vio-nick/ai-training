# NHANES 公开数据交接：nhanes_exploratory_v2

## 输入与输出

- 特征表：`data/public/nhanes_2013_2018/processed/feature_table_nhanes_exploratory_v2.csv`
- 切分表：`data/public/nhanes_2013_2018/processed/split_nhanes_exploratory_v2.csv`
- 质量元数据：`data/public/nhanes_2013_2018/processed/nhanes_exploratory_v2_quality.json`
- 配置：`configs/nhanes_exploratory_v2.json`
- 训练入口：`scripts/run_nhanes_baseline_v2.py`

## 特征契约

除 `participant_id`、`fracture_history`、调查权重/分层/PSU 元数据外，模型可使用：

`osteoporosis_diagnosis`, `age_years`, `sex_at_birth`, `bmi_kg_m2`, `lumbar_spine_bmd_g_cm2`, `pelvis_bmd_g_cm2`, `head_bmd_g_cm2`, `smoking_ever`, `smoking_current`, `alcohol_ever`, `vigorous_activity`, `moderate_activity`, `high_blood_pressure`, `diabetes`, `coronary_heart_disease`, `heart_attack`, `stroke`, `survey_cycle`。

问卷拒答/不知道编码在特征表中为空。训练脚本只在训练集拟合插补、缺失指示器、标准化和编码；测试集不能反向影响这些步骤。

## 标签和禁止表述

`fracture_history` 只表示 `OSQ080` 的既往骨折史。交接结果必须写成“NHANES 横断面历史标签探索”，禁止写成“24个月骨折预测”“临床验证”或“筛查效度”。正式24个月终点需要另行发布兼容接口版本。

## 复现与验收

```text
D:\anaconda3\python.exe scripts/build_nhanes_multi_cycle.py
D:\anaconda3\python.exe scripts/run_nhanes_baseline_v2.py
```

验收重点：特征表和切分表 ID 一致；固定种子和配置哈希写入指标 JSON；同时报告固定阈值和验证集选择阈值；消融结果与重复交叉验证结果不能只保留最佳一次运行。

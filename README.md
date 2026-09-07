# 骨质疏松相关传感器原型：GSTRIDE 跌倒者识别

本仓库当前聚焦一个可由公开数据验证的传感器核心原型：使用 GSTRIDE 官方步态参数识别“步态测试前一年内报告过跌倒”的受试者。这里的标签是回顾性跌倒史，因此当前任务是跌倒者识别（retrospective faller identification），不是对未来跌倒的前瞻性预测，也不是骨折风险预测或临床诊断。

## 分阶段目标

- **阶段 1（当前）**：只使用步态/足部传感器可获得的六项 GSTRIDE 参数，完成数据版本、特征管线、可复现实验和轻量级基线模型。
- **阶段 2（后续）**：在医院伦理审批、数据使用授权和去标识化完成后，引入骨密度（BMD）、纵向跌倒/骨折终点及必要临床协变量，评估骨密度融合模型。阶段 1 的结果不能替代阶段 2 的临床终点。

## 当前数据与特征

数据来自 Zenodo 记录 [8003441](https://zenodo.org/records/8003441)，GSTRIDE v1.0，DOI `10.5281/zenodo.8003441`，许可证 CC BY 4.0。仓库仅保存构建当前原型所需的 `Database_register.csv`；没有保存约 1.37 GB 的原始传感器压缩包。

样本包含 163 名参与者：86 人报告测试前一年内发生过跌倒，77 人未报告。标签列 `faller_last_year` 定义为 `1=YES`、`0=NO`。六项固定输入及派生方式如下：

| 特征 | 单位 | 定义 |
| --- | --- | --- |
| `step_speed_m_s` | m/s | GSTRIDE Step Speed Avg |
| `cadence_strides_per_min` | strides/min | GSTRIDE Cadence Avg |
| `stride_length_m` | m | GSTRIDE Stride Length Avg |
| `double_support_pct` | % | `Load Avg + Push Avg`，对应双支撑的起始与末端阶段 |
| `swing_to_stance_ratio` | 无量纲 | `Swing Avg / (Load Avg + Foot Flat Avg + Push Avg)` |
| `stride_time_cv_pct` | % | `100 × Stride Time Std / Stride Time Avg`，作为步态时间变异度 |

模型不使用年龄、性别、BMI、衰弱评分、SPPB、TUG、既往疾病等非传感器字段，以保持“传感器核心原型”的范围。详细字段契约见 [`docs/interfaces/gstride_fall_v1.md`](docs/interfaces/gstride_fall_v1.md)。

## 复现训练

在已安装 `pandas`、`numpy` 和 `scikit-learn` 的 Python 环境中运行：

```powershell
D:\anaconda3\python.exe scripts\build_gstride_fall_dataset.py
D:\anaconda3\python.exe scripts\run_gstride_fall_baseline.py
D:\anaconda3\python.exe -m unittest discover -s tests
```

构建脚本输出：

- `data/public/gstride_v1/processed/feature_table_gstride_fall_v1.csv`
- `data/public/gstride_v1/processed/split_gstride_fall_v1.csv`
- `data/public/gstride_v1/processed/gstride_fall_v1_quality.json`

训练脚本输出：

- `reports/baselines/gstride_fall_v1_metrics.json`
- `reports/baselines/gstride_fall_v1_test_predictions.csv`
- [`docs/experiments/gstride_fall_v1.md`](docs/experiments/gstride_fall_v1.md)

实验使用固定的参与者级 60%/20%/20% 分层切分（训练 98、验证 32、测试 33），预处理只在训练集拟合，阈值只在验证集选择，测试集只用于最终评估；训练集另做 5 折 × 5 次重复交叉验证。当前提供逻辑回归和随机森林两个基线。

最近一次可复现实验（Python 3.13.9、scikit-learn 1.7.2）的测试结果为：逻辑回归 AUROC 0.7868、AUPRC 0.7771；随机森林 AUROC 0.7868、AUPRC 0.8030。固定测试集只有 33 人，结果仅用于原型可行性和管线审计，不应被解释为临床性能或可推广性证据。

## 数据版本、质量与模型卡

- [`docs/data_versions/gstride_fall_v1.md`](docs/data_versions/gstride_fall_v1.md)：来源、版本、哈希与特征映射。
- [`docs/data_quality/gstride_fall_v1.md`](docs/data_quality/gstride_fall_v1.md)：完整性、范围和切分审计。
- [`docs/interfaces/gstride_fall_v1.md`](docs/interfaces/gstride_fall_v1.md)：输入、标签和推理接口契约。
- [`docs/model_cards/gstride_fall_v1.md`](docs/model_cards/gstride_fall_v1.md)：用途、性能、限制和禁止用途。
- [`docs/experiments/gstride_fall_v1_execution_log.md`](docs/experiments/gstride_fall_v1_execution_log.md)：数据、训练、验证与清理工作的执行记录。

仓库中旧的 NHANES、合成数据和骨折纵向材料保留为历史探索或方法参考，不属于当前 GSTRIDE 主模型线；它们不能被当作正式骨折终点或医院验证结果。

## 研究边界

当前数据是单次步态测试与既往一年跌倒史的横断面/回顾性组合，存在回忆偏倚、选择偏倚、设备与场地差异以及小样本不确定性。模型不提供诊断、治疗或骨折风险建议。阶段 2 必须重新定义纵向终点、完成伦理与数据授权、按时间划分医院队列，并进行独立外部验证和校准评估。

## 许可证与引用

GSTRIDE 数据按 Zenodo 页面标示的 CC BY 4.0 使用；再分发、改编或发表时应保留原作者、数据集版本、DOI 和许可证信息。请同时引用 Zenodo 记录 `10.5281/zenodo.8003441`。

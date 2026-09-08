# GSTRIDE 步态跌倒者识别原型

本仓库当前实现基于 GSTRIDE 官方步态参数的传感器核心原型，识别“步态测试前一年内自报发生过跌倒”的参与者。

这是回顾性跌倒者识别研究，不是未来跌倒预测、骨折风险预测、骨质疏松诊断或临床筛查工具。

## 当前范围

阶段 1 使用公开 GSTRIDE v1.0 数据集验证传感器特征管线。模型固定使用六项步态特征：步速、步频、跨步长、双支撑比例、摆动/支撑比、步态时间变异度，字段和单位见 [数据接口](docs/interfaces/gstride_fall_v1.md)。

数据来自 [Zenodo 记录 8003441](https://zenodo.org/records/8003441)，DOI 为 `10.5281/zenodo.8003441`，许可证为 CC BY 4.0。当前版本 `gstride_fall_v1` 有 163 名参与者，标签为测试前一年自报跌倒史（86 阳性、77 阴性）。

阶段 2 只有在获得合规医院数据、伦理审批和纵向结局后，才会建立新数据版本来评估 BMD、临床变量与传感器特征的真实融合。现阶段不报告或暗示 BMD 融合性能。

## 成员 A 与成员 B

- 成员 A：数据构建、质量报告、参与者级 60/20/20 切分、逻辑回归和随机森林基线。
- 成员 B：配置驱动训练/评估、输入版本和字段校验、验证集阈值、校准与阈值影响分析、错误案例、解释性输出、轻量推理，以及早期/晚期/混合融合接口。
- 融合接口只用无标签 fixture 验证输入、权重和缺失模态策略；不是真实多模态训练或性能实验。

固定测试集只有 33 人，性能数字仅用于原型管线审计，不能视为临床性能或外部验证证据。

| 模型 | 测试 AUROC | 测试 AUPRC | 敏感度 | 特异度 | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| 逻辑回归 | 0.7868 | 0.7771 | 0.8824 | 0.6250 | 0.2037 |
| 随机森林 | 0.7868 | 0.8030 | 0.8235 | 0.7500 | 0.1840 |

## 复现

需要 Python 3.11+，并安装 `pandas`、`numpy` 和 `scikit-learn`。

```powershell
D:\anaconda3\python.exe scripts\build_gstride_fall_dataset.py
D:\anaconda3\python.exe scripts\run_gstride_fall_baseline.py
D:\anaconda3\python.exe scripts\run_member_b_analysis.py
D:\anaconda3\python.exe scripts\run_member_b_gstride_workflow.py
D:\anaconda3\python.exe scripts\run_member_b_fusion_fixture.py
D:\anaconda3\python.exe -m unittest discover -s tests
```

成员 B 正式单模态产物位于 `reports/member_b/`。`gstride_fall_v1_member_b_workflow.json` 包含配置哈希、测试集指标、校准、阈值影响、推理示例和融合未运行原因；`member_b_fusion_fixture_v1_smoke.json` 只记录融合接口 smoke test，不含标签或性能指标。

## 主要文档

- [数据与特征接口](docs/interfaces/gstride_fall_v1.md)
- [成员 B 推理与融合接口](docs/interfaces/member_b_inference_fusion_v1.md)
- [GSTRIDE 基线实验](docs/experiments/gstride_fall_v1.md)
- [成员 B 实验与模型比较](docs/experiments/member_b_gstride_fall_v1.md)
- [成员 B 集成审计](docs/experiments/member_b_integration_audit.md)
- [成员 B 执行日志](docs/experiments/member_b_execution_log.md)
- [模型卡](docs/model_cards/gstride_fall_v1.md)
- [IMU 接入与使用说明](docs/deployment/imu_integration_and_use_v1.md)

## 研究边界

标签来自既往自报，样本量小且缺少外部验证与临床协变量。不得将任何模型输出用于诊断、治疗、分诊、骨质疏松判断、骨折风险判断或未来跌倒风险判断。

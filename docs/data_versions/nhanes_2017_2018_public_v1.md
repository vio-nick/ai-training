# 数据版本：nhanes_2017_2018_public_v1

## 来源与用途

- 来源：CDC/NCHS National Health and Nutrition Examination Survey (NHANES) 2017–2018 public-use files。
- 下载日期：2026-08-08。
- 数据页面：<https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Demographics&Cycle=2017-2018>、<https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Questionnaire&Cycle=2017-2018>、<https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&Cycle=2017-2018>。
- 数据使用说明：<https://www.cdc.gov/nchs/policy/data-user-agreement.html>。
- 用途：公开数据探索性建模和管线验证，不作为本项目 24 个月骨折终点的正式验证。

NHANES 是横断面调查，公开文件没有本项目定义的“索引时点后 24 个月骨质疏松性骨折”随访终点。本版本将 `OSQ080` 定义为 `fracture_history` 探索性标签，将 `OSQ060` 保留为 `osteoporosis_diagnosis` 描述字段；二者都不能替代正式的时间窗骨折标签。

## 官方文件与校验值

| 文件 | 官方变量说明 | 官方 XPT 下载 | SHA-256 |
| --- | --- | --- | --- |
| `DEMO_J` | <https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.htm> | <https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.xpt> | `c0b46e0345ea19404928656277c8b0d10b0cca348a9b2fe4fc3c67e8b7ee73ec` |
| `BMX_J` | <https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BMX_J.htm> | <https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BMX_J.xpt> | `8d675e42d8826ac98714b2c3dd4c5138a5e353fb4424f7eff5e6db4a01ce838` |
| `OSQ_J` | <https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/OSQ_J.htm> | <https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/OSQ_J.xpt> | `e278ea3b723456a7760f5ec288e0eac199300089587c1c457eb88a2a1b99e02b` |
| `DXX_J` | <https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DXX_J.htm> | <https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DXX_J.xpt> | `de16420ff3f978f72dbcc6a83c64de041440f29ad1f516f5e1cc5476af994ff1` |

原始文件保存在 `data/public/nhanes_2017_2018/raw/`，处理脚本为 `scripts/build_nhanes_exploratory.py`。原始文件为 CDC public-use 数据，不含项目成员自行收集的身份信息；仍须遵守 CDC 数据使用说明、引用要求和本项目伦理边界。

处理结果为 865 行：`fracture_history=1` 为 184 行、`0` 为 681 行；`osteoporosis_diagnosis=1` 为 55 行、`0` 为 810 行。训练/验证/测试分别为 519/173/173。处理文件校验值为：特征表 `8e3135791b2ef70469bc84ee2ca181f1f1d249c7a91b7e6f32748037c7ea4f4f`，切分清单 `50f182d40b4d7e19726935dae81d4bd3b1c38445303a21b835fd05a53ae19f12`。

## 预处理范围

- 仅保留 `RIDAGEYR >= 50` 且 `SEQN`、性别、BMI 和两个探索性标签有效的参与者；`SEQN` 重命名为仓库内的 `participant_id`。
- 以 `SEQN` 合并 DEMO、BMX、OSQ 和 DXX；不合并其他非必要文件。
- BMD 使用 `DXXLSBMD`（腰椎）、`DXXPEBMD`（骨盆）和 `DXXHEBMD`（头部）原始测量值，单位为 g/cm2；缺失值交由训练集拟合的插补器处理。
- NHANES 复杂抽样权重、分层和聚类变量保留为元数据但不作为本次探索性分类器的训练权重；因此结果不应解释为总体估计。

## 已知限制

- `fracture_history` 是“医生曾告知其他骨折”的横断面历史标签，没有事件日期和 24 个月观察窗。
- OSQ 和 DXA 子样本存在选择性缺失；结果可能受调查设计和检查资格影响。
- 本版本不包含步态或足底压力变量，不能代表完整多模态方案。
- 训练结果仅用于公开数据探索，不构成诊断、筛查建议或临床性能声明。

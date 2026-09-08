# NHANES 公开数据交接：nhanes_exploratory_v1

## 交接范围

本交接包面向公开横断面探索，不替代 `fracture_within_24m` 正式接口。标签为 `fracture_history`（CDC OSQ080），`osteoporosis_diagnosis`（CDC OSQ060）仅作为描述字段保留。

## 输入与产物

| 项目 | 路径 |
| --- | --- |
| 原始 XPT | `data/public/nhanes_2017_2018/raw/` |
| 数据版本 | `docs/data_versions/nhanes_2017_2018_public_v1.md` |
| 字段接口 | `docs/interfaces/nhanes_public_exploratory_v1.md` |
| 特征表 | `data/public/nhanes_2017_2018/processed/feature_table_nhanes_exploratory_v1.csv` |
| 切分清单 | `data/public/nhanes_2017_2018/processed/split_nhanes_exploratory_v1.csv` |
| 质量报告 | `docs/data_quality/nhanes_2017_2018_public_v1.md` |
| 基线配置 | `configs/nhanes_exploratory_v1.json` |
| 基线结果 | `reports/baselines/nhanes_exploratory_v1_metrics.json` |

## 复现命令

将 `<python-3.12>` 替换为成员本机的 Python 3.12 可执行文件：

```powershell
<python-3.12> scripts\build_nhanes_exploratory.py
<python-3.12> scripts\run_nhanes_baseline.py
<python-3.12> -m pytest -q
```

## 审查结论

- [x] 官方 CDC/NCHS 数据页面、变量文档和 XPT 下载链接已记录；原始文件 SHA-256 已记录。
- [x] 合并后 865 行，ID 唯一，标签编码仅为 0/1，切分为 519/173/173。
- [x] 缺失 BMD 仅由训练集合拟合的插补器处理；测试集合未用于选阈值。
- [x] 逻辑回归和随机森林基线已完成并绑定数据版本、配置哈希和环境。
- [ ] 成员 B 确认该公开数据接口是否适配融合模型。
- [ ] 项目负责人确认公开数据的引用、许可和研究方案纳入方式。

## 结果限制

测试 AUROC 约为 0.49（随机森林）和 0.56（逻辑回归），特异度约为 4%–7%。该结果不支持当前特征组合用于筛查，也不支持任何临床结论；应优先检查终点定义、横断面偏倚、调查子样本缺失和特征可得性，再决定是否继续扩展数据或更换终点。

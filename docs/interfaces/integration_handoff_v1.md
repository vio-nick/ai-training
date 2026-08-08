# 成员 A 集成交接：v1

## 交接状态

本交接包针对 `synthetic_v1` 工程验证数据，已完成成员 A 的本地审查；正式临床数据、正式研究接口和成员 B 的共同确认仍是后续集成前置条件。

## 输入与版本

| 项目 | 版本/路径 |
| --- | --- |
| 数据版本 | `synthetic_v1` |
| 数据字典 | `docs/data_dictionary.md` |
| 特征表接口 | `docs/interfaces/feature_table_v1.md` |
| 切分接口 | `docs/interfaces/data_split_v1.md` |
| 特征表 | `data/processed/feature_table_v1.csv` |
| 切分清单 | `data/splits/split_v1.csv` |
| 基线配置 | `configs/baseline_synthetic_v1.json` |
| 基线结果接口 | `docs/interfaces/baseline_results_v1.md` |

## 可复现命令

在 Python 3.12 环境中从仓库根目录执行；将 `<python-3.12>` 替换为成员本机的 Python 3.12 可执行文件路径。本次审查使用 `D:\python-3.12.8\python.exe`：

```powershell
<python-3.12> scripts\generate_synthetic_data.py
<python-3.12> scripts\run_data_pipeline.py
<python-3.12> scripts\audit_split.py
<python-3.12> scripts\run_baselines.py
<python-3.12> -m pytest -q
```

## 已完成核对

- [x] 合成数据生成固定随机种子且可重复；
- [x] 数据字典、特征表列顺序和切分清单版本一致；
- [x] 必填字段、允许范围、缺失编码和 ID 唯一性已自动校验；
- [x] 特征表和切分清单 ID 集合完全一致，无参与者跨集合；
- [x] 训练/验证/测试为 144/48/48，每个集合均含两类标签；
- [x] 插补、编码和标准化只在训练集合拟合；
- [x] 逻辑回归和随机森林基线结果绑定数据版本、配置哈希、随机种子和环境；
- [x] 四项数据测试与两项基线输出测试通过；
- [ ] 成员 B 确认字段、切分、结果格式和适配器接口；
- [ ] 使用获批的真实/公开数据完成正式数据审查和重跑。

## 交付限制

本交接包中的数据和指标均为合成开发产物，不支持临床性能或部署结论。成员 B 接入正式数据时必须先运行接口校验；任何字段、标签、单位、缺失编码或切分不兼容都应发布新接口版本并保留适配器。

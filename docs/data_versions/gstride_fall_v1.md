# GSTRIDE 跌倒者识别数据版本：v1

## 来源与许可

- 数据集：GSTRIDE v1.0，Zenodo 记录 [8003441](https://zenodo.org/records/8003441)
- DOI：`10.5281/zenodo.8003441`
- 许可证：CC BY 4.0（以 Zenodo 页面当前条款为准）
- 原始文件：`GSTRIDE_database/Database_register.csv`
- 本地原始文件：`data/public/gstride_v1/raw/Database_register.csv`
- 原始文件编码：CP1252；字段分隔符：分号

仓库只保留 register 表中构建六项汇总步态特征所需的文件，不提交约 1.37 GB 的完整原始传感器压缩包。原文件 SHA-256：`5350c8a2c4cac19d6c420af5dfaf28ebd13d5ed8dea2c400e14f16c89c99ed25`。

## 内容与标签

当前版本包含 163 名参与者，`faller_last_year` 为步态测试前一年内自报跌倒史：`1=YES`（至少一次）、`0=NO`。类别计数为 86/77。每名参与者仅保留一行；参与者 ID 在特征表和切分表中唯一。

处理后文件为 `feature_table_gstride_fall_v1.csv`、`split_gstride_fall_v1.csv` 和 `gstride_fall_v1_quality.json`。随机种子 `20260907` 下切分为训练 98、验证 32、测试 33；处理后文件 SHA-256 由质量 JSON 记录。

## 处理规则

脚本 `scripts/build_gstride_fall_dataset.py` 读取前三行文档/表头后，按显式列索引映射来源字段，清理空值、`-` 和 `Incapable`，再计算六项特征。当前数据六项特征均无缺失；若未来版本出现缺失，训练管线仅允许在训练集拟合中位数插补器。

该版本用于传感器核心原型，不代表 GSTRIDE 全部原始信号，也不构成临床验证队列。

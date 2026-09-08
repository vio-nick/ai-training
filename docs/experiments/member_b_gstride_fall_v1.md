# 成员 B 实验与模型比较：GSTRIDE v1

成员 B 已将训练、评估、校准、解释和轻量推理框架接入 `gstride_fall_v1`。正式结果只针对六项步态特征的回顾性跌倒者识别。无版本兼容的 BMD 或医院纵向第二模态，因此没有真实融合模型或融合性能。

## 复现

```powershell
D:\anaconda3\python.exe scripts\run_member_b_gstride_workflow.py
D:\anaconda3\python.exe scripts\run_member_b_fusion_fixture.py
```

固定测试集（33 人）结果：

| 模型 | AUROC | AUPRC | 阈值 | 敏感度 | 特异度 | F1 | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 逻辑回归 | 0.7868 | 0.7771 | 0.286539 | 0.8824 | 0.6250 | 0.7895 | 0.2037 | 0.1394 |
| 随机森林 | 0.7868 | 0.8030 | 0.388417 | 0.8235 | 0.7500 | 0.8000 | 0.1840 | 0.0861 |

阈值仅在验证集以平衡准确率选择；校准、阈值影响、错误案例和解释输出来自固定测试集。样本量小，不能据此宣布临床性能或外部可推广性。

产物位于 `reports/member_b/`：工作流 JSON、测试集预测、错误案例、亚组不可用模板、逻辑回归逐样本贡献和随机森林全局重要性。亚组字段在六特征契约中不存在，报告明确记录为不可用。

`member_b_fusion_fixture_v1_smoke.json` 使用无标签合成分数验证早期、晚期、混合融合及三种缺失模态策略；它不是训练结果，不能与 GSTRIDE 指标比较。

环境：Python 3.13.9、pandas 2.3.3、numpy 2.3.5、scikit-learn 1.7.2。依赖声明在 `pyproject.toml`。

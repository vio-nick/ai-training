# 成员 B 推理与融合接口：v1

本接口建立在 `gstride_fall_v1` 六特征契约之上。正式可运行模型为单一步态传感器模态；融合接口仅为未来合规医院数据预留。

## 单模态推理

`FallerInferenceModel.predict_record` 接收六个数值字段：`step_speed_m_s`、`cadence_strides_per_min`、`stride_length_m`、`double_support_pct`、`swing_to_stance_ratio`、`stride_time_cv_pct`。`participant_id` 可选并原样返回。调用方应提供 `data_version="gstride_fall_v1"`。

接口拒绝缺列、非数值、无穷值和版本不兼容的输入。`faller_last_year` 不是推理输入，即使记录带有该列也不会被读取。输出包括 `predicted_probability`、`predicted_label`、`decision_threshold`、`data_version` 和 `model_name`。

推理阶段只使用训练时已拟合的插补器/标准化器，不会重新拟合。

## 融合接口

`ModalitySpec` 显式声明模态名称、特征列和是否必需。早期融合拼接特征，晚期融合按权重平均概率，混合融合组合早期与晚期概率。三者都返回溯源元数据。

缺失模态策略：

- `error`：任一声明模态不可用时失败；
- `available_only`：丢弃不完整的可选模态并记录；
- `impute_zero`：缺失模态列或空值置零并记录。

必需模态不可用时始终失败。当前没有 BMD 或医院纵向数据，因此融合 fixture 只验证接口行为，不产生临床性能结论。

# GSTRIDE IMU 步态参数算法

本目录包含 GSTRIDE 的步态算法。原始 `main.py` 从专用文本日志读取数据，并应用设备标定；`csv_adapter.py` 为仓库中的已标定 CSV 提供输入接口，再调用相同的 AHRS、事件检测、INS/ZUPT 和步态参数算法。

## 处理流程

```text
已标定 CSV -> 列/数值/时间戳检查 -> 短缺口重采样 -> Mahony AHRS
           -> 步态事件检测 -> midFF 锚点 INS/ZUPT -> 周期参数 -> 训练口径特征
```

`midFF` 是原始 `main.py` 传给 `INS_ZUPT.UPDATE_ZUPT` 的事件。CSV 接入时也固定使用该事件作为 ZUPT 锚点。CSV 适配器负责数据接入和采样率规整，不能把它理解为另一套 ZUPT 锚点方案。已经标定的数据不能再次套用其他设备的 `calib_*.mat`。

## 模型输入特征

仓库中的模型特征由训练表 `Database_register.csv` 的汇总值构建，不是从一组共同筛选后的周期直接计算。GSTRIDE 的 `Gait_Parameters_Estimation.statistics` 对每个周期指标分别调用 `trunc_array`。当前实现的 `limit=0.4`，对正中位数保留其中位数 40%–140% 范围内的值，再分别计算均值和标准差。六项模型特征按训练表字段派生：

```text
step_speed_m_s          = Step Speed Avg
cadence_strides_per_min = Cadence Avg
stride_length_m         = Stride Length Avg
double_support_pct      = Load Avg + Push Avg
swing_to_stance_ratio   = Swing Avg / (Load Avg + Foot Flat Avg + Push Avg)
stride_time_cv_pct      = 100 * Stride Time Std / Stride Time Avg
```

GPE 汇总值的精度也影响派生值：周期时长、周期标准差、步频、步速和跨步长按 3 位小数；相位百分比均值按 2 位小数。CSV 模型输入复用这一精度，再按上面的训练表公式派生特征。不同特征的周期截尾集合可以不同，审计结果会保留各特征使用的周期索引和数量。

## 周期质量与末尾静止

推理前对周期索引顺序、周期时长、步长和步速进行质量检查；不满足条件的周期不会进入候选汇总。通过检查后，每个周期指标仍按训练算法独立截尾。当前仓库没有训练样本的原始周期波形，无法从训练数据推出“与结尾静止段重叠就删除整周期”的规则，因此不会默认执行这种删除。结尾静止段可帮助事件检测，但本身不构成自动删周期的依据。

具体固定规则与适用边界见 [`docs/imu_gait_processing_rules.md`](../../docs/imu_gait_processing_rules.md)。模型分数表示步态测试前一年内自报跌倒史的识别分数，不是未来跌倒概率或临床诊断。

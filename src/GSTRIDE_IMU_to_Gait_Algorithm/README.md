# GSTRIDE IMU -> 步态参数算法

本文件夹提取了 GSTRIDE 数据库中“IMU 原始数据 -> 步态参数”的 Python 算法，源代码逻辑保持原样。

## 文件

- `main.py`：完整流程入口?
- `readLogFileArduinoCalib.py`：读取原始 IMU 文本，应用传感器标定和时钟校正。
- `MahonyAHRS.py`：Mahony 姿态估计。
- `StepDetection_MultiCorrector_Pitch.py`：静止/运动/足底平放以及 Toe-off/Heel-strike 检测。
- `INS_ZUPT.py`：重力去除、速度积分、 ZUPT 和位移估计。
- `Gait_Parameters_Estimation.py`：步态周期、步频、速度、步长和相位比例计算。
- `Calibration/`：传感器 `calib_*.mat` 标定文件。
- `requirements.txt`：依赖版本。

## 处理链

```text
原始 IMU -> 标定 -> Mahony AHRS -> 步态事件检测 -> INS/ZUPT -> 步态参数
```

## 输入和运行

原始文件为六列 `accX accY accZ gyrX gyrY gyrZ`，文件名如 `V001_45C9.txt`。运行前安装依赖?

```bash
pip install -r requirements.txt
```

`main.py` 是从原仓库复制的入口，默认从 `../Test_recordings_raw/` 读取数据。如果从本目录运行，请把 `main.py` 中的 `dir` 改为实际原始数据路径，或按原仓库目录结构运行。

## 输入表字段的派生

```text
swing_to_stance_ratio = Pswing / (100 - Pswing)
double_support_pct    = Load Avg + Push Avg
                       = mean(PHS) + mean(PTO)
stride_time_cv_pct    = 100 * std(tcycle) / mean(tcycle)
```

`double_support_pct` 是按本数据集的 `Load + Push` 约定计算；源代码本身没有单独的同名变量。

原始数据、已标定数据和已有结果仍位于原仓库的 `Test_recordings_raw/`、`Test_recordings_calibrated/`、`Test_outputs_gait_analysis/` 和 `Database_register.csv`。

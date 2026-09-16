# GSTRIDE 跌倒风险推理 API

本目录提供一个稳定的 HTTP API 包装层，供前端或其他服务调用模型推理能力。

> **重要提示**：当前模型为研究原型，输出的是 GSTRIDE 公开数据集中"过去一年是否自报跌倒"的回顾性分数，**不能用于骨折风险预测、临床诊断、治疗决策或健康筛查**。

---

## 安装依赖

在仓库根目录执行：

```powershell
cd D:\我的桌面\git\ai-training
python -m pip install -e ".[api]"
```

如果 `pyproject.toml` 尚未更新 `api` 可选依赖，则手动安装：

```powershell
python -m pip install fastapi uvicorn pydantic
```

---

## 发布模型

API 启动时需要加载一个已发布的模型包。如果尚未发布，先执行：

```powershell
cd D:\我的桌面\git\ai-training
python scripts/release_gstride_fall_model.py
```

默认发布目录为：

```text
artifacts/model_releases/gstride_fall_v1_random_forest_full_historical/
```

如果发布目录不同，启动 API 时通过环境变量指定：

```powershell
$env:MODEL_RELEASE_DIR = "D:\你的\发布\目录"
```

---

## 启动 API

```powershell
cd D:\我的桌面\git\ai-training\api
uvicorn main:app --host 0.0.0.0 --port 8000
```

启动后访问：

- 接口文档：`http://localhost:8000/docs`
- 健康检查：`http://localhost:8000/health`
- 根路径：`http://localhost:8000/`

---

## 接口约定

### 1. 单条预测

**请求**

```http
POST /predict
Content-Type: application/json

{
  "participant_id": "user-001",
  "step_speed_m_s": 0.82,
  "cadence_strides_per_min": 52.0,
  "stride_length_m": 0.95,
  "double_support_pct": 24.0,
  "swing_to_stance_ratio": 0.31,
  "stride_time_cv_pct": 8.5
}
```

**响应**

```json
{
  "participant_id": "user-001",
  "predicted_probability": 0.45,
  "predicted_probability_percent": 45.0,
  "risk_level": "medium",
  "risk_level_display": "中风险",
  "predicted_label": 1,
  "decision_threshold": 0.388,
  "data_version": "gstride_fall_v1",
  "model_name": "random_forest"
}
```

### 2. 批量预测

**请求**

```http
POST /predict/batch
Content-Type: application/json

[
  {
    "participant_id": "user-001",
    "step_speed_m_s": 0.82,
    ...
  },
  {
    "participant_id": "user-002",
    "step_speed_m_s": 0.75,
    ...
  }
]
```

**响应**：预测结果数组。

### 3. 健康检查

```http
GET /health
```

用于确认服务状态和模型加载情况。

---

## 前端调用示例（JavaScript）

```javascript
const response = await fetch("http://localhost:8000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    participant_id: "user-001",
    step_speed_m_s: 0.82,
    cadence_strides_per_min: 52.0,
    stride_length_m: 0.95,
    double_support_pct: 24.0,
    swing_to_stance_ratio: 0.31,
    stride_time_cv_pct: 8.5
  })
});

const result = await response.json();
console.log(result.risk_level_display); // "中风险"
```

---

## 部署模式说明

当前模型为**无状态模型**：每次请求独立处理，不会从使用中学习或更新。因此 API 服务可以按需启动，不需要为了"保持模型状态"而持续运行。

如需固定公网访问地址，可部署到任意支持 Python 的容器/函数计算平台，并暴露统一的 API URL。

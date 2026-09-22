# GSTRIDE HTTP API

接口最初来自 `origin/memberB` 的 `90e1a98`；当前在 `feat/frontend` 中接入 `dev` 的六特征贡献解释，并补齐请求/响应类型、错误处理和生命周期加载。

## 本地启动（仓库根目录）

```sh
uv venv .venv
uv pip install --python .venv/bin/python -e '.[api,dev]'
.venv/bin/python scripts/release_gstride_fall_model.py
.venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

发布脚本只需首次生成模型时运行。若发布目录已有文件，它会拒绝覆盖；有意重新训练时才使用 `--overwrite`。模型包在 Git 忽略的 `artifacts/` 中，换机器需重新生成或取得受信任的发布包。生成完成后再启动 API；运行中的服务不会自动重新加载模型。

`MODEL_RELEASE_DIR` 可指定模型包目录。默认不开放跨域，前端使用 Vite `/api` 代理；需要直接跨域时设置 `CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173`。

## 接口

- `GET /health`：模型可用时 200，不可用时 503。
- `POST /predict`：单条评分。
- `POST /predict/batch`：1–100 条记录的 JSON 数组。
- `GET /docs`：完整 OpenAPI 请求与响应结构。

```json
{
  "participant_id": "demo-001",
  "step_speed_m_s": 0.82,
  "cadence_strides_per_min": 52,
  "stride_length_m": 0.95,
  "double_support_pct": 24,
  "swing_to_stance_ratio": 0.31,
  "stride_time_cv_pct": 8.5
}
```

记录编号可省略；六项参数必须是有限非负数，双支撑比例不超过 100。步频单位是 strides/min（跨步/分钟），不能直接输入 steps/min；跨步长单位为 m。拒绝未知字段。

响应保留概率、展示分级、二元判定与模型版本，新增类型明确的 `feature_contributions` 六项贡献、解释方法、基准概率和重构概率。基准概率加六项贡献等于最终概率。`predicted_label` 依据发布包阈值，低/中/高展示分组依据 30%/70%，两者不同。

错误统一有字符串 `detail`：422 输入错误（可含 `errors` 字段列表）、503 模型未就绪、500 评分异常。模型文件路径和内部异常只写服务日志。

## 验证

```sh
.venv/bin/python -m unittest discover -s tests
```

`test_api.py` 使用临时目录生成真实模型，验证健康检查、评分及贡献加和、批量一致性、错误输入、模型缺失和响应类型。

这是既往自报跌倒识别原型，不是未来跌倒或骨折预测。接口不保存记录，不上传数据到第三方服务。

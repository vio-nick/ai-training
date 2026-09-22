import { z } from "zod";
export const fields = [
  {
    key: "step_speed_m_s",
    label: "步速",
    unit: "m/s",
    example: 0.82,
    hint: "标准化步行测试的平均速度",
  },
  {
    key: "cadence_strides_per_min",
    label: "步频",
    unit: "strides/min",
    example: 52,
    hint: "跨步/分钟，请勿直接填入 steps/min",
  },
  {
    key: "stride_length_m",
    label: "跨步长",
    unit: "m",
    example: 0.95,
    hint: "同一只脚相邻两次着地之间的距离",
  },
  {
    key: "double_support_pct",
    label: "双支撑比例",
    unit: "%",
    example: 24,
    hint: "两脚同时接触地面的时间占比，0–100",
  },
  {
    key: "swing_to_stance_ratio",
    label: "摆动/支撑比",
    unit: "",
    example: 0.31,
    hint: "摆动时间与支撑时间之比",
  },
  {
    key: "stride_time_cv_pct",
    label: "跨步时间变异度",
    unit: "%",
    example: 8.5,
    hint: "跨步时间标准差 ÷ 均值 × 100",
  },
] as const;
const number = z.number({ error: "请输入有效数值" }).min(0, "不能小于 0");
export const inputSchema = z.object({
  participant_id: z.string().max(80, "编号最多 80 个字符"),
  step_speed_m_s: number,
  cadence_strides_per_min: number,
  stride_length_m: number,
  double_support_pct: number.max(100, "不能超过 100%"),
  swing_to_stance_ratio: number,
  stride_time_cv_pct: number,
});
export type Inputs = z.infer<typeof inputSchema>;
const contribution = z.object({
  feature: z.string(),
  display_name: z.string(),
  unit: z.string(),
  value: z.number().nullable(),
  contribution_probability_percent: z.number(),
  direction: z.enum(["positive", "negative", "neutral"]),
});
const prediction = z.object({
  participant_id: z.string().nullable().optional(),
  predicted_probability_percent: z.number(),
  risk_level: z.enum(["low", "medium", "high"]),
  risk_level_display: z.string(),
  model_name: z.string(),
  data_version: z.string(),
  feature_contribution_baseline_probability: z.number(),
  feature_contributions: z.array(contribution).length(6),
});
export type Prediction = z.infer<typeof prediction>;
export async function predict(
  input: Inputs,
  signal: AbortSignal,
): Promise<Prediction> {
  let response: Response;
  try {
    response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
      signal,
    });
  } catch (error) {
    if (signal.aborted) throw error;
    throw new Error("无法连接评分服务，请确认后端已启动后重试。");
  }
  let data: unknown;
  try {
    data = await response.json();
  } catch {
    throw new Error("评分服务未返回有效数据，请检查后端连接。");
  }
  if (!response.ok) {
    const detail = (data as { detail?: unknown }).detail;
    throw new Error(
      typeof detail === "string" ? detail : "输入未通过校验，请检查各项参数。",
    );
  }
  const parsed = prediction.safeParse(data);
  if (!parsed.success)
    throw new Error("评分结果格式不兼容，请确认后端已包含特征解释接口。");
  return parsed.data;
}

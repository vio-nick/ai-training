import { useEffect, useRef, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import {
  Activity,
  ArrowRight,
  Check,
  LoaderCircle,
  RotateCcw,
} from "lucide-react";
import {
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  fields,
  inputSchema,
  predict,
  type Inputs,
  type Prediction,
} from "@/lib/api";

export default function App() {
  const [started, setStarted] = useState(false);

  if (started) return <Workbench />;

  return (
    <div className="home-page">
      <header className="topbar">
        <div className="brand">
          <Activity aria-hidden="true" />
          GSTRIDE
        </div>
        <span>v1.0</span>
      </header>
      <main className="home-main">
        <section className="home-content" aria-labelledby="home-title">
          <div className="home-icon" aria-hidden="true">
            <Activity />
          </div>
          <p className="home-eyebrow">步态分析 · 跌倒风险研究</p>
          <h1 id="home-title">
            <span>基于IMU步态参数的</span>
            <span>跌倒风险研究评估软件</span>
          </h1>
          <p className="home-description">
            输入六项步态参数，查看既往跌倒识别分数和参数贡献。
          </p>
          <Button size="lg" className="home-start" onClick={() => setStarted(true)}>
            开始使用
            <ArrowRight data-icon="inline-end" aria-hidden="true" />
          </Button>
        </section>
      </main>
    </div>
  );
}

function Workbench() {
  const titleRef = useRef<HTMLHeadingElement>(null);
  useEffect(() => titleRef.current?.focus(), []);
  const {
    register,
    handleSubmit,
    reset,
    formState: { errors },
  } = useForm<Inputs>({
    resolver: zodResolver(inputSchema),
    defaultValues: { participant_id: "" },
  });
  const [result, setResult] = useState<{
    prediction: Prediction;
    input: Inputs;
  } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);
  async function submit(input: Inputs) {
    controller.current?.abort();
    const request = new AbortController();
    controller.current = request;
    setBusy(true);
    setResult(null);
    setError("");
    const timer = window.setTimeout(() => request.abort(), 30000);
    try {
      const prediction = await predict(input, request.signal);
      setResult({ prediction, input });
    } catch (e) {
      setError(
        request.signal.aborted
          ? "评分超时，请稍后重试。"
          : e instanceof Error
            ? e.message
            : "评分失败，请重试。",
      );
    } finally {
      clearTimeout(timer);
      setBusy(false);
    }
  }
  function invalidSubmit() {
    setResult(null);
    setError("评估失败：请检查六项步态参数，所有参数都必须填写有效数值并处于允许范围。");
  }
  const data = result?.prediction.feature_contributions.map((c) => ({
    ...c,
    name: fields.find((f) => f.key === c.feature)?.label ?? c.display_name,
    score: c.contribution_probability_percent,
  }));
  return (
    <>
      <header className="topbar">
        <div className="brand">
          <Activity aria-hidden="true" />
          GSTRIDE
        </div>
        <span>研究原型</span>
      </header>
      <main>
        <section className="intro">
          <h1 ref={titleRef} tabIndex={-1}>步态评估工作台</h1>
          <p>输入六项步态参数，查看既往跌倒识别评分与特征解释。</p>
        </section>
        <div className="workbench">
          <section className="panel input-panel" aria-labelledby="input-title">
            <div className="panel-heading">
              <h2 id="input-title">步态参数</h2>
              <Button
                variant="outline"
                disabled={busy}
                onClick={() => {
                  reset({
                    participant_id: "demo-001",
                    ...Object.fromEntries(
                      fields.map((f) => [f.key, f.example]),
                    ),
                  } as Inputs);
                  setResult(null);
                  setError("");
                }}
              >
                填入示例
              </Button>
            </div>
            <form
              onSubmit={handleSubmit(submit, invalidSubmit)}
              onChange={() => {
                setResult(null);
                setError("");
              }}
              noValidate
            >
              <fieldset disabled={busy}>
                <div className="record">
                  <label htmlFor="participant_id">
                    记录编号 <span>（可选）</span>
                  </label>
                  <Input
                    id="participant_id"
                    placeholder="如：P001"
                    maxLength={80}
                    {...register("participant_id")}
                  />
                  {errors.participant_id && (
                    <p className="field-error">
                      {errors.participant_id.message}
                    </p>
                  )}
                </div>
                <div className="field-grid">
                  {fields.map((f) => (
                    <div className="field" key={f.key}>
                      <label htmlFor={f.key}>{f.label}</label>
                      <div className="input-wrap">
                        <Input
                          id={f.key}
                          type="number"
                          inputMode="decimal"
                          step="any"
                          min="0"
                          max={f.key === "double_support_pct" ? 100 : undefined}
                          placeholder="请输入数值"
                          aria-invalid={!!errors[f.key]}
                          aria-describedby={`${f.key}-hint ${f.key}-error`}
                          {...register(f.key, { valueAsNumber: true })}
                        />
                        <span className="unit">{f.unit}</span>
                      </div>
                      <p className="hint" id={`${f.key}-hint`}>
                        {f.hint}
                      </p>
                      <p className="field-error" id={`${f.key}-error`}>
                        {errors[f.key]?.message}
                      </p>
                    </div>
                  ))}
                </div>
                <div className="form-actions">
                  <Button type="submit" className="primary">
                    {busy ? <LoaderCircle className="spin" /> : <Activity />}
                    {busy ? "正在评估…" : "开始评估"}
                  </Button>
                  <Button
                    variant="outline"
                    type="button"
                    onClick={() => {
                      reset({
                        participant_id: "",
                        ...Object.fromEntries(fields.map((f) => [f.key, NaN])),
                      } as Inputs);
                      setResult(null);
                      setError("");
                    }}
                  >
                    <RotateCcw />
                    重置
                  </Button>
                </div>
              </fieldset>
            </form>
            {error && (
              <div className="error-banner" role="alert">
                {error}
              </div>
            )}
          </section>
          <section
            className="panel result-panel"
            aria-labelledby="result-title"
            aria-busy={busy}
          >
            <div className="panel-heading">
              <h2 id="result-title">评估结果</h2>
              {result && !busy && !error && (
                <span className="completed">
                  <Check size={15} />
                  已完成
                </span>
              )}
            </div>
            <p className="disclaimer">
              此结果是基于六项步态参数的研究分数，仅用于研究与技术演示，不能预测未来跌倒，不能用于诊断、筛查或医疗决策。
            </p>
            <div aria-live="polite">
              {error && <p className="result-error" role="alert">{error}</p>}
              {busy && <p className="stale">正在计算本次评分，请稍候…</p>}
              {!result ? (
                <div className="empty">
                  <div className="empty-icon">
                    <Activity />
                  </div>
                  <h3>{busy ? "正在评估" : error ? "评估失败" : "等待评估"}</h3>
                  <p>
                    {busy
                      ? "模型正在计算评分与特征贡献。"
                      : error
                        ? "请修正输入或稍后重试。"
                        : "完成参数输入后开始评估。"}
                  </p>
                </div>
              ) : (
                <div className="score-block">
                  <p className="score-label">既往跌倒识别评分</p>
                  <div className="score-line">
                    <strong>
                      {result.prediction.predicted_probability_percent.toFixed(
                        1,
                      )}
                      <small>%</small>
                    </strong>
                    <span className={`level ${result.prediction.risk_level}`}>
                      {result.prediction.risk_level_display}分组
                    </span>
                  </div>
                  <div className="score-track">
                    <span
                      style={{
                        left: `${result.prediction.predicted_probability_percent}%`,
                      }}
                    />
                  </div>
                  <div className="scale-labels">
                    <span>0%</span>
                    <span>30%</span>
                    <span>70%</span>
                    <span>100%</span>
                  </div>
                  <p className="hint">
                    分组仅用于研究展示，不代表未来跌倒或骨折概率。
                  </p>
                </div>
              )}
            </div>
            <div className="contribution">
              <h2>特征贡献</h2>
              {!data ? (
                <div className="chart-empty">
                  <div className="zero-line" />
                  <p>评估后展示各项参数对本次分数的影响。</p>
                </div>
              ) : (
                <>
                  <p className="hint">
                    青色降低评分，橙色提高评分；单位为百分点。
                  </p>
                  <div
                    className="chart"
                    role="img"
                    aria-label="六项参数对评分的贡献图"
                  >
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={data}
                        layout="vertical"
                        margin={{ left: 0, right: 30, top: 15, bottom: 5 }}
                      >
                        <XAxis
                          type="number"
                          tickFormatter={(v) => `${v > 0 ? "+" : ""}${v}`}
                          axisLine={false}
                          tickLine={false}
                          fontSize={12}
                        />
                        <YAxis
                          type="category"
                          dataKey="name"
                          width={116}
                          axisLine={false}
                          tickLine={false}
                          fontSize={12}
                        />
                        <ReferenceLine x={0} stroke="#ccd7df" />
                        <Tooltip
                          formatter={(value) => [
                            `${Number(value).toFixed(2)} 个百分点`,
                            "贡献",
                          ]}
                          cursor={{ fill: "#f4f7f9" }}
                        />
                        <Bar
                          dataKey="score"
                          barSize={14}
                          radius={3}
                          isAnimationActive={false}
                        >
                          {data.map((d) => (
                            <Cell
                              key={d.feature}
                              fill={d.score > 0 ? "#c78148" : "#168e98"}
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <details>
                    <summary>查看本次输入与贡献明细</summary>
                    <table>
                      <thead>
                        <tr>
                          <th>参数</th>
                          <th>输入</th>
                          <th>贡献（百分点）</th>
                        </tr>
                      </thead>
                      <tbody>
                        {data.map((d) => (
                          <tr key={d.feature}>
                            <td>{d.name}</td>
                            <td>
                              {d.value} {d.unit}
                            </td>
                            <td>
                              {d.score > 0 ? "+" : ""}
                              {d.score.toFixed(2)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </details>
                  <p className="hint explanation">
                    基准分数{" "}
                    {(
                      result!.prediction
                        .feature_contribution_baseline_probability * 100
                    ).toFixed(1)}
                    % 加上各项贡献得到本次评分。贡献表示模型关联，不是因果效应。
                  </p>
                  <div className="metadata">
                    <span>
                      记录：{result!.input.participant_id || "未填写"}
                    </span>
                    <span>
                      {result!.prediction.model_name} ·{" "}
                      {result!.prediction.data_version}
                    </span>
                  </div>
                </>
              )}
            </div>
          </section>
        </div>
        <div className="workflow">
          {[
            { title: "输入参数", text: "六项步态参数和可选记录编号。" },
            { title: "模型评分", text: "固定模型根据输入计算分数。" },
            { title: "结果解释", text: "查看分数和各参数的贡献。" },
          ].map((s, i) => (
            <div className="workflow-step" key={s.title}>
              <span className="step-number">{i + 1}</span>
              <div>
                <h3>{s.title}</h3>
                <p>{s.text}</p>
              </div>
              {i < 2 && <ArrowRight className="step-arrow" />}
            </div>
          ))}
        </div>
      </main>
    </>
  );
}

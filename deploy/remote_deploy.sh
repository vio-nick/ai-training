#!/usr/bin/env bash
set -euo pipefail

sha="${1:?缺少提交哈希}"
initialize_model="${2:-false}"
web_port="${3:-8080}"
[[ "$sha" =~ ^[0-9a-f]{40}$ ]] || { echo '提交哈希格式错误' >&2; exit 1; }
[[ "$initialize_model" == true || "$initialize_model" == false ]] || { echo '模型初始化参数错误' >&2; exit 1; }
[[ "$web_port" =~ ^[0-9]+$ ]] && (( 10#$web_port >= 1 && 10#$web_port <= 65535 )) || {
  echo 'WEB_PORT 必须在 1–65535 之间' >&2
  exit 1
}

base="$HOME/gstride"
release="$base/releases/$sha"
model_dir="$base/model_release"
archive="$base/incoming/$sha.tar.gz"
current="$base/current"
test -f "$archive" || { echo "部署包不存在：$archive" >&2; exit 1; }
mkdir -p "$release" "$model_dir"
tar -xzf "$archive" -C "$release"

export MODEL_RELEASE_DIR="$model_dir" WEB_PORT="$web_port" DEPLOY_SHA="$sha"
compose=(docker compose -p gstride -f "$release/deploy/compose.yml")
"${compose[@]}" config --quiet
"${compose[@]}" build

if [[ "$initialize_model" == true ]]; then
  if [[ -e "$model_dir/model.joblib" || -e "$model_dir/manifest.json" ]]; then
    echo '模型发布包已存在；本次拒绝重新训练或覆盖。请关闭首次初始化选项。' >&2
    exit 1
  fi
  docker run --rm \
    --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$release,dst=/work,readonly" \
    --mount "type=bind,src=$model_dir,dst=/model_release" \
    --workdir /work \
    --entrypoint python \
    "gstride-api:$sha" \
    scripts/release_gstride_fall_model.py --output-dir /model_release
fi

if [[ ! -s "$model_dir/model.joblib" || ! -s "$model_dir/manifest.json" ]]; then
  echo '缺少固定模型发布包。首次部署请勾选 initialize_model。' >&2
  exit 1
fi

previous=""
if [[ -L "$current" ]]; then
  previous="$(readlink -f "$current")"
fi

smoke_test() {
  curl --fail --silent --show-error --retry 6 --retry-delay 2 --retry-connrefused \
    "http://127.0.0.1:$web_port/api/health" > /dev/null
  curl --fail --silent --show-error \
    -H 'Content-Type: application/json' \
    -d '{"step_speed_m_s":0.82,"cadence_strides_per_min":52,"stride_length_m":0.95,"double_support_pct":24,"swing_to_stance_ratio":0.31,"stride_time_cv_pct":8.5}' \
    "http://127.0.0.1:$web_port/api/predict" > /dev/null
}

if ! "${compose[@]}" up --detach --no-build --wait || ! smoke_test; then
  echo '新版本健康检查失败，尝试恢复旧版本。' >&2
  if [[ -n "$previous" && -f "$previous/deploy/compose.yml" ]]; then
    export DEPLOY_SHA="$(basename "$previous")"
    docker compose -p gstride -f "$previous/deploy/compose.yml" up --detach --no-build --wait
    smoke_test
  fi
  exit 1
fi

ln -sfn "$release" "$current"
rm -f "$archive"
echo "部署完成：$sha，访问端口：$web_port"

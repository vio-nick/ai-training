# GitHub Actions 部署到 Linux 服务器

本方案由 GitHub 托管的流水线检查代码，通过 SSH 把当前提交的部署文件传到 Linux 服务器，然后在服务器上用 Docker Compose 构建并启动前端与 API。浏览器访问 `http://服务器IP:8080`；只有前端端口对外开放，API 由前端容器转发。默认端口可通过仓库变量 `WEB_PORT` 修改。

模型发布包保存在服务器的 `~/gstride/model_release/`，不在 Git 中，也不随普通部署重新训练。首次部署勾选 `initialize_model`，流水线才会使用仓库中固定的历史 GSTRIDE 特征表生成发布包；已有模型时会拒绝覆盖。

## 一次性准备

1. 在服务器上确认当前 SSH 用户可以执行 `docker info` 和 `docker compose version`。如需把用户加入 Docker 组，应重新登录后确认权限。开放 SSH 端口和网页端口（默认 TCP 8080）。
2. 为部署单独创建一把 Ed25519 SSH 密钥，把公钥加入服务器上部署用户的 `~/.ssh/authorized_keys`。不要把私钥提交到仓库。
3. 在 GitHub 仓库的 **Settings → Secrets and variables → Actions** 中设置以下 *repository secrets*：

   | 名称 | 内容 |
   | --- | --- |
   | `DEPLOY_HOST` | 服务器 IP |
   | `DEPLOY_USER` | 有 Docker 权限的 SSH 用户 |
   | `DEPLOY_SSH_KEY` | 上一步的完整私钥 |
   | `DEPLOY_KNOWN_HOSTS` | 已核对指纹的服务器 SSH 主机公钥记录，格式与 `known_hosts` 一致 |

   主机公钥记录可先在可信网络中用 `ssh-keyscan -p 22 服务器IP` 取得，再与服务器本机显示的指纹核对。流水线不会自动信任未知主机。

4. 如果 SSH 不是 22 端口，添加 repository variable `DEPLOY_PORT`；如果网页不是 8080 端口，添加 `WEB_PORT`。变量均为纯数字。
5. 将 `.github/workflows/deploy.yml` 合并到默认分支 `main`。GitHub 的手动运行按钮要求工作流文件已存在于默认分支。此后每次把要发布的代码推送到 `main`，在 **Actions → 部署 GSTRIDE → Run workflow** 选择 `main` 并运行。流水线仅允许从 `main` 部署。

## 首次与后续部署

首次运行时勾选 `initialize_model`。流水线会先运行前端构建与 API 测试，再上传当前提交的代码，构建两个镜像，生成固定模型，启动服务并检查 `/api/health`。成功后访问 `http://服务器IP:8080`。

后续更新代码时，先提交并推送到 `main`，再运行同一个工作流，保持 `initialize_model` 不勾选。部署脚本会复用原模型、为新提交构建镜像、等待服务健康，并在新版本健康检查失败时尝试恢复上一个成功版本。提交 SHA 对应的部署包保存在 `~/gstride/releases/`，当前版本由 `~/gstride/current` 指向。

服务器上可用以下命令检查运行状态：

```sh
docker compose -p gstride -f ~/gstride/current/deploy/compose.yml ps
curl -fsS http://127.0.0.1:8080/api/health
```

服务器构建镜像时默认从清华 PyPI 镜像和 npmmirror 下载依赖。如果服务器连接其他源更快，可在执行部署脚本前设置 `PIP_INDEX_URL` 和 `NPM_REGISTRY` 环境变量；Docker Compose 会把它们传给镜像构建步骤。

模型文件应单独备份 `~/gstride/model_release/model.joblib` 和 `manifest.json`。如果未来有意发布新模型，应先审查数据与训练配置，再走单独的模型发布流程；普通代码部署不会覆盖它们。

当前方案按 IP 与端口提供 HTTP。若要录入真实个人数据，先配置域名、HTTPS 和访问控制。

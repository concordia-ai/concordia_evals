---
name: port-benchmark-to-inspect
description: 把一个不兼容 inspect 的外部 agent benchmark（带 Docker 环境/自定义评分/harness）移植为本仓（concordia_evals）的 inspect-native eval。覆盖调研、保真移植、inspect_swe 多 harness 接入、断网隔离、分层验证与冒烟测试方法论。当需要移植新 benchmark 或评审这类移植时使用。
---

# 移植外部 Agent Benchmark 到 Inspect（concordia_evals）

本仓的标杆实现：`src/concordia_evals/cybergym_e2e/`（CyberGym-E2E）。先读它，再动手。
它的 `DEVLOG.md` 记录了全部上游事实与踩坑明细，本文是方法论提炼。

## 0. 总原则：保真优先

移植的最大风险是**评分语义跑偏**。三条铁律：

1. **上游评分脚本原样 vendor**（放 `vendor/`，与上游逐字节一致，md5 记录；
  从 ruff/mypy 排除）。我们的代码只做编排，不改判定逻辑。
2. **prompt 逐字移植**，不做任何"润色"——prompt 文本是 benchmark 可比性契约的一部分。
3. **一切 pin 死**：上游代码仓 commit、数据集 revision、基础镜像 digest、
  harness 二进制版本。

## 1. 调研阶段（写代码前必须完成并记录）

把下列事实写进新 benchmark 目录的 `DEVLOG.md`，再开始实现：

- 上游运行链路：入口脚本怎么跑 agent → 产出什么 artifact → 怎么判分。
  **把上游的 runner、判分脚本、工具函数完整读一遍**（不是只读 README）。
- 数据形态：数据集在哪（HF/GitHub/对象存储）、是否 gated、总量多大、
  单任务文件构成；脚本与数据是否分离。
- 环境形态：每任务镜像还是共享镜像、镜像数量与大小、是否需要特权
  （sysctl、--privileged、网络）。
- 判分口径：通过条件是什么、多阶段还是单阶段、阶段间是否需要干净环境
  （决定评分要不要起全新容器）。
- 反作弊设计：上游怎么防 agent 查答案（删 .git、网络隔离、隐藏 ground truth）。
- 现成实现检索：先查有没有官方/社区的 inspect 实现（inspect_evals 仓、
  harbor adapters、论文官方仓库），有就复用，没有才移植。
- 官方文档核对：https://inspect.aisi.org.uk/agents.html 与
  https://meridianlabs-ai.github.io/inspect_swe/ ——harness 优先用现成的。

## 2. 架构蓝本（照抄这个分层）

```
src/concordia_evals/<bench>/
├── __init__.py
├── <bench>.py          # @task 入口：参数、Sample 构建、solver 链组装
├── dataset.py          # 上游脚本仓/数据集的缓存式同步（platformdirs），
│                       # compose 文件生成（per-sample image + 只读 volume 挂数据）
├── prompts.py          # 上游 prompt 逐字移植
├── setup_solver.py     # 样本容器内环境布置（上游 setup 逻辑逐步移植）
├── validation.py       # 评分编排（需要干净环境时每阶段起全新容器）
├── scorer.py           # @scorer：读 agent 产出 → 编排 → Score(value=dict)
├── vendor/             # 上游判分脚本（字节一致）+ README 注明来源与校验方法
└── DEVLOG.md           # 调研事实、决策、踩坑、验证结果（全程维护！）
tests/<bench>/          # 无 docker 单元测试 + marker 隔离的 docker 冒烟
```

通用层放 `src/concordia_evals/agents/`（所有 benchmark 共享，勿放进单个
benchmark 内）：`registry.py`（harness 名 → Agent）与 `lockdown.py`（断网 solver）。

solver 链的标准形态：

```python
solver = chain(
    setup_workspace(),              # benchmark 特有
    network_lockdown(lockdown),     # 通用层：agent 阶段容器全断网
    as_solver(resolve_agent(harness, **harness_opts)),
)
```

## 3. 多 harness 接入：用 inspect_swe，别自写

`inspect_swe`（`pip install inspect-swe`）把 CLI harness 做成标准 Inspect agent：
`claude_code / codex_cli / kimi_code / gemini_cli / opencode`。机制是
`sandbox_agent_bridge`：容器内 localhost:13131 代理把 CLI 的模型调用转发回
inspect 的 model provider，因此：

- **任意 harness × 任意 inspect 模型**自由组合（`--model` 选模型，`-T harness=` 选
  harness），包括经 openai 兼容网关的 deepseek/kimi 等。
- transcript、token 用量、成本、time/token limit 全部原生获得；不需要往容器里
  透传 API key。
- 二进制由宿主下载后拷入（`download_agent_binary` 可预热缓存），容器无需网络安装。

只有 inspect_swe 没覆盖的 harness 才用 `sandbox_agent_bridge()` 自写薄 agent
（官方 agent-bridge 文档就是教程，约百行），写完放通用层供全仓复用。

## 4. 网络隔离：agent 阶段全断网，而非域名白名单

上游常用 squid 白名单（只允许 LLM API）。用了 sandbox_agent_bridge 后可以更简单
且更强：setup 阶段容器正常联网（装依赖、prepare），跑 agent 前用
`network_lockdown` 把容器从所有 docker 网络断开。依据：bridge 的传输是
docker exec 流 + 沙箱文件系统（`sandbox_service`），不走 TCP，容器零外网路由时
模型调用照常。compose 的 `web_search`/`code_execution` 等 granted capabilities
保持默认关闭。

## 5. 镜像与数据策略

- **镜像按需拉取**：不要在 dataset 构建时预拉（全量可能 TB 级）。compose 文件里
  **不要写 `x-local: true`**（那会跳过 inspect 的 pull-missing 流程）；inspect 在
  沙箱启动时自动 `docker compose pull --policy missing`。
- **数据按需下载**：大数据集（百 GB 级）按任务惰性下载进 platformdirs 缓存，
  pin revision；gated 数据集要 HF_TOKEN。
- **大文件走 volume 挂载**（compose volumes 只读挂宿主目录），不走
  `sandbox().write_file`（小文件才用它）。

## 6. 评分编排的时限语义（易踩）

inspect 的 `--time-limit` 只管 solver；超时后 scoring 另获 `time_limit/2` 的预算。
评分要真实编译/跑容器时，把 `--time-limit` 设大（如 14400，给评分 2h），
或不设。scorer 里跑同步重活用 `asyncio.to_thread` 包一层。

## 7. 测试方法论（四层递进）

1. **单元测试（无 docker）**：config 合并、prompt 内容、样本构建、评分跳过逻辑、
   registry 解析。
2. **合成任务 ground_truth 冒烟**（marker 隔离，如 `docker_smoke`）：自造一个
   含已知 bug 的最小项目 + 官方答案直接写入的 solver。在无模型的情况下验证
   compose→setup→验证→评分全链路，期望满分。这是最快暴露管线错误的一步。
3. **真实最小任务 + 真 harness + 便宜模型**：选全 benchmark 最小的任务，用
   便宜模型（如 deepseek-v4.1-flash）跑通 agent 真实链路。目的是验证管线，
   不是测能力。
4. **随机任务能力测试**：固定种子随机抽 3 个任务，观察 agent 真实表现与判分
   分布，确认评分能区分成功/失败（别用 agent 必过的题自嗨）。

## 8. 踩坑速查表（这次移植实际踩过的）

| 症状 | 原因 | 解法 |
|---|---|---|
| `No module named 'acp.helpers'` | agent-client-protocol 1.0.0 删了 acp.helpers，inspect_ai 约束太松 | pyproject pin `agent-client-protocol>=0.12.0,<1.0` |
| `OpenAI API requires at least version 3.4.0 of package openai` | inspect_ai 0.3.288 要求 openai>=3.4 | 升 openai，回归全仓测试 |
| 断网 solver 静默跳过、`sandbox()` 没有 `_project` | inspect 0.3.288 的 sandbox() 返回 SandboxEnvironmentProxy，私有属性拿不到 | 通过 sample metadata 传 compose 文件路径，用容器标签 `com.docker.compose.project.config_files` 定位容器 |
| inspect_swe kimi_code 报 "Context length metadata is unavailable" | 自定义模型无 context-length 元数据 | `harness_opts={"max_context_size": N}` |
| codex_cli 的 agent 工具调用全部失败、胡言 "tools namespace" | inspect_swe 把 slug 推导为 code_mode_only 的新 GPT（JS 代码工具），非 OpenAI 模型驱动不了 | 固定 `model_config="gpt-5.2"` 经典函数工具；真 GPT 模型再改回 |
| `403 ... gated repo` | HF 数据集 gated | 申请访问 + HF_TOKEN |
| eval 启动时串行预拉全部镜像（TB 级） | 在 build_samples 里 ensure_image | 删掉，交给 compose pull policy=missing |
| `uv sync` 后 pytest 没了 | 裸 sync 卸掉 dev extras | `uv sync --extra dev` |
| PyPI/GitHub 下载极慢 | 直连慢 | UV_INDEX_URL 清华镜像；HTTPS_PROXY=127.0.0.1:7890（本机代理）；LLM 网关是 127.0.0.1，no_proxy 已覆盖 |
| Docker Hub 网页/API 不通但 pull 正常 | hub.docker.com 被墙，registry-1.docker.io 可通 | 拉镜像不受影响，查信息用别的方式 |
| 读不了 /proc/sys/vm/mmap_rnd_bits | 受限 shell | sanitizer 兼容项，异常时再让宿主 `sudo sysctl -w vm.mmap_rnd_bits=28` |
| 以为 agent 卡死（无 socket/无进程） | CLI 的模型请求是短连接，瞬时值不可靠 | 读 CLI 自己的 session 日志（如 ~/.kimi-code/.../kimi-code.log 的 llm request/response 行） |

## 9. 收尾清单

- `pytest -m "not docker_smoke"`、ruff check、ruff format、mypy 全绿
  （vendor/ 已排除）
- README（eval 级：用法/参数/前置条件/保真说明）+ 主 README 登记表 + DEVLOG 完整
- `git status` 检查无测试残留（logs/ 应被 ignore；容器无残留）
- 单独分支提交，不要夹带他人未提交的文件

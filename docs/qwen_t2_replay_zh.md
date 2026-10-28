# 用自托管 Qwen 重跑 SERA T2，并与已发布 GLM 轨迹配对比较

推荐重跑已发布数据中的同一批 T2 任务。这样固定任务、初始代码、工具和提示词，比较模型生成的轨迹。重新跑完整 SVG pipeline 会产生新的 T1 修改和 PR 描述，适合研究数据生成分布，但不能把差异仅归因于 T2 solver。

## 已核对的信息

- [数据集](https://huggingface.co/datasets/allenai/Sera-4.6-Lite-T2)：36,083 条 T2；数据卡称来自 121 个 SWE-smith codebases，teacher 为 GLM-4.6。不是 SWE-bench Lite 的仓库子集。
- 固定版本：`d7d753f319f688797b8183c6668b17670992bdeb`；单个 JSONL 约 9.81 GB，脚本默认流式读取前 100 条，不下载整个文件。
- 字段：`messages`、`instance_id`、`rollout_patch`、`func_name`、`func_path`、`problem_statement`、`target_patch`、`docker_image`。`messages` 必须再 `json.loads()` 一次，已抽查样本包含原始 SWE-agent history 和结构化 `tool_calls`。
- `target_patch` 是 T1 的参考修改；`rollout_patch` 是 GLM 在 T2 的修改。它们不是经过独立测试确认的正确答案。数据卡明确说主要实验没有做 patch verification。
- 当前 SERA checkout：`1ca8673bd527b164dc9eb89841c96416ec8ce1a7`；SWE-agent submodule 固定于 `a593220bfdf9e07f7761179977b8ddf02fa981d5`。

## 1. 在运行 agent 的机器安装环境

建议在 Linux x86_64、有 Docker 权限的机器上执行。模型服务可以在另一台机器；GPU 只需部署在模型服务端。数据集镜像名明确包含 `x86_64`，在 Apple Silicon 上需要模拟，速度和兼容性需要另行验证。

以下命令均从 SERA 仓库根目录运行：

```bash
conda create -n sera-replay python=3.12 -y
conda activate sera-replay
git submodule update --init modules/SERA-SWE-Agent
pip install -e . -e modules/code2flow -e modules/SERA-SWE-Agent
docker info
```

只做这里的 T2 replay 不需要 mini-swe-agent、GitHub mirror、重新解析代码调用图，也不需要重新运行 T1。当前本地 checkout 的两个 agent submodule 目录尚为空，需要初始化。

## 2. 确认模型服务支持 OpenAI tools

这里的模型名称使用你的 `/v1/models` 返回的真实 ID，不假定“Qwen3.8 27B”对应某个公开权重名称。

```bash
export QWEN_BASE_URL='http://YOUR_HOST:YOUR_PORT/v1'
export QWEN_MODEL_ID='YOUR_SERVED_MODEL_ID'
export OPENAI_API_KEY="${OPENAI_API_KEY:-not-needed}"
curl -sS "$QWEN_BASE_URL/models" -H "Authorization: Bearer $OPENAI_API_KEY"
```

应验证 `chat.completions` 能返回结构化 `message.tool_calls`，其中包括合法 JSON `function.arguments`。仅能在文本里输出 `<tool_call>` 不足以让这里的 function-calling harness 正常工作。可用下面的小请求检查：

```bash
python - <<'PY'
import os
from openai import OpenAI
client = OpenAI(base_url=os.environ['QWEN_BASE_URL'], api_key=os.environ['OPENAI_API_KEY'])
r = client.chat.completions.create(
    model=os.environ['QWEN_MODEL_ID'],
    messages=[{'role': 'user', 'content': 'Call bash with command pwd. Do not answer with ordinary text.'}],
    tools=[{'type': 'function', 'function': {
        'name': 'bash', 'description': 'Run a shell command',
        'parameters': {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']}}}],
    temperature=0.6, max_tokens=1024)
print(r.choices[0].message.model_dump_json(indent=2))
PY
```

仓库的 `sera/datagen/inference/launch_qwen3_models.sh` 是既有 Qwen3/vLLM 示例，采用 `--enable-auto-tool-choice --tool-call-parser hermes`。你的具体模型与 serving 版本可能需要其他 parser/chat template，应以实际返回的结构化 tool calls 为准。

**推理文本需要单独检查**：固定版本的 `sweagent/agent/models.py` 保存 `message.content` 和 `tool_calls`，没有复制一般的 `reasoning_content` 字段。如果服务把推理全部放进该字段，原始 `.traj` 可能丢失这部分；`visible_think_chars=0` 不能解释为模型没有推理。比较推理长度之前，需确认服务把 `<think>…</think>` 保留在 `content`，或者修改 harness 显式记录单独的 reasoning 字段，并对两边采取一致的分析口径。

## 3. 准备固定任务集

```bash
python scripts/replay_t2.py prepare \
  --limit 100 \
  --out experiments/qwen_t2_inputs
```

输出包括：

- `instances.json`：SWE-agent 输入，只包含原始初始 prompt、镜像和 base commit。
- `prompt_overrides.yaml`：保留数据集已有完整首轮 user prompt，避免重新套模板导致空白或指令差异；system prompt 也从数据集读取。
- `references.jsonl`：同一批 GLM 轨迹、T1 target patch、GLM rollout patch、repo 信息，留作比较。
- `manifest.json`：来源版本、选中任务数、各 repo 数量与输入 SHA256。

脚本通过 `sera/constants.py` 的 `SWESMITH_IMAGES` 将**实际数据行的镜像**映射到 repo 和 base commit，映射不到会报错；不会把配置中额外的 repos 混入实验。输入中不额外暴露参考 patch 或采样函数位置；数据集原始问题描述已有的信息会原样保留。

默认是源文件顺序的前 N 条，便于 smoke test，**不是随机或有代表性的评估集**。可限定 repo，或每个 repo 取最多 10 个：

```bash
python scripts/replay_t2.py prepare \
  --repo msiemens/tinydb --limit 10 \
  --out experiments/tinydb_t2_inputs

# 扫描所有行，实际出现的每个 repo 最多保留 10 条。
python scripts/replay_t2.py prepare \
  --limit 0 --per-repo 10 \
  --out experiments/qwen_t2_per_repo10
```

按 repo 筛选可能需要读很大一段源文件；`--limit 0 --per-repo 10` 会扫描整个约 9.81 GB JSONL。反复实验建议先下载固定版本到本地，然后用 `--source /path/to/dataset.jsonl`。输出目录必须为空，防止更换任务集后误用旧结果。全量 replay 用 `--limit 0`，并保留足够磁盘存储 reference 和轨迹。

## 4. 先跑 1 个任务，再跑固定任务集

以下直接调用本仓库固定版本的 SERA-SWE-Agent。T2 使用普通任务求解模式 `--pipeline False`。这个 fork 即使关闭 pipeline 也会读取 `pipeline_yaml`，因此一并传入。

```bash
sweagent run-batch \
  --config "$PWD/sera/configs/sweagent/e2e.yaml" \
  --config "$PWD/experiments/qwen_t2_inputs/prompt_overrides.yaml" \
  --instances.type file \
  --instances.path "$PWD/experiments/qwen_t2_inputs/instances.json" \
  --instances.slice 1 \
  --instances.shuffle False \
  --num_workers 1 \
  --pipeline False \
  --pipeline_yaml "$PWD/sera/configs/pipeline/default_pipeline.yaml" \
  --agent.model.name "openai/$QWEN_MODEL_ID" \
  --agent.model.api_base "$QWEN_BASE_URL" \
  --agent.model.api_key '$OPENAI_API_KEY' \
  --agent.model.temperature 0.6 \
  --agent.model.per_instance_call_limit 115 \
  --agent.model.per_instance_cost_limit 0 \
  --agent.model.total_cost_limit 0 \
  --agent.model.max_input_tokens 28672 \
  --agent.model.max_output_tokens 4096 \
  --agent.model.completion_kwargs '{"max_tokens":4096}' \
  --output_dir "$PWD/experiments/qwen_t2_raw"
```

检查第一个任务的 `.traj`、`.pred`、日志和 `exit_status`，确认镜像能拉取、工具可用、模型确实执行工具。再删除命令中的 `--instances.slice 1`，将 `--num_workers` 改为 2 或 4，运行全部选中任务；同一 output directory 会按 harness 的恢复规则跳过已有完成轨迹。一个 27B endpoint 的并发从小值开始，根据服务吞吐和 Docker 容量提高。

这些是**本次可控的试跑预算**：当前仓库默认 temperature=0.6、call limit=115；这里另选 32K 总上下文、最多 28K 输入和 4K 单次输出。数据卡未提供足够配置来确认发布轨迹的原始预算与这些完全一致，因此不可宣称精确复现历史采样。先固定并记录自己的预算，再检查两边差异。模型不同 tokenizer 的 token 长度也不是等价单位。

固定版本对 OpenAI provider 不会自动把 `max_output_tokens` 转发成请求的 `max_tokens`，所以这里也显式设置 `completion_kwargs.max_tokens`。输入 token 计数可能使用 LiteLLM fallback tokenizer；严格预算比较时，应配置适合实际权重的 `agent.model.custom_tokenizer`。服务端也应有明确的上下文限制，并记录 chat template、thinking 开关、sampling 参数和版本。

这里从镜像的原始 commit 开始求解；**不要预先应用 `target_patch` 或 `rollout_patch`**，否则把答案写进环境。实际运行后还应记录镜像 digest、容器内 `git rev-parse HEAD`、SWE-agent 的版本和实际工具 schemas；镜像名或 tag 本身不能保证未来内容不变。

## 5. 比较原始轨迹

```bash
python scripts/replay_t2.py compare \
  --references experiments/qwen_t2_inputs/references.jsonl \
  --trajs experiments/qwen_t2_raw \
  --out experiments/qwen_t2_comparison
```

`summary.json` 和 `pairs.jsonl` 包含：

- 覆盖率、missing 数、Qwen `exit_status`、提交率、非空 patch 比例。
- 同一批有 Qwen `.traj` 的 ID 上，双方 assistant turns、tool calls、工具分布与顺序、assistant 字符数、可见 `<think>` 字符数。
- 双方相对 T1 `target_patch` 的 changed-line recall/precision、recall≥0.5 和 recall=1 的比例；Qwen patch 与 GLM patch 完全一致的比例。
- 首轮 system/user prompt 是否与数据集逐字匹配。出现不匹配应先排查再做模型归因。

脚本保留失败和缺失任务。提交率、patch 匹配率使用所有选中任务作为分母；没有轨迹计为未达成。长度均值只比较双方都有原始轨迹的同一批 ID。细粒度分析可从 `pairs.jsonl` 按 repo 聚合；正式比较建议同时报告 micro 和 repo-macro 结果，并在固定样本上重复多次 Qwen rollout 来衡量采样波动。

Recall 与本仓库 `compare_patch_recall` 采用相同的非空增删行匹配口径：保留正负号和重复次数，忽略 diff header、hunk、文件位置。**Recall=1 不等于 patch 完全相同，更不等于功能正确**；额外修改不会降低 recall。不同实现也可能功能等价，需要另行运行相关测试/复现脚本判定。参考 GLM 自身也应对 `target_patch` 打分，避免默认它总是正确。

行为差异还可抽样人工标注：定位路径、是否先复现、是否运行验证、编辑次数、重复搜索/命令、工具参数错误、无效编辑和是否修改测试文件。基础脚本只统计可客观提取的指标，不把“命令里出现 pytest”自动当成验证成功。

不要先走 `postprocess` 再做轨迹比较：默认 `enforce_submit=true` 会过滤未提交任务，`keep_only_think` 会改变 assistant 文本。保留 `.traj`/`.pred` 为分析来源；需要训练数据时另行生成相同口径的 Hermes/训练格式。

## 如果目标是让 Qwen 自己生成全新的 T1→T2 数据

这属于另一种实验。可用 `swesmith_scaling`，将 `generate.existing_repos` 限制为参考数据实际出现的 repos，并替换模型：

```bash
python sera/main.py --config-name=swesmith_scaling \
  name=qwen_svg_new_tasks \
  distill.model.name="openai/$QWEN_MODEL_ID" \
  distill.model.url="$QWEN_BASE_URL" \
  distill.sweagent_wrapper_config.num_workers=4 \
  generate.fns_per_repo=10
```

此命令默认仍使用配置文件内完整 repo 列表，必须先按你的选样要求修改/派生该列表。生成顺序为函数采样→T1 修改及合成 PR→T2 求解→patch recall 筛选→格式化。`stage=distill_stage_two` 表示从该阶段继续执行后面的 eval/postprocess，不是从 Hugging Face 自动导入 T2；它还依赖已有实验实例文件，不能单靠这个参数重跑发布数据。

对于当前“Qwen 与数据集轨迹差异”的目标，优先采用上面的同任务 replay。

# 用 Qwen 重跑 SERA T2

用自托管 Qwen 重跑[已发布的 T2 任务](https://huggingface.co/datasets/allenai/Sera-4.6-Lite-T2)，与原始 GLM 轨迹逐任务比较。

## 1. 准备环境

在可使用 Docker 的 Linux x86_64 机器上执行，所有命令从 SERA 仓库根目录运行。模型服务可部署在另一台机器。

```bash
conda create -n sera-replay python=3.12 -y
conda activate sera-replay
git submodule update --init modules/SERA-SWE-Agent
pip install -e . -e modules/code2flow -e modules/SERA-SWE-Agent
docker info
```

填写服务地址和 `/v1/models` 返回的模型 ID；需要认证时设置真实 API key。服务必须返回结构化 `tool_calls`。

```bash
export QWEN_BASE_URL='http://YOUR_HOST:YOUR_PORT/v1'
export QWEN_MODEL_ID='YOUR_SERVED_MODEL_ID'
export OPENAI_API_KEY="${OPENAI_API_KEY:-not-needed}"
curl -sS "$QWEN_BASE_URL/models" -H "Authorization: Bearer $OPENAI_API_KEY"
```

## 2. 准备任务

先取前 100 条试跑；这不是随机评估集。输出目录必须为空。

```bash
python scripts/replay_t2.py prepare \
  --limit 100 \
  --out experiments/qwen_t2_inputs
```

## 3. 先跑一个任务

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

检查日志和 `.traj`，确认模型能调用工具。成功后删除 `--instances.slice 1`，将 `--num_workers` 改为 2 或 4，运行全部选中任务。

不要预先应用参考 patch。保留原始 `.traj` / `.pred`，直接用于比较。

## 4. 比较结果

```bash
python scripts/replay_t2.py compare \
  --references experiments/qwen_t2_inputs/references.jsonl \
  --trajs experiments/qwen_t2_raw \
  --out experiments/qwen_t2_comparison
```

查看输出目录中的 `summary.json`（汇总）和 `pairs.jsonl`（逐任务结果）：提交率、工具调用、轨迹长度及 patch 匹配情况。先确认 `initial_prompt_mismatches` 为空。

- patch recall 衡量修改行的重合度，不能代替功能测试；GLM 参考 patch 也未必正确。
- 这里的预算不保证与发布轨迹一致；推理长度只统计可见的 `<think>` 文本。

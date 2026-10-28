# 用 Qwen 重跑 SERA T2

固定 20 条[已发布 T2 任务](https://huggingface.co/datasets/allenai/Sera-4.6-Lite-T2)，用原版 SERA agent 生成 Qwen 轨迹。每个仓库一条；人工排除了明显描述错配和只有格式错误的记录，因此不是随机评估集。

在有 Docker 的 Linux x86_64 主机上，从仓库根目录执行：

```bash
git submodule update --init modules/SERA-SWE-Agent
# Python 3.12
pip install -e modules/SERA-SWE-Agent -r sera/configs/replay/requirements.txt
export QWEN_BASE_URL='http://YOUR_HOST:YOUR_PORT/v1'
export OPENAI_API_KEY="${OPENAI_API_KEY:-not-needed}"
```

服务使用原生模板、`qwen3` reasoning parser 和 `qwen3_coder` tool parser，并开放 `/tokenize`。固定模型版本及采样参数见 [qwen_128k.json](../sera/configs/replay/qwen_128k.json)：128K 总上下文，每次预留 8192 输出 token（含思考），输入最多 122880；保留历史思考、每步一个工具、115 次调用、串行执行。该预算和采样设置不保证与历史 GLM 相同。

准备任务；已有输入可直接复用。所有输出目录须为空：

```bash
python scripts/replay_t2.py prepare \
  --selection sera/configs/replay/qwen_t2_20.json \
  --out experiments/qwen_t2_inputs
```

先运行一条，检查 `trajs/` 中的退出状态与工具调用，再显式启动 20 条：

```bash
python scripts/run_qwen_t2.py --upstream "$QWEN_BASE_URL" \
  --out experiments/qwen_t2_smoke

python scripts/run_qwen_t2.py --upstream "$QWEN_BASE_URL" --all \
  --out experiments/qwen_t2_run
```

运行前按 diff 的旧 blob 校验初始代码；镜像已含参考修改时，仅反向还原并建立独立基线。镜像名后缀不能当作容器的 Git commit。

`run.json` 保存配置、依赖版本、输入哈希、镜像及基线身份；`api/` 保存每次原始请求、响应和 token ID。`.traj` 中以 `<think>` 标注思考，下一轮还原成原生 reasoning 字段；think 外的普通回答完整保留。不要正向应用参考 patch，也不要对比较轨迹运行训练清洗。

阅读原始内容，只展开换行并格式化；输出包含 `.txt`、`.json` 和一个简单选择页：

```bash
python scripts/view_t2.py \
  --references experiments/qwen_t2_inputs/references.jsonl \
  --trajs experiments/qwen_t2_run/trajs \
  --out experiments/qwen_t2_reader.html
```

只读 GLM 时省略 `--trajs`；需要查看实际 API 全记录时增加 `--api experiments/qwen_t2_run/api`。

```bash
python scripts/replay_t2.py compare \
  --references experiments/qwen_t2_inputs/references.jsonl \
  --trajs experiments/qwen_t2_run/trajs \
  --out experiments/qwen_t2_comparison
```

查看 `summary.json` 和 `pairs.jsonl`，先确认 `initial_prompt_mismatches` 为空。失败和缺失任务保留在统计中；无法识别思考边界时记为未知。Patch 重合度不能代替功能测试，GLM 参考结果也未经过完整验证。

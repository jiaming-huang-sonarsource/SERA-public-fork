# 用 Qwen 重跑 SERA T2

固定 20 条[已发布 T2 任务](https://huggingface.co/datasets/allenai/Sera-4.6-Lite-T2)，用原版 SERA agent 生成 Qwen 轨迹。每个仓库一条；人工排除了明显描述错配和只有格式错误的记录，因此不是随机评估集。

从 Mac 启动，agent 在 helix-dev 的独立 CPU sandbox 中运行，模型使用已有 Qwen 服务。本地需要 Python 3、kubectl、AWS CLI，以及可用的 `codenext-dev` 访问配置；不需要 Docker 或本地 GPU。从仓库根目录执行：

```bash
git submodule update --init modules/SERA-SWE-Agent
```

服务使用原生模板、`qwen3` reasoning parser 和 `qwen3_coder` tool parser，并开放 `/tokenize`。固定模型版本及采样参数见 [qwen_128k.json](../sera/configs/replay/qwen_128k.json)：128K 总上下文，每次预留 8192 输出 token（含思考），输入最多 122880；保留历史思考、每步一个工具、115 次调用、串行执行。该预算和采样设置不保证与历史 GLM 相同。

准备任务；已有 `experiments/qwen_t2_inputs` 可跳过：

```bash
python3 scripts/replay_t2.py prepare \
  --selection sera/configs/replay/qwen_t2_20.json \
  --out experiments/qwen_t2_inputs
```

先运行一条；完成后输出本地阅读页路径。默认生成新的输出目录，也可用 `--out` 指定空目录：

```bash
python3 scripts/run_qwen_t2_cluster.py
```

默认只运行 TinyDB。指定另一条用 `--instance-id ptyprocess_10174`；显式运行 20 条：

```bash
python3 scripts/run_qwen_t2_cluster.py --all \
  --out experiments/qwen_t2_run
```

每条任务使用原始镜像，运行前校验初始代码；镜像已含参考修改时反向还原并校验。任务串行执行，每条默认最多两小时（含准备）。结束或 Ctrl-C 后取回记录并清理本次 sandbox；本地进程意外退出时，远端资源也有到期清理时间。

输出目录中的 `reader.html` 按同一任务左右并排展示 SERA/GLM 与 Qwen，左右独立滚动；`reader/*.txt` 可直接阅读。两侧均为完整原始历史，仅格式化换行。`trajs/` 保存 `.traj` 和提交结果，`api/` 保存每次实际请求、原始响应及 token ID；`run.json`、`tasks/` 保存环境身份和运行日志。思考与普通回答均保留；不要对比较轨迹运行训练清洗。

只读已发布的 GLM 历史：

```bash
python3 scripts/view_t2.py \
  --references experiments/qwen_t2_inputs/references.jsonl \
  --out experiments/qwen_t2_reader.html
```

需要格式化 API 全记录时增加 `--api experiments/qwen_t2_run/api`。生成对比统计：

```bash
python3 scripts/replay_t2.py compare \
  --references experiments/qwen_t2_inputs/references.jsonl \
  --trajs experiments/qwen_t2_run/trajs \
  --out experiments/qwen_t2_comparison
```

查看 `summary.json` 和 `pairs.jsonl`，先确认 `initial_prompt_mismatches` 为空。失败和缺失任务保留在统计中；无法识别思考边界时记为未知。Patch 重合度不能代替功能测试，GLM 参考结果也未经过完整验证。

# AGENTS.md

## 文档与 AI 上下文

- 面向人的文档保持简洁，只保留必要步骤与关键提醒。AI 专用背景资料放在仓库外，并在此登记路径。
- 处理 Qwen T2 重跑或比较任务时，先读取 `/tmp/sera-qwen-t2-replay/ai-context.md`；其他任务无需加载。
- 处理 Qwen3.8-27B 服务时，先读取 `/private/tmp/qwen3-27b-takeover-20261028/execution/AI-CONTEXT.md`。
- 上述上下文包含详细背景与精简前文档。`/tmp` 文件可能被清理；路径失效时核实并重建上下文，再更新索引。

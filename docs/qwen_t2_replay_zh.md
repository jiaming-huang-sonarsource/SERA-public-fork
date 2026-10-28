# Replay SERA T2 with self-hosted Qwen and compare against published GLM trajectories

Replay the same T2 tasks from the published dataset. This holds the tasks, initial code, tools, and prompts fixed while comparing model-generated trajectories. Rerunning the full SVG pipeline produces new T1 changes and PR descriptions, which is useful for studying the data-generation distribution, but the resulting differences cannot be attributed solely to the T2 solver.

## Verified information

- [Dataset](https://huggingface.co/datasets/allenai/Sera-4.6-Lite-T2): 36,083 T2 records. The dataset card reports 121 SWE-smith codebases and GLM-4.6 as the teacher. This is not a subset of the SWE-bench Lite repositories.
- Pinned revision: `d7d753f319f688797b8183c6668b17670992bdeb`. The single JSONL file is approximately 9.81 GB; by default, the script streams the first 100 records without downloading the entire file.
- Fields: `messages`, `instance_id`, `rollout_patch`, `func_name`, `func_path`, `problem_statement`, `target_patch`, and `docker_image`. `messages` requires an additional `json.loads()` call. Spot-checked samples contain the original SWE-agent history and structured `tool_calls`.
- `target_patch` is the T1 reference change; `rollout_patch` is GLM's T2 change. Neither is a correct answer confirmed by independent testing. The dataset card explicitly states that patch verification was not performed for the main experiments.
- Current SERA checkout: `1ca8673bd527b164dc9eb89841c96416ec8ce1a7`; the SWE-agent submodule is pinned to `a593220bfdf9e07f7761179977b8ddf02fa981d5`.

## 1. Set up the environment on the agent host

Use a Linux x86_64 host with Docker access. The model service can run on another machine; GPUs are needed only on the model-serving host. The dataset image names explicitly include `x86_64`, so Apple Silicon requires emulation, with performance and compatibility verified separately.

Run all commands below from the SERA repository root:

```bash
conda create -n sera-replay python=3.12 -y
conda activate sera-replay
git submodule update --init modules/SERA-SWE-Agent
pip install -e . -e modules/code2flow -e modules/SERA-SWE-Agent
docker info
```

This T2 replay does not require mini-swe-agent, a GitHub mirror, rebuilding code call graphs, or rerunning T1. Both agent submodule directories in the current local checkout are empty and need initialization.

## 2. Verify that the model service supports OpenAI tools

Use the actual model ID returned by your `/v1/models` endpoint. Do not assume that "Qwen3.8 27B" corresponds to a particular public model weight name.

```bash
export QWEN_BASE_URL='http://YOUR_HOST:YOUR_PORT/v1'
export QWEN_MODEL_ID='YOUR_SERVED_MODEL_ID'
export OPENAI_API_KEY="${OPENAI_API_KEY:-not-needed}"
curl -sS "$QWEN_BASE_URL/models" -H "Authorization: Bearer $OPENAI_API_KEY"
```

Verify that `chat.completions` returns structured `message.tool_calls`, including valid JSON in `function.arguments`. Emitting `<tool_call>` in plain text is insufficient for this function-calling harness. Use this small request to check:

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

The repository's `sera/datagen/inference/launch_qwen3_models.sh` is an existing Qwen3/vLLM example using `--enable-auto-tool-choice --tool-call-parser hermes`. Your model and serving version may require a different parser or chat template; verify the structured tool calls actually returned.

**Check reasoning text separately**: the pinned version of `sweagent/agent/models.py` saves `message.content` and `tool_calls` but does not copy the commonly used `reasoning_content` field. If the service puts all reasoning in that field, the original `.traj` may omit it; `visible_think_chars=0` does not mean the model performed no reasoning. Before comparing reasoning lengths, confirm that the service preserves `<think>…</think>` in `content`, or modify the harness to explicitly record the separate reasoning field and apply consistent analysis criteria to both models.

## 3. Prepare a fixed task set

```bash
python scripts/replay_t2.py prepare \
  --limit 100 \
  --out experiments/qwen_t2_inputs
```

The outputs include:

- `instances.json`: SWE-agent input containing only the original initial prompt, image, and base commit.
- `prompt_overrides.yaml`: preserves the dataset's complete first user prompt to avoid whitespace or instruction differences introduced by reapplying templates; the system prompt is also read from the dataset.
- `references.jsonl`: the corresponding GLM trajectories, T1 target patches, GLM rollout patches, and repository information for comparison.
- `manifest.json`: source revision, selected task count, counts per repository, and input SHA256 hashes.

The script uses `SWESMITH_IMAGES` in `sera/constants.py` to map **the images in actual dataset rows** to repositories and base commits, and raises an error if no mapping exists. Extra repositories in the configuration are not added to the experiment. Inputs do not expose reference patches or sampled function locations beyond what is already present in the dataset's original problem statement, which is preserved unchanged.

By default, the script selects the first N records in source-file order for a smoke test; this is **not a random or representative evaluation set**. You can restrict selection to a repository or take up to 10 tasks per repository:

```bash
python scripts/replay_t2.py prepare \
  --repo msiemens/tinydb --limit 10 \
  --out experiments/tinydb_t2_inputs

# Scan all rows and retain up to 10 records per repository present in the data.
python scripts/replay_t2.py prepare \
  --limit 0 --per-repo 10 \
  --out experiments/qwen_t2_per_repo10
```

Filtering by repository may require reading a large portion of the source file; `--limit 0 --per-repo 10` scans the entire approximately 9.81 GB JSONL. For repeated experiments, download the pinned revision locally first, then use `--source /path/to/dataset.jsonl`. The output directory must be empty to prevent accidental reuse of old results after changing the task set. For a full replay, use `--limit 0` and reserve enough disk space for references and trajectories.

## 4. Run one task first, then the fixed task set

The command below directly invokes the repository's pinned SERA-SWE-Agent version. T2 uses the standard task-solving mode, `--pipeline False`. This fork reads `pipeline_yaml` even when the pipeline is disabled, so that argument is included as well.

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

Inspect the first task's `.traj`, `.pred`, logs, and `exit_status` to confirm that the image can be pulled, the tools are available, and the model actually executes tools. Then remove `--instances.slice 1` and set `--num_workers` to 2 or 4 to run all selected tasks. When reusing the same output directory, the harness skips existing completed trajectories according to its resume rules. Start with low concurrency for a 27B endpoint and increase it based on serving throughput and Docker capacity.

These are **controlled budgets for this trial run**: the current repository defaults to temperature=0.6 and a call limit of 115; this example additionally selects a 32K total context, up to 28K input tokens, and 4K output tokens per call. The dataset card does not provide enough configuration details to confirm that the published trajectories used identical budgets, so this cannot be claimed as an exact reproduction of historical sampling. Fix and record your own budgets before examining differences. Token lengths from different model tokenizers are not equivalent units either.

For the OpenAI provider, the pinned version does not automatically forward `max_output_tokens` as the request's `max_tokens`, so `completion_kwargs.max_tokens` is also set explicitly. Input token counts may use a LiteLLM fallback tokenizer; for strict budget comparisons, configure `agent.model.custom_tokenizer` to match the actual model weights. Set an explicit context limit on the server as well, and record the chat template, thinking setting, sampling parameters, and versions.

Solving starts from the image's original commit; **do not apply `target_patch` or `rollout_patch` beforehand**, as that would write the answer into the environment. After running, also record the image digest, `git rev-parse HEAD` inside the container, the SWE-agent version, and the actual tool schemas. An image name or tag alone does not guarantee that its contents will remain unchanged.

## 5. Compare raw trajectories

```bash
python scripts/replay_t2.py compare \
  --references experiments/qwen_t2_inputs/references.jsonl \
  --trajs experiments/qwen_t2_raw \
  --out experiments/qwen_t2_comparison
```

`summary.json` and `pairs.jsonl` contain:

- Coverage, missing-task count, Qwen `exit_status`, submission rate, and the fraction of nonempty patches.
- Assistant turns, tool calls, tool distributions and sequences, assistant character counts, and visible `<think>` character counts for both models on the same IDs with Qwen `.traj` files.
- Changed-line recall/precision against the T1 `target_patch` for both models, the fractions with recall≥0.5 and recall=1, and the fraction of Qwen patches that exactly match the GLM patches.
- Whether the first system/user prompts match the dataset verbatim. Investigate mismatches before attributing differences to the models.

The script retains failed and missing tasks. Submission and patch-match rates use all selected tasks as the denominator; tasks without trajectories count as unsuccessful. Mean lengths are compared only on the same IDs with raw trajectories available for both models. For finer analysis, aggregate `pairs.jsonl` by repository. For a formal comparison, report both micro and repository-macro results, and repeat Qwen rollouts on a fixed sample to measure sampling variability.

Recall uses the same nonempty added/deleted-line matching criteria as the repository's `compare_patch_recall`: signs and duplicate counts are preserved, while diff headers, hunks, and file locations are ignored. **Recall=1 does not imply an identical patch, let alone functional correctness**; extra changes do not reduce recall. Different implementations may also be functionally equivalent, which requires separate tests or reproduction scripts to determine. Score the GLM reference itself against `target_patch` as well, rather than assuming it is always correct.

Behavioral differences can also be manually annotated on a sample: how the agent locates relevant code, whether it reproduces the issue first, whether it runs validation, edit counts, repeated searches or commands, tool argument errors, ineffective edits, and changes to test files. The basic script reports only objectively extractable metrics; it does not automatically treat a command containing "pytest" as successful validation.

Do not run `postprocess` before comparing trajectories: the default `enforce_submit=true` filters out tasks without submissions, and `keep_only_think` changes assistant text. Keep `.traj`/`.pred` as the analysis sources; if training data is needed, generate Hermes/training-format data separately using consistent criteria.

## If the goal is to have Qwen generate entirely new T1→T2 data

This is a different experiment. Use `swesmith_scaling`, restrict `generate.existing_repos` to repositories actually present in the reference data, and replace the model:

```bash
python sera/main.py --config-name=swesmith_scaling \
  name=qwen_svg_new_tasks \
  distill.model.name="openai/$QWEN_MODEL_ID" \
  distill.model.url="$QWEN_BASE_URL" \
  distill.sweagent_wrapper_config.num_workers=4 \
  generate.fns_per_repo=10
```

By default, this command still uses the full repository list in the configuration file; first modify that list or derive a new one to match your sampling requirements. Generation proceeds as function sampling → T1 changes and synthetic PR → T2 solving → patch recall filtering → formatting. `stage=distill_stage_two` resumes from that stage and continues through the subsequent eval/postprocess steps; it does not automatically import T2 tasks from Hugging Face. It also depends on existing experiment instance files, so this argument alone cannot replay the published data.

For the current goal of comparing Qwen trajectories with dataset trajectories, use the same-task replay described above.

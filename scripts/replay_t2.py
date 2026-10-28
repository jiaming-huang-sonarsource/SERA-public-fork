#!/usr/bin/env python3
"""Prepare paired SERA T2 replays and compare their raw SWE-agent outputs.

Uses only the Python standard library. Run --help for the two subcommands.
"""

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
from statistics import mean
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
REVISION = "d7d753f319f688797b8183c6668b17670992bdeb"
DATA_URL = (
    "https://huggingface.co/datasets/allenai/Sera-4.6-Lite-T2/resolve/"
    f"{REVISION}/sera-4.6-lite-t2_36083_string_enriched.jsonl"
)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def image_map():
    # Avoid importing constants.py through the optional training dependencies.
    for node in ast.parse((ROOT / "sera/constants.py").read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SWESMITH_IMAGES" for t in node.targets
        ):
            return {v["image_name"]: dict(repo=k, **v) for k, v in ast.literal_eval(node.value).items()}
    raise ValueError("SWESMITH_IMAGES not found")


def text_content(message):
    content = message.get("content") or ""
    if isinstance(content, str):
        return content
    if any(part.get("type") != "text" for part in content):
        raise ValueError("Only text messages are supported")
    return "".join(part["text"] for part in content)


def messages(row):
    value = row["messages"]
    return json.loads(value) if isinstance(value, str) else value


def initial_prompts(row):
    history = messages(row)
    if [m["role"] for m in history[:2]] != ["system", "user"]:
        raise ValueError(f"Unexpected initial history in {row['instance_id']}")
    return text_content(history[0]), text_content(history[1])


@contextmanager
def input_lines(source):
    if source.startswith(("https://", "http://")):
        with urlopen(source, timeout=120) as response:
            yield response
    else:
        with open(source, "rb") as stream:
            yield stream


def prepare(args):
    selection_path = getattr(args, "selection", None)
    wanted = None
    if selection_path:
        selection = json.loads(Path(selection_path).read_text())
        ids = selection["instance_ids"]
        if not ids or not isinstance(ids, list) or len(ids) != len(set(ids)):
            raise ValueError("Selection must contain a nonempty list of unique instance IDs")
        if selection["revision"] != REVISION:
            raise ValueError("Selection revision does not match the pinned dataset revision")
        if args.repo or args.per_repo or (args.limit and args.limit < len(ids)):
            raise ValueError("Selection conflicts with repository filters or the task cap")
        wanted = set(ids)
    out = Path(args.out)
    # Refuse to replace an existing selection or silently change a resumed run.
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    mapping = image_map()
    selected = []
    instances = []
    counts = Counter()
    seen = set()
    system_prompt = None
    scanned = 0
    with input_lines(args.source) as stream:
        for line in stream:
            if not line.strip():
                continue
            scanned += 1
            row = json.loads(line)
            if wanted is not None and row["instance_id"] not in wanted:
                continue
            image = row["docker_image"]
            if image not in mapping:
                raise ValueError(f"Image absent from sera/constants.py: {image}")
            repo_info = mapping[image]
            repo = repo_info["repo"]
            if args.repo and repo not in args.repo:
                continue
            if args.per_repo and counts[repo] >= args.per_repo:
                continue
            iid = row["instance_id"]
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", iid) or iid in {".", ".."}:
                raise ValueError(f"Unsafe instance ID: {iid}")
            if iid in seen:
                raise ValueError(f"Duplicate instance ID: {iid}")
            seen.add(iid)
            system, user = initial_prompts(row)
            if system_prompt is not None and system_prompt != system:
                raise ValueError("Selected tasks have different system prompts")
            system_prompt = system
            # Use the published full initial prompt, preserving whitespace and instructions.
            # Reference patches and the sampled function are kept out of model-visible fields.
            instances.append({
                "instance_id": iid,
                "image_name": image,
                "repo_name": "testbed",
                "base_commit": repo_info["base_commit"],
                "problem_statement": user,
                "extra_fields": {},
            })
            selected.append(dict(row, repo=repo, base_commit=repo_info["base_commit"]))
            counts[repo] += 1
            if (wanted is not None and seen == wanted) or (args.limit and len(selected) >= args.limit):
                break
    if wanted is not None:
        if missing := wanted - seen:
            raise ValueError(f"Selected tasks absent from source: {sorted(missing)}")
        order = {iid: i for i, iid in enumerate(ids)}
        instances.sort(key=lambda item: order[item["instance_id"]])
        selected.sort(key=lambda item: order[item["instance_id"]])
    if not selected:
        raise ValueError("No matching tasks")
    instances_path = out / "instances.json"
    write_json(instances_path, instances)
    # JSON is valid YAML; SWE-agent reads and merges this as a second --config.
    write_json(out / "prompt_overrides.yaml", {"agent": {"templates": {
        "system_template": system_prompt,
        "instance_template": "{{problem_statement}}",
    }}})
    with (out / "references.jsonl").open("w") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "source": args.source,
        "published_revision": REVISION if args.source == DATA_URL else None,
        "scanned_rows": scanned,
        "selected_rows": len(selected),
        "selection": "fixed instance ID list" if wanted is not None else "first matching rows in source order; no random sampling",
        "instance_ids": [row["instance_id"] for row in selected],
        "per_repo": dict(sorted(counts.items())),
        "instances_sha256": hashlib.sha256(instances_path.read_bytes()).hexdigest(),
    }
    if selection_path:
        manifest["selection_revision"] = selection["revision"]
        manifest["selection_file_sha256"] = hashlib.sha256(Path(selection_path).read_bytes()).hexdigest()
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def changed_lines(patch):
    # Matches sera/datagen/data/eval/eval.py: retain signed, nonblank changed lines.
    return [line for line in (patch or "").splitlines()
            if not line.startswith(("diff", "---", "+++", "@@"))
            and line.startswith(("+", "-")) and line[1:].strip()]


def patch_metrics(patch, target):
    predicted, expected = Counter(changed_lines(patch)), Counter(changed_lines(target))
    overlap = sum((predicted & expected).values())
    return {
        "nonempty_patch": bool(patch and patch.strip()),
        "changed_lines": sum(predicted.values()),
        "target_line_recall": overlap / sum(expected.values()) if expected else None,
        "target_line_precision": overlap / sum(predicted.values()) if predicted else None,
        "exact_target_patch": bool(patch and target and patch.strip() == target.strip()),
    }


def trajectory_metrics(history):
    assistants = [m for m in history if m["role"] == "assistant"]
    calls = [c for m in assistants for c in (m.get("tool_calls") or [])]
    names = [c["function"]["name"] for c in calls]
    contents = [text_content(m) for m in assistants]
    reasoning = []
    assistant_chars = 0
    for message, content in zip(assistants, contents):
        explicit = message.get('reasoning')
        if explicit is None:
            explicit = message.get('reasoning_content')
        tagged = re.findall(r"<think>(.*?)</think>", content, re.S)
        reasoning.append(len(explicit) if explicit is not None else
                         sum(map(len, tagged)) if tagged else None)
        assistant_chars += len(content) + (len(explicit) if explicit is not None and not tagged else 0)
    unknown = sum(value is None for value in reasoning)
    return {
        "assistant_turns": len(assistants),
        "tool_calls": len(calls),
        "tool_counts": dict(Counter(names)),
        "tool_sequence": names,
        "assistant_chars": assistant_chars,
        "visible_think_chars": None if unknown else sum(reasoning),
        "reasoning_boundary_unknown_turns": unknown,
        "has_submit_call": "submit" in names,
    }


def compare(args):
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    traj_root = Path(args.trajs)
    combined_path = traj_root / "preds.json"
    combined = json.loads(combined_path.read_text()) if combined_path.exists() else {}
    pairs = []
    with open(args.references) as stream:
        for line in stream:
            row = json.loads(line)
            iid, target = row["instance_id"], row["target_patch"]
            baseline = dict(trajectory_metrics(messages(row)), **patch_metrics(row["rollout_patch"], target))
            traj_path = traj_root / iid / f"{iid}.traj"
            pair = {"instance_id": iid, "repo": row["repo"], "glm": baseline, "qwen": None}
            if traj_path.exists():
                traj = json.loads(traj_path.read_text())
                pred_path = traj_root / iid / f"{iid}.pred"
                pred = json.loads(pred_path.read_text()) if pred_path.exists() else combined.get(iid, {})
                patch = pred.get("model_patch") or traj.get("info", {}).get("submission") or ""
                pair["qwen"] = dict(trajectory_metrics(traj["history"]), **patch_metrics(patch, target))
                pair["qwen"]["exit_status"] = traj.get("info", {}).get("exit_status")
                pair["qwen"]["submitted"] = str(pair["qwen"]["exit_status"]).lower() == "submitted"
                pair["qwen"]["exact_glm_patch"] = bool(patch and patch.strip() == row["rollout_patch"].strip())
                pair["qwen"]["initial_prompts_match"] = (
                    [(m["role"], text_content(m)) for m in traj["history"][:2]]
                    == [(m["role"], text_content(m)) for m in messages(row)[:2]]
                )
            pairs.append(pair)
    if not pairs:
        raise ValueError("Reference file is empty")
    present = [p for p in pairs if p["qwen"] is not None]
    summary = {
        "expected_tasks": len(pairs),
        "found_trajectories": len(present),
        "missing_trajectories": len(pairs) - len(present),
        "qwen_exit_statuses": dict(Counter(p["qwen"]["exit_status"] for p in present)),
        "initial_prompt_mismatches": [p["instance_id"] for p in present if not p["qwen"]["initial_prompts_match"]],
        "rates_denominator": "all selected tasks; missing trajectories count as false",
        "qwen_submitted_rate": sum(p["qwen"]["submitted"] for p in present) / len(pairs),
        "qwen_nonempty_patch_rate": sum(p["qwen"]["nonempty_patch"] for p in present) / len(pairs),
        "qwen_exact_glm_patch_rate": sum(p["qwen"]["exact_glm_patch"] for p in present) / len(pairs),
    }
    for model in ("glm", "qwen"):
        records = [p[model] for p in present]  # Same paired IDs for both models.
        summary[model + "_paired_means"] = {
            key: mean(values) if (values := [r[key] for r in records if r[key] is not None]) else None
            for key in ("assistant_turns", "tool_calls", "assistant_chars", "visible_think_chars", "target_line_recall")
        }
        for threshold in (0.5, 1.0):
            summary[f"{model}_target_recall_ge_{threshold}_rate"] = sum(
                bool(p[model] and p[model]["nonempty_patch"]
                     and p[model]["target_line_recall"] is not None
                     and p[model]["target_line_recall"] >= threshold) for p in pairs
            ) / len(pairs)
    with (out / "pairs.jsonl").open("w") as stream:
        for pair in pairs:
            stream.write(json.dumps(pair, ensure_ascii=False) + "\n")
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Stream selected published tasks into SWE-agent inputs")
    prep.add_argument("--source", default=DATA_URL, help="Public JSONL URL or local JSONL file")
    prep.add_argument("--out", required=True)
    prep.add_argument("--limit", type=int, default=100, help="Total task cap; 0 means all")
    prep.add_argument("--repo", action="append", help="Optional exact org/repo; repeatable")
    prep.add_argument("--per-repo", type=int, default=0, help="Per-repository cap; 0 means no cap")
    prep.add_argument("--selection", help="JSON file containing a pinned revision and fixed instance_ids")
    prep.set_defaults(func=prepare)
    comp = commands.add_parser("compare", help="Compare all selected IDs, including failures and missing outputs")
    comp.add_argument("--references", required=True)
    comp.add_argument("--trajs", required=True)
    comp.add_argument("--out", required=True)
    comp.set_defaults(func=compare)
    args = parser.parse_args()
    if args.command == "prepare" and (args.limit < 0 or args.per_repo < 0):
        parser.error("Task caps must be nonnegative")
    args.func(args)


if __name__ == "__main__":
    main()

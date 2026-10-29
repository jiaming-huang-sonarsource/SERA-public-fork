#!/usr/bin/env python3
"""Run the pinned SERA agent with lossless Qwen recording; one task by default."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
from uuid import uuid4

from qwen_t2_proxy import Proxy

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / 'sera/configs/replay/qwen_128k.json'
SELECTION = ROOT / 'sera/configs/replay/qwen_t2_20.json'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def validate_inputs(inputs):
    selection = json.loads(SELECTION.read_text())
    instances = json.loads((inputs / 'instances.json').read_text())
    if [i['instance_id'] for i in instances] != selection['instance_ids']:
        raise ValueError('Prepared inputs must match the fixed 20-task selection in order')
    manifest = json.loads((inputs / 'manifest.json').read_text())
    if manifest['instances_sha256'] != sha256(inputs / 'instances.json'):
        raise ValueError('Prepared instances differ from their manifest')
    return instances


def prepare_image(image, patch, config):
    """Restore verified task preimages before the agent sees the container."""
    container = 'sera-t2-input-' + uuid4().hex
    bootstrap = (ROOT / 'scripts/t2_image.py').read_text()
    try:
        result = subprocess.check_output(
            ['docker', 'run', '-i', '--name', container, '--network', 'none',
             '--entrypoint', 'python', image, '-c', bootstrap],
            input=json.dumps({'target_patch': patch}), text=True)
        identity = json.loads(result)
        if identity['restored_preimage']:
            # SWE-ReX uses this local name as FROM when adding standalone Python.
            image = 'sera-t2-initial:' + uuid4().hex
            identity['prepared_image_id'] = subprocess.check_output(
                ['docker', 'commit',
                 '--change', 'ENTRYPOINT ' + json.dumps(config.get('Entrypoint') or []),
                 '--change', 'CMD ' + json.dumps(config.get('Cmd') or []),
                 container, image], text=True).strip()
        return image, identity
    finally:
        subprocess.run(['docker', 'rm', '-f', container], check=True, stdout=subprocess.DEVNULL)


def run_agent(settings, inputs, out, run_instances, upstream, manifest, local=False):
    """Use the same model, policy and recorder for Docker and cluster sandboxes."""
    proxy = Proxy(upstream, out / 'api', context_window=settings['context_window'])
    authorization = 'Bearer ' + os.environ.get('OPENAI_API_KEY', 'not-needed')
    status, models = proxy.request('/v1/models', authorization=authorization)
    model = next((m for m in models.get('data', []) if m['id'] == settings['served_model']), None)
    if status != 200 or model is None or model.get('max_model_len') != settings['context_window']:
        raise ValueError('Model ID or context window differs from the fixed config')
    with proxy.server(port=0) as server:
        model_config = dict(
            name='openai/' + settings['served_model'],
            api_base=f'http://127.0.0.1:{server.server_port}/v1', api_key='$OPENAI_API_KEY',
            temperature=settings['temperature'], top_p=settings['top_p'],
            per_instance_call_limit=115, per_instance_cost_limit=0, total_cost_limit=0,
            max_input_tokens=0, max_output_tokens=settings['max_output_tokens'],
            completion_kwargs=dict(max_tokens=settings['max_output_tokens'], seed=settings['seed'],
                                   parallel_tool_calls=False, presence_penalty=0.0,
                                   extra_body=settings['extra_body']),
        )
        config = out / 'model.json'
        config.write_text(json.dumps({'agent': {'model': model_config}}, indent=2) + '\n')
        deployment = (['--instances.deployment.type', 'local'] if local else
                      ['--instances.deployment.platform', 'linux/amd64'])
        command = [sys.executable, '-m', 'sweagent', 'run-batch',
                   '--config', str(ROOT / 'sera/configs/sweagent/e2e.yaml'),
                   '--config', str(inputs / 'prompt_overrides.yaml'), '--config', str(config),
                   '--instances.type', 'file', '--instances.path', str(run_instances),
                   '--instances.shuffle', 'False', *deployment, '--num_workers', '1',
                   '--pipeline', 'False', '--pipeline_yaml', str(ROOT / 'sera/configs/pipeline/default_pipeline.yaml'),
                   '--output_dir', str(out / 'trajs')]
        sources = [SETTINGS, SELECTION, ROOT / 'sera/configs/sweagent/e2e.yaml',
                   ROOT / 'sera/configs/pipeline/default_pipeline.yaml', Path(__file__),
                   ROOT / 'scripts/qwen_t2_proxy.py', ROOT / 'scripts/t2_image.py',
                   config, run_instances, *inputs.iterdir()]
        manifest.update(started_at=datetime.now(timezone.utc).isoformat(), settings=settings,
                        sweagent_revision=settings['sweagent_revision'], command=command, upstream=upstream,
                        models=models, hashes={str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): sha256(p)
                                               for p in sources if p.is_file()},
                        python=sys.version, dependencies={p.metadata['Name']: p.version for p in importlib.metadata.distributions()})
        receipt = out / 'run.json'
        receipt.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        threading.Thread(target=server.serve_forever, daemon=True).start()
        env = dict(os.environ, OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY', 'not-needed'))
        try:
            result = subprocess.run(command, cwd=ROOT, env=env)
            manifest['returncode'] = result.returncode
        finally:
            server.shutdown()
            manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
            receipt.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        return result.returncode


def run(args):
    settings = json.loads(SETTINGS.read_text())
    inputs, out = args.inputs.resolve(), args.out.resolve()
    instances = validate_inputs(inputs)
    if git('-C', 'modules/SERA-SWE-Agent', 'rev-parse', 'HEAD') != settings['sweagent_revision']:
        raise ValueError('Initialize the pinned SERA-SWE-Agent submodule')
    if git('-C', 'modules/SERA-SWE-Agent', 'status', '--porcelain', '--untracked-files=no'):
        raise ValueError('Pinned SERA-SWE-Agent has local source changes')
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Use a fresh output directory: {out}')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('Run on a Linux x86_64 Docker host for the published dataset images')
    out.mkdir(parents=True, exist_ok=True)
    snapshots = out / 'inputs'
    snapshots.mkdir()
    for name in ('instances.json', 'manifest.json', 'prompt_overrides.yaml', 'references.jsonl'):
        (snapshots / name).write_bytes((inputs / name).read_bytes())
    selected = copy.deepcopy(instances if args.all else instances[:1])
    references = {r['instance_id']: r for r in
                  (json.loads(line) for line in (inputs / 'references.jsonl').read_text().splitlines() if line.strip())}
    # Resolve the task image identities before executing any model-generated action.
    images = {}
    for instance in selected:
        image = instance['image_name']
        subprocess.run(['docker', 'pull', image], check=True)
        info = json.loads(subprocess.check_output(['docker', 'image', 'inspect', image], text=True))[0]
        if info['Architecture'] != 'amd64' or info['Os'] != 'linux':
            raise ValueError(f'Unexpected task image architecture: {image}')
        images[image] = {key: info.get(key) for key in ('Id', 'RepoDigests', 'Architecture', 'Os')}
        if not info.get('RepoDigests'):
            raise ValueError(f'No immutable digest for task image: {image}')
        instance['image_name'], baseline = prepare_image(
            info['RepoDigests'][0], references[instance['instance_id']]['target_patch'], info['Config'])
        instance['base_commit'] = baseline['base_commit']
        images[image]['initial_state'] = baseline
        images[image]['prepared_image'] = instance['image_name']
    run_instances = out / 'run-instances.json'
    run_instances.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + '\n')
    manifest = dict(repository_commit=git('rev-parse', 'HEAD'), repository_status=git('status', '--porcelain'),
                    image_identities=images, instance_ids=[i['instance_id'] for i in selected])
    return run_agent(settings, snapshots, out, run_instances, args.upstream, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', required=True, help='Qwen endpoint ending in /v1')
    parser.add_argument('--inputs', type=Path, default=ROOT / 'experiments/qwen_t2_inputs')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--all', action='store_true', help='Explicitly run all 20 selected tasks')
    args = parser.parse_args()
    try:
        return run(args)
    except (ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    sys.exit(main())

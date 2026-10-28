#!/usr/bin/env python3
"""Launch isolated CPU task sandboxes on helix-dev; one task by default."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from uuid import uuid4

from run_qwen_t2 import ROOT, SELECTION, SETTINGS, git, run_agent, sha256, validate_inputs

CONTEXT = 'arn:aws:eks:us-east-1:609803702158:cluster/helix-dev'
NAMESPACE = 'harbor-sandbox'
UPSTREAM = 'http://qwen38-27b-1d4bf0f.vllm.svc.cluster.local:8000/v1'
REMOTE = '/tmp/sera-t2'
SOURCES = [
    'scripts/run_qwen_t2.py', 'scripts/run_qwen_t2_cluster.py', 'scripts/qwen_t2_proxy.py',
    'scripts/t2_image.py', 'sera/configs/replay/qwen_128k.json',
    'sera/configs/replay/qwen_t2_20.json', 'sera/configs/replay/requirements.txt',
    'sera/configs/sweagent/e2e.yaml', 'sera/configs/pipeline/default_pipeline.yaml',
]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


class Cluster:
    def __init__(self):
        self.env = dict(os.environ, AWS_PROFILE='codenext-dev', AWS_REGION='us-east-1',
                        AWS_DEFAULT_REGION='us-east-1')

    def call(self, *args, timeout=60, **kwargs):
        return subprocess.run(
            ['kubectl', '--context', CONTEXT, f'--request-timeout={timeout}s', *args],
            env=self.env, check=True, timeout=timeout + 10, **kwargs)

    def json(self, *args, **kwargs):
        return json.loads(self.call(*args, text=True, stdout=subprocess.PIPE, **kwargs).stdout)

    def get(self, kind, name):
        value = self.call('get', kind, name, '-n', NAMESPACE, '--ignore-not-found', '-o', 'json',
                          text=True, stdout=subprocess.PIPE).stdout
        return json.loads(value) if value.strip() else None

    def execute(self, pod, *command, **kwargs):
        return self.call('exec', '-i', '-n', NAMESPACE, pod, '-c', 'main', '--', *command, **kwargs)


def sandbox_spec(name, image, timeout):
    labels = {'app.kubernetes.io/name': 'sera-qwen-t2', 'harbor.dev/session': name}
    return {'apiVersion': 'agents.x-k8s.io/v1beta1', 'kind': 'Sandbox',
            'metadata': {'name': name, 'namespace': NAMESPACE, 'labels': labels},
            'spec': {'shutdownPolicy': 'Delete',
                     'shutdownTime': (datetime.now(timezone.utc) + timedelta(seconds=timeout + 600)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                     'podTemplate': {'metadata': {'labels': labels}, 'spec': {
                         'runtimeClassName': 'gvisor', 'automountServiceAccountToken': False,
                         'restartPolicy': 'Never', 'terminationGracePeriodSeconds': 10,
                         'nodeSelector': {'kubernetes.io/arch': 'amd64'},
                         'containers': [{'name': 'main', 'image': image, 'imagePullPolicy': 'Always',
                                         'command': ['/bin/bash', '-lc', f'sleep {timeout + 600}'],
                                         'resources': {'requests': {'cpu': '1', 'memory': '4Gi', 'ephemeral-storage': '4Gi'},
                                                       'limits': {'cpu': '2', 'memory': '8Gi', 'ephemeral-storage': '12Gi'}},
                                         'securityContext': {'allowPrivilegeEscalation': False,
                                                             'capabilities': {'drop': ['ALL']},
                                                             'seccompProfile': {'type': 'RuntimeDefault'}}}]}}}}


def owned_pods(cluster, name, uid):
    pods = cluster.json('get', 'pods', '-n', NAMESPACE, '-l', f'harbor.dev/session={name}', '-o', 'json')['items']
    for pod in pods:
        if not any(owner['uid'] == uid for owner in pod['metadata'].get('ownerReferences', [])):
            raise ValueError(f'Refusing unrelated Pod: {pod["metadata"]["name"]}')
    return pods


def wait_pod(cluster, name, uid, deadline):
    while time.monotonic() < deadline:
        for pod in owned_pods(cluster, name, uid):
            if pod['status'].get('phase') in ('Failed', 'Succeeded'):
                raise RuntimeError(f'Task Pod exited during startup: {pod["metadata"]["name"]}')
            if any(c['type'] == 'Ready' and c['status'] == 'True' for c in pod['status'].get('conditions', [])):
                return pod
        time.sleep(5)
    raise TimeoutError(f'Sandbox startup timed out: {name}')


def cleanup(cluster, name, uid):
    current = cluster.get('sandbox', name)
    if current:
        if current['metadata']['uid'] != uid:
            raise ValueError(f'Refusing replacement Sandbox: {name}')
        owned_pods(cluster, name, uid)
        cluster.call('delete', 'sandbox', name, '-n', NAMESPACE, '--wait=false', stdout=subprocess.DEVNULL)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if cluster.get('sandbox', name) is None and not owned_pods(cluster, name, uid):
            return
        time.sleep(3)
    raise TimeoutError(f'Cleanup not yet complete: {name}')


def source_bundle(path, inputs, bundle):
    """Upload only code, the pinned agent, and prompt templates; no reference answers."""
    with tarfile.open(path, 'w:gz') as archive:
        for source in SOURCES:
            archive.add(ROOT / source, arcname=source)
        archive.add(inputs / 'prompt_overrides.yaml', arcname='inputs/prompt_overrides.yaml')
        archive.add(bundle, arcname='pinned-agent.bundle')


def extract_results(archive_path, destination):
    """Model-created files must never escape the fresh local results directory."""
    destination = Path(destination)
    with tarfile.open(archive_path, 'r:gz') as archive:
        for member in archive:
            relative = Path(member.name)
            if relative.is_absolute() or '..' in relative.parts or not (member.isfile() or member.isdir()):
                raise ValueError(f'Unsafe result archive entry: {member.name}')
            path = destination / relative
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, path.open('xb') as target:
                    shutil.copyfileobj(source, target)


def stop_worker(cluster, pod):
    # Stop inference before collecting an interrupted run. Never kill by a broad name.
    cluster.execute(pod, 'python', '-c', '''
import json, os, signal, time
from pathlib import Path
p = Path('/tmp/sera-t2/worker-pid.json')
if p.exists():
    pid = json.loads(p.read_text())['pid']
    cmd = Path('/proc') / str(pid) / 'cmdline'
    if cmd.exists() and b'/tmp/sera-t2/scripts/run_qwen_t2_cluster.py' in cmd.read_bytes():
        if os.getpgid(pid) != pid:
            raise RuntimeError('Unexpected worker process group')
        os.killpg(pid, signal.SIGTERM)
        time.sleep(3)
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
''', timeout=30)


SETUP = '''set -eu
cd /tmp/sera-t2
python -m pip install -q uv
uv python install 3.12
uv venv --python 3.12 /tmp/sera-runtime
git clone -q pinned-agent.bundle modules/SERA-SWE-Agent
git -C modules/SERA-SWE-Agent checkout --detach {revision}
uv pip install --python /tmp/sera-runtime/bin/python -e modules/SERA-SWE-Agent -r sera/configs/replay/requirements.txt
'''


def task(cluster, instance, reference, source, out, receipt, upstream, timeout):
    iid = instance['instance_id']
    folder = out / 'tasks' / iid
    folder.mkdir(parents=True)
    name = 'sera-t2-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-') + uuid4().hex[:8]
    state = {'instance_id': iid, 'sandbox': name, 'status': 'starting'}
    receipt['tasks'].append(state)

    def save():
        write_json(out / 'run.json', receipt)

    save()
    uid, pod_name = None, None
    deadline = time.monotonic() + timeout
    try:
        print(f'{iid}: creating {name}', flush=True)
        created = cluster.json('create', '-f', '-', '-o', 'json', input=json.dumps(sandbox_spec(name, instance['image_name'], timeout)))
        uid = state['sandbox_uid'] = created['metadata']['uid']
        save()
        pod = wait_pod(cluster, name, uid, min(deadline, time.monotonic() + 900))
        pod_name = pod['metadata']['name']
        state['pod'] = {'name': pod_name, 'uid': pod['metadata']['uid'], 'node': pod['spec']['nodeName'],
                        'image': instance['image_name'], 'image_id': pod['status']['containerStatuses'][0]['imageID']}
        save()
        # Patch text is stdin to this preflight process only, never an agent file.
        restored = cluster.execute(pod_name, 'python', '-c', (ROOT / 'scripts/t2_image.py').read_text(),
                                   input=json.dumps({'target_patch': reference['target_patch']}),
                                   text=True, stdout=subprocess.PIPE, timeout=120)
        baseline = state['initial_state'] = json.loads(restored.stdout)
        save()
        with source.open('rb') as data:
            cluster.execute(pod_name, '/bin/bash', '-lc',
                            f'mkdir -p {REMOTE}; tar --no-same-owner -xzf - -C {REMOTE}', stdin=data, timeout=180)
        initial = dict(instance, image_name='', base_commit=baseline['base_commit'])
        metadata = {'instance': initial, 'initial_state': baseline, 'pod': state['pod'],
                    'repository_commit': receipt['repository_commit'], 'source_sha256': receipt['source_sha256']}
        cluster.execute(pod_name, 'python', '-c',
                        f'import sys; open("{REMOTE}/task.json", "w").write(sys.stdin.read())',
                        input=json.dumps(metadata), text=True)
        print(f'{iid}: installing original SERA runtime (see {folder / "setup.log"})', flush=True)
        with (folder / 'setup.log').open('wb') as log:
            cluster.execute(pod_name, '/bin/bash', '-lc',
                            SETUP.format(revision=receipt['settings']['sweagent_revision']),
                            stdout=log, stderr=subprocess.STDOUT, timeout=min(900, max(1, int(deadline - time.monotonic()))))
        # Detached worker survives a brief API disconnection; the Sandbox has a deadline.
        command = shlex.join(['/tmp/sera-runtime/bin/python', '-u', f'{REMOTE}/scripts/run_qwen_t2_cluster.py',
                              '--worker', '--upstream', upstream])
        cluster.execute(pod_name, '/bin/bash', '-lc',
                        f'cd {REMOTE}; mkdir result; nohup {command} >result/agent-console.txt 2>&1 </dev/null &')
        state['status'] = 'running'
        save()
        print(f'{iid}: running; remote log: kubectl --context {CONTEXT} -n {NAMESPACE} exec {pod_name} -- tail -30 {REMOTE}/result/agent-console.txt', flush=True)
        while time.monotonic() < deadline:
            progress = cluster.execute(pod_name, 'python', '-c',
                                       'import json; from pathlib import Path; '
                                       f'p=Path("{REMOTE}/result/worker-status.json"); '
                                       f'print(p.read_text() if p.exists() else json.dumps({{"status":"running", "api_calls":len(list(Path("{REMOTE}/result/api").glob("*.json")))}}))',
                                       text=True, stdout=subprocess.PIPE)
            result = json.loads(progress.stdout)
            if result['status'] != 'running':
                state.update(result)
                break
            if result['api_calls'] != state.get('api_calls'):
                state['api_calls'] = result['api_calls']
                print(f'{iid}: {result["api_calls"]} model requests recorded', flush=True)
            time.sleep(10)
        else:
            raise TimeoutError(f'Task exceeded {timeout}s: {iid}')
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        raise
    finally:
        try:
            if pod_name:
                if state['status'] in ('failed', 'interrupted'):
                    stop_worker(cluster, pod_name)
                archive_path = folder / 'results.tar.gz'
                with archive_path.open('wb') as stream:
                    cluster.execute(pod_name, 'tar', 'czf', '-', '--ignore-failed-read', '-C', REMOTE, 'result',
                                    stdout=stream, timeout=180)
                extract_results(archive_path, folder)
                result_dir = folder / 'result'
                for category in ('trajs', 'api'):
                    directory = result_dir / category
                    if directory.exists():
                        (out / category).mkdir(exist_ok=True)
                        for item in directory.iterdir():
                            target = out / category / item.name
                            if target.exists():
                                raise ValueError(f'Refusing to overwrite results: {target}')
                            shutil.move(str(item), target)
                state['results_collected'] = result_dir.exists()
                archive_path.unlink()
        except Exception as error:
            state['collection_error'] = str(error)
            print(f'{iid}: result collection failed: {error}', file=sys.stderr, flush=True)
        finally:
            try:
                # A create response may have been lost; recover only our unique session.
                if uid is None:
                    current = cluster.get('sandbox', name)
                    if current and current['metadata'].get('labels', {}).get('harbor.dev/session') == name:
                        uid = state['sandbox_uid'] = current['metadata']['uid']
                if uid:
                    cleanup(cluster, name, uid)
                    state['cleaned_up'] = True
            except Exception as error:
                state['cleanup_error'] = str(error)
                print(f'{iid}: cleanup failed; resource expires automatically: {error}', file=sys.stderr, flush=True)
            save()
    return state.get('returncode', 1) == 0 and state.get('results_collected') and state.get('cleaned_up')


def worker(upstream):
    """Run inside the task's isolated container, never directly on the Mac."""
    os.setsid()
    write_json(ROOT / 'worker-pid.json', {'pid': os.getpid()})
    os.chdir(ROOT)
    metadata = json.loads((ROOT / 'task.json').read_text())
    out = ROOT / 'result'
    status = {'status': 'failed', 'returncode': 1}
    try:
        settings = json.loads(SETTINGS.read_text())
        if git('-C', 'modules/SERA-SWE-Agent', 'rev-parse', 'HEAD') != settings['sweagent_revision']:
            raise ValueError('Unexpected SERA-SWE-Agent revision')
        instances = ROOT / 'one-instance.json'
        write_json(instances, [metadata['instance']])
        code = run_agent(settings, ROOT / 'inputs', out, instances, upstream,
                         dict(metadata, deployment='LocalDeployment inside isolated Linux x86_64 gVisor Sandbox'), local=True)
        status.update(status='completed' if code == 0 else 'failed', returncode=code)
        iid = metadata['instance']['instance_id']
        traj = out / 'trajs' / iid / f'{iid}.traj'
        if traj.exists():
            value = json.loads(traj.read_text())
            status.update(exit_status=value['info'].get('exit_status'), steps=len(value['trajectory']))
    except BaseException as error:
        status['error'] = str(error)
        raise
    finally:
        write_json(out / 'worker-status.json', status)
    return status['returncode']


def run(args):
    settings = json.loads(SETTINGS.read_text())
    inputs = args.inputs.resolve()
    instances = validate_inputs(inputs)
    if git('-C', 'modules/SERA-SWE-Agent', 'rev-parse', 'HEAD') != settings['sweagent_revision']:
        raise ValueError('Initialize the pinned SERA-SWE-Agent submodule')
    if git('-C', 'modules/SERA-SWE-Agent', 'status', '--porcelain', '--untracked-files=no'):
        raise ValueError('Pinned SERA-SWE-Agent has local source changes')
    selected = instances if args.all else instances[:1]
    if args.instance_id:
        selected = [i for i in instances if i['instance_id'] == args.instance_id]
        if not selected:
            raise ValueError('Instance ID must belong to the fixed 20-task selection')
    references = {r['instance_id']: r for r in
                  (json.loads(line) for line in (inputs / 'references.jsonl').read_text().splitlines() if line.strip())}
    out = (args.out or ROOT / 'experiments' / datetime.now().strftime('qwen_t2_cluster_%Y%m%d_%H%M%S')).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Use a fresh output directory: {out}')
    if args.timeout < 60:
        raise ValueError('Timeout must be at least 60 seconds')
    out.mkdir(parents=True, exist_ok=True)
    cluster = Cluster()
    serving = cluster.json('get', 'deployment', 'qwen38-27b-1d4bf0f', '-n', 'vllm', '-o', 'json')
    if serving['status'].get('readyReplicas', 0) < 1:
        raise ValueError('Qwen service is not Ready')
    receipt = dict(context=CONTEXT, namespace=NAMESPACE, settings=settings,
                   started_at=datetime.now(timezone.utc).isoformat(), repository_commit=git('rev-parse', 'HEAD'),
                   repository_status=git('status', '--porcelain'), instance_ids=[i['instance_id'] for i in selected],
                   input_hashes={p.name: sha256(p) for p in inputs.iterdir() if p.is_file()},
                   serving={'uid': serving['metadata']['uid'], 'generation': serving['metadata']['generation']},
                   tasks=[])
    print(f'Output: {out}', flush=True)
    passed = True
    try:
        with tempfile.TemporaryDirectory(prefix='sera-t2-') as temporary:
            temporary = Path(temporary)
            bundle = temporary / 'pinned-agent.bundle'
            subprocess.run(['git', '-C', str(ROOT / 'modules/SERA-SWE-Agent'), 'bundle', 'create', str(bundle), 'HEAD'], check=True)
            source = temporary / 'source.tar.gz'
            source_bundle(source, inputs, bundle)
            receipt['source_sha256'] = sha256(source)
            for instance in selected:
                passed = task(cluster, instance, references[instance['instance_id']], source, out,
                              receipt, args.upstream, args.timeout) and passed
    finally:
        receipt['finished_at'] = datetime.now(timezone.utc).isoformat()
        write_json(out / 'run.json', receipt)
    if (out / 'trajs').exists():
        from view_t2 import build_reader
        build_reader(inputs / 'references.jsonl', out / 'reader.html', out / 'trajs')
        print(f'Read trajectories: {out / "reader.html"}', flush=True)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=ROOT / 'experiments/qwen_t2_inputs')
    parser.add_argument('--out', type=Path, help='Fresh local output directory; generated by default')
    parser.add_argument('--upstream', default=UPSTREAM)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--all', action='store_true', help='Explicitly run the selected 20 tasks serially')
    selection.add_argument('--instance-id', help='Run one particular task from the fixed selection')
    parser.add_argument('--timeout', type=int, default=7200, help='Per-task deadline in seconds, including setup')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        return worker(args.upstream) if args.worker else run(args)
    except KeyboardInterrupt:
        print('Interrupted; see run.json for collection and cleanup status.', file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    sys.exit(main())

import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import run_qwen_t2_cluster as runner


def archive_bytes(entries):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for name, data, kind in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return buffer.getvalue()


class ClusterRunnerTests(unittest.TestCase):
    def test_transient_status_failure_does_not_stop_detached_worker(self):
        for failure in [subprocess.CalledProcessError(1, ['kubectl']), SimpleNamespace(stdout='')]:
            with self.subTest(failure=failure):
                cluster = Mock()
                cluster.execute.side_effect = [failure, SimpleNamespace(stdout='{"status":"running","api_calls":3}')]
                with patch.object(runner.time, 'sleep') as sleep:
                    result = runner.poll_worker(cluster, 'our-pod', runner.time.monotonic() + 60)
                self.assertEqual(result, {'status': 'running', 'api_calls': 3})
                self.assertEqual(cluster.execute.call_count, 2)
                sleep.assert_called_once_with(5)

    def test_persistent_status_failure_is_bounded(self):
        cluster = Mock()
        cluster.execute.side_effect = subprocess.CalledProcessError(1, ['kubectl'])
        with patch.object(runner.time, 'sleep'), self.assertRaises(subprocess.CalledProcessError):
            runner.poll_worker(cluster, 'our-pod', runner.time.monotonic() + 60)
        self.assertEqual(cluster.execute.call_count, 6)

    def test_kubectl_always_uses_explicit_context_and_aws_profile(self):
        with patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(stdout='{}')) as run:
            runner.Cluster().call('get', 'pods')
        args, kwargs = run.call_args
        self.assertEqual(args[0][1:3], ['--context', runner.CONTEXT])
        self.assertEqual(kwargs['env']['AWS_PROFILE'], 'codenext-dev')
        self.assertEqual(kwargs['env']['AWS_REGION'], 'us-east-1')

    def test_refuse_replacement_sandbox_and_unrelated_pod(self):
        cluster = Mock()
        cluster.get.return_value = {'metadata': {'uid': 'replacement'}}
        with self.assertRaisesRegex(ValueError, 'replacement'):
            runner.cleanup(cluster, 'our-session', 'original')
        cluster.call.assert_not_called()
        cluster.get.return_value = {'metadata': {'uid': 'original'}}
        cluster.json.return_value = {'items': [{'metadata': {'name': 'other-pod', 'ownerReferences': [{'uid': 'other'}]}}]}
        with self.assertRaisesRegex(ValueError, 'unrelated'):
            runner.cleanup(cluster, 'our-session', 'original')
        cluster.call.assert_not_called()

    def test_source_upload_excludes_reference_answers_and_local_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'prompt_overrides.yaml').write_text('original templates')
            (root / 'references.jsonl').write_text('REFERENCE ANSWER')
            (root / '.env').write_text('LOCAL SECRET')
            bundle = root / 'agent.bundle'
            bundle.write_bytes(b'pinned agent')
            archive = root / 'source.tar.gz'
            runner.source_bundle(archive, root, bundle)
            with tarfile.open(archive) as source:
                names = source.getnames()
                self.assertIn('inputs/prompt_overrides.yaml', names)
                self.assertIn('pinned-agent.bundle', names)
                self.assertFalse(any('references' in name or name.endswith('.env') for name in names))

    def test_reject_archive_traversal_and_links_without_overwriting_files(self):
        for name, kind in [('../escape', tarfile.REGTYPE), ('/escape', tarfile.REGTYPE),
                           ('result/link', tarfile.SYMTYPE), ('result/link', tarfile.LNKTYPE)]:
            with self.subTest(name=name, kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                archive = root / 'archive.tar.gz'
                archive.write_bytes(archive_bytes([(name, b'', kind)]))
                with self.assertRaises(ValueError):
                    runner.extract_results(archive, root / 'out')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'archive.tar.gz'
            archive.write_bytes(archive_bytes([('result/data', b'original', tarfile.REGTYPE)]))
            runner.extract_results(archive, root / 'out')
            with self.assertRaises(FileExistsError):
                runner.extract_results(archive, root / 'out')
            self.assertEqual((root / 'out/result/data').read_bytes(), b'original')

    def test_worker_reports_startup_failure_instead_of_polling_until_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'result').mkdir()
            with patch.object(runner, 'ROOT', root), patch.object(runner.os, 'setsid', side_effect=OSError('startup failed')):
                with self.assertRaises(OSError):
                    runner.worker(runner.UPSTREAM)
            status = json.loads((root / 'result/worker-status.json').read_text())
            self.assertEqual(status, {'status': 'failed', 'returncode': 1, 'error': 'startup failed', 'api_calls': 0})

    def test_interrupt_stops_inference_collects_records_and_cleans_exact_resource(self):
        events = []
        cluster = Mock()
        cluster.json.return_value = {'metadata': {'uid': 'our-uid'}}
        pod = {'metadata': {'name': 'our-pod', 'uid': 'pod-uid'}, 'spec': {'nodeName': 'node'},
               'status': {'containerStatuses': [{'imageID': 'sha256:actual'}]}}

        def execute(pod, *command, **kwargs):
            if command[:2] == ('python', '-c') and 'restore(' in command[2]:
                return SimpleNamespace(stdout=json.dumps({'base_commit': 'verified'}))
            if command[:2] == ('python', '-c') and 'worker-status.json' in command[2]:
                raise KeyboardInterrupt()
            if command[0] == 'tar':
                events.append('collect')
                kwargs['stdout'].write(archive_bytes([('result/api/attempt.json', b'{"status":"in_flight"}', tarfile.REGTYPE)]))
            return SimpleNamespace(stdout='', returncode=0)

        cluster.execute.side_effect = execute
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            source = out / 'source.tar.gz'
            source.write_bytes(b'local sources')
            receipt = {'tasks': [], 'settings': {'sweagent_revision': 'pinned'},
                       'repository_commit': 'repo', 'source_sha256': 'source'}
            with patch.object(runner, 'wait_pod', return_value=pod), \
                 patch.object(runner, 'stop_worker', side_effect=lambda *a: events.append('stop')), \
                 patch.object(runner, 'cleanup', side_effect=lambda *a: events.append(('cleanup', a[1], a[2]))):
                with self.assertRaises(KeyboardInterrupt):
                    runner.task(cluster, {'instance_id': 'task', 'image_name': 'image'}, {'target_patch': 'answer'},
                                source, out, receipt, runner.UPSTREAM, 7200)
            self.assertEqual(events[:2], ['stop', 'collect'])
            self.assertEqual(events[2][0], 'cleanup')
            self.assertEqual(events[2][2], 'our-uid')
            self.assertEqual(json.loads((out / 'api/attempt.json').read_text())['status'], 'in_flight')
            state = json.loads((out / 'run.json').read_text())['tasks'][0]
            self.assertEqual(state['status'], 'interrupted')
            self.assertEqual(state['api_calls'], 1)
            self.assertTrue(state['cleaned_up'])


if __name__ == '__main__':
    unittest.main()

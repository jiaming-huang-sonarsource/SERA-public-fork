import subprocess
from pathlib import Path
import tempfile
import unittest

from t2_image import git, patch_files, preimage_from_hunks, restore


class InitialStateTests(unittest.TestCase):
    def setup_repo(self, root):
        git(root, 'init', '-q')
        git(root, 'config', 'user.name', 'test')
        git(root, 'config', 'user.email', 'test@localhost')
        (root / 'code.py').write_text('original\n')
        git(root, 'add', '.')
        git(root, 'commit', '-qm', 'initial')
        (root / 'code.py').write_text('changed\n')
        patch = subprocess.check_output(['git', '-C', str(root), 'diff'], text=True)
        git(root, 'commit', '-qam', 'changed')
        return patch

    def test_restore_exact_preimage_without_solution_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            patch = self.setup_repo(root)
            identity = restore(root, patch)
            self.assertTrue(identity['restored_preimage'])
            self.assertEqual((root / 'code.py').read_text(), 'original\n')
            self.assertEqual(git(root, 'rev-list', '--count', 'HEAD'), '1')
            self.assertEqual(git(root, 'status', '--porcelain'), '')
            self.assertEqual(git(root, 'remote', 'get-url', 'origin'), str(root))
            git(root, 'fetch')  # The pinned agent's unchanged reset command still works.
            self.assertFalse(restore(root, patch)['restored_preimage'])

    def test_unmatched_image_fails_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.setup_repo(root)
            original = ''.join(f'line {i}\n' for i in range(20))
            (root / 'code.py').write_text(original)
            git(root, 'commit', '-qam', 'long original')
            changed = original.replace('line 10\n', 'changed\n')
            (root / 'code.py').write_text(changed)
            patch = subprocess.check_output(['git', '-C', str(root), 'diff'], text=True)
            unrelated = changed.replace('line 0\n', 'unknown outside patch\n')
            (root / 'code.py').write_text(unrelated)
            git(root, 'commit', '-qam', 'unrelated')
            before = git(root, 'rev-parse', 'HEAD')
            with self.assertRaisesRegex(ValueError, 'neither'):
                restore(root, patch)
            self.assertEqual(git(root, 'rev-parse', 'HEAD'), before)
            self.assertEqual((root / 'code.py').read_text(), unrelated)
        with self.assertRaises(ValueError):
            patch_files('diff --git a/../escape b/../escape\nindex 111..222\n')

    def test_mixed_preimages_and_postimages_restore_only_modified_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_patch = self.setup_repo(root)
            (root / 'second.py').write_text('second original\n')
            git(root, 'add', '.')
            git(root, 'commit', '-qm', 'second original')
            (root / 'second.py').write_text('second changed\n')
            second_patch = subprocess.check_output(['git', '-C', str(root), 'diff'], text=True)
            git(root, 'checkout', '--', 'second.py')
            identity = restore(root, first_patch + second_patch)
            self.assertTrue(identity['restored_preimage'])
            self.assertEqual((root / 'code.py').read_text(), 'original\n')
            self.assertEqual((root / 'second.py').read_text(), 'second original\n')
            self.assertEqual(git(root, 'rev-list', '--count', 'HEAD'), '1')
            self.assertEqual(git(root, 'status', '--porcelain'), '')

    def test_partial_hunks_recover_only_when_whole_file_hash_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.setup_repo(root)
            original = ''.join(f'line {i}\n' for i in range(30))
            (root / 'code.py').write_text(original)
            git(root, 'commit', '-qam', 'long original')
            (root / 'code.py').write_text(original.replace('line 3\n', 'first change\n').replace('line 25\n', 'second change\n'))
            patch = subprocess.check_output(['git', '-C', str(root), 'diff'], text=True)
            (root / 'code.py').write_text(original.replace('line 3\n', 'first change\n'))
            git(root, 'commit', '-qam', 'partial image')
            identity = restore(root, patch)
            self.assertEqual((root / 'code.py').read_text(), original)
            self.assertEqual(identity['preimages_recovered_from_hunks'], ['code.py'])
            self.assertEqual(git(root, 'rev-list', '--count', 'HEAD'), '1')

    def test_hunk_recovery_rejects_incomplete_or_wrong_hash_evidence(self):
        self.assertIsNone(preimage_from_hunks('source\n', 'diff --git a/a b/a\n@@ -1,2 +1 @@\n-source\n+new\n', 'a', 'abc'))
        self.assertIsNone(preimage_from_hunks('source\n', '', 'a', 'abc'))


if __name__ == '__main__':
    unittest.main()

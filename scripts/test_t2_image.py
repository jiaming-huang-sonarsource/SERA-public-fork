import subprocess
from pathlib import Path
import tempfile
import unittest

from t2_image import git, patch_files, restore


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
            patch = self.setup_repo(root)
            (root / 'code.py').write_text('unrelated\n')
            git(root, 'commit', '-qam', 'unrelated')
            before = git(root, 'rev-parse', 'HEAD')
            with self.assertRaisesRegex(ValueError, 'neither'):
                restore(root, patch)
            self.assertEqual(git(root, 'rev-parse', 'HEAD'), before)
            self.assertEqual((root / 'code.py').read_text(), 'unrelated\n')
        with self.assertRaises(ValueError):
            patch_files('diff --git a/../escape b/../escape\nindex 111..222\n')


if __name__ == '__main__':
    unittest.main()

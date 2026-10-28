#!/usr/bin/env python3
"""Verify published patch preimages and restore them in a disposable task image."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def git(root, *args, **kwargs):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True, **kwargs).strip()


def patch_files(patch):
    files = []
    path = None
    for line in patch.splitlines():
        if line.startswith('diff --git '):
            match = re.fullmatch(r'diff --git a/(\S+) b/(\S+)', line)
            if not match or match[1] != match[2] or '..' in Path(match[1]).parts or Path(match[1]).is_absolute():
                raise ValueError('Only same-path text patches are supported')
            path = match[1]
        elif line.startswith('index '):
            match = re.match(r'index ([0-9a-f]+)\.\.([0-9a-f]+)', line)
            if not path or not match:
                raise ValueError('Patch must have original and resulting blob IDs')
            files.append((path, match[1], match[2]))
    if not files:
        raise ValueError('Patch has no verifiable blob IDs')
    return files


def restore(root, patch):
    root = Path(root)
    if not (root / '.git').is_dir() or (root / '.git').is_symlink():
        raise ValueError('Expected an isolated task image with a Git directory')
    files = patch_files(patch)
    if git(root, 'status', '--porcelain'):
        raise ValueError('Task image must start with a clean worktree')
    original_head = git(root, 'rev-parse', 'HEAD')

    def matches(column):
        for path, old, new in files:
            expected = (old, new)[column]
            file = root / path
            actual = git(root, 'hash-object', '--', path) if file.is_file() else '0' * 40
            if not actual.startswith(expected):
                return False
        return True

    restored = False
    if not matches(0):
        if not matches(1):
            raise ValueError('Image source matches neither published preimage nor postimage')
        subprocess.run(['git', '-C', str(root), 'apply', '--reverse', '--check', '-'], input=patch, text=True, check=True)
        subprocess.run(['git', '-C', str(root), 'apply', '--reverse', '-'], input=patch, text=True, check=True)
        if not matches(0):
            raise ValueError('Restored source does not match published preimage blobs')
        # Keep the solution and restoration diff out of the agent's Git history.
        shutil.rmtree(root / '.git')
        git(root, 'init', '-q')
        git(root, 'symbolic-ref', 'HEAD', 'refs/heads/sera-t2')
        git(root, 'add', '-A')
        git(root, '-c', 'user.name=SERA replay', '-c', 'user.email=replay@localhost',
            'commit', '-qm', 'Published T2 initial state',
            env=dict(os.environ, GIT_AUTHOR_DATE='2000-01-01T00:00:00+00:00',
                     GIT_COMMITTER_DATE='2000-01-01T00:00:00+00:00'))
        git(root, 'remote', 'add', 'origin', str(root))
        restored = True
    return dict(original_head=original_head, base_commit=git(root, 'rev-parse', 'HEAD'),
                restored_preimage=restored, verified_preimage_blobs={p: old for p, old, new in files})


if __name__ == '__main__':
    import sys
    print(json.dumps(restore('/testbed', json.load(sys.stdin)['target_patch'])))

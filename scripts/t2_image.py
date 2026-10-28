#!/usr/bin/env python3
"""Verify published patch preimages and restore them in a disposable task image."""

import hashlib
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


def preimage_from_hunks(source, patch, path, expected):
    """Recover known old lines, accepting only the published whole-file blob."""
    hunks, old_lines, start, count = [], None, None, None
    selected, previous_old = False, False
    for line in patch.splitlines(keepends=True):
        if line.startswith('diff --git ') or line.startswith('@@ '):
            if old_lines is not None:
                if len(old_lines) != count:
                    return None
                hunks.append((start, count, old_lines))
                old_lines = None
        if line.startswith('diff --git '):
            selected = line.rstrip('\n') == f'diff --git a/{path} b/{path}'
        elif selected and line.startswith('@@ '):
            match = re.match(r'@@ -(\d+)(?:,(\d+))? \+', line)
            if not match:
                return None
            start, count = int(match[1]) - 1, int(match[2] or 1)
            old_lines = []
        elif old_lines is not None:
            if line.startswith((' ', '-')):
                old_lines.append(line[1:])
                previous_old = True
            elif line.startswith('+'):
                previous_old = False
            elif line.startswith('\\ No newline') and old_lines and previous_old:
                old_lines[-1] = old_lines[-1].removesuffix('\n')
    if old_lines is not None:
        if len(old_lines) != count:
            return None
        hunks.append((start, count, old_lines))
    if not hunks:
        return None
    lines = source.splitlines(keepends=True)
    for start, count, old_lines in reversed(hunks):
        if start < 0 or start + count > len(lines):
            return None
        lines[start:start + count] = old_lines
    candidate = ''.join(lines).encode()
    digest = hashlib.sha1(b'blob ' + str(len(candidate)).encode() + b'\0' + candidate).hexdigest()
    return candidate if digest.startswith(expected) else None


def restore(root, patch):
    root = Path(root)
    if not (root / '.git').is_dir() or (root / '.git').is_symlink():
        raise ValueError('Expected an isolated task image with a Git directory')
    files = patch_files(patch)
    if git(root, 'status', '--porcelain'):
        raise ValueError('Task image must start with a clean worktree')
    original_head = git(root, 'rev-parse', 'HEAD')

    def blob(path):
        return git(root, 'hash-object', '--', path) if (root / path).is_file() else '0' * 40

    # A published patch may touch several files, while the image contains only
    # some postimages. Validate every file before reversing just those changes.
    postimages, recovered = [], {}
    for path, old, new in files:
        actual = blob(path)
        if actual.startswith(old):
            continue
        if not actual.startswith(new):
            candidate = None
            if (root / path).is_file():
                candidate = preimage_from_hunks((root / path).read_text(), patch, path, old)
            if candidate is None:
                raise ValueError('Image source matches neither published preimage nor postimage')
            recovered[path] = candidate
        else:
            postimages.append(path)

    restored = False
    if postimages or recovered:
        command = ['git', '-C', str(root), 'apply', '--reverse', *['--include=' + path for path in postimages]]
        if postimages:
            subprocess.run([*command, '--check', '-'], input=patch, text=True, check=True)
            subprocess.run([*command, '-'], input=patch, text=True, check=True)
        for path, data in recovered.items():
            (root / path).write_bytes(data)
        if not all(blob(path).startswith(old) for path, old, new in files):
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
                restored_preimage=restored, verified_preimage_blobs={p: old for p, old, new in files},
                preimages_recovered_from_hunks=list(recovered))


if __name__ == '__main__':
    import sys
    print(json.dumps(restore('/testbed', json.load(sys.stdin)['target_patch'])))

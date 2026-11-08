#!/usr/bin/env python3
"""Copy the pinned Sheets Python/runtime out of its read-only image.

Run inside sheets-q1.0.sif with an empty, owned output directory bound at /stage.
No FSL command suite or FreeSurfer installation is copied.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    out = Path(sys.argv[1]).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Extraction destination must be empty')
    out.mkdir(parents=True, exist_ok=True)
    prefix = Path('/opt/fsl/fsl_install_dir')
    roots = [Path('/opt/venvs/sheets'), Path('/opt/sheets-q'), Path('/opt/sheets-q-tests')]
    for source in roots:
        shutil.copytree(source, out / source.relative_to('/'), symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    stdlib = prefix / 'lib/python3.12'
    shutil.copytree(stdlib, out / stdlib.relative_to('/'), symlinks=True,
                    ignore=shutil.ignore_patterns('site-packages', '__pycache__', '*.pyc'))
    executable = prefix / 'bin/python3.12'
    target = out / executable.relative_to('/')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(executable, target)
    (target.parent / 'python3').symlink_to('python3.12')
    mrtrix = Path('/opt/mrtrix3')
    peak_tools = ['sh2peaks', 'mrconvert', 'mrinfo', 'sh2amp']
    for name in peak_tools:
        source = mrtrix/'bin'/name
        destination = out/'opt/sheets-mrtrix3/bin'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    # Keep non-glibc system dependencies private to the Sheets interpreter.
    private = out / 'opt/geometric-organization/sheets-libs'
    private.mkdir(parents=True)
    files = [executable] + [mrtrix/'bin'/name for name in peak_tools]
    for root in (roots[0], stdlib):
        for path in root.rglob('*'):
            if root == stdlib and 'site-packages' in path.relative_to(stdlib).parts:
                continue
            if path.is_file() and '.so' in path.name:
                files.append(path)
    seen, external, source_unresolved = set(), {}, {}
    glibc = re.compile(r'^(ld-linux|lib(c|m|pthread|dl|rt|util|resolv|anl)\.so)')
    while files:
        path = files.pop()
        if str(path) in seen:
            continue
        seen.add(str(path))
        result = subprocess.run(['ldd', str(path)], capture_output=True, text=True)
        if 'not found' in result.stdout:
            # Optional extension backends (for example Numba's TBB plugin) may
            # already be unavailable in the frozen source image. Preserve and
            # record that state; actual scientific imports/execution are checked.
            source_unresolved[str(path)] = [line.strip() for line in result.stdout.splitlines() if 'not found' in line]
        dependencies = re.findall(r'(?:=>\s+)?(/[^\s]+)\s+\(', result.stdout)
        for value in dependencies:
            source = Path(value)
            if source.is_relative_to(roots[0]) or source.is_relative_to(stdlib):
                continue
            if glibc.match(source.name):
                continue
            if source.is_relative_to(prefix):
                destination = out / source.relative_to('/')
            elif source.is_relative_to(mrtrix):
                destination = out / 'opt/sheets-mrtrix3' / source.relative_to(mrtrix)
            else:
                destination = private / source.name
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination, follow_symlinks=True)
            external[str(source)] = str(destination.relative_to(out))
            files.append(source)
    records = {}
    links = {}
    for path in sorted(out.rglob('*')):
        if path.is_symlink():
            links[str(path.relative_to(out))] = str(path.readlink())
        elif path.is_file():
            records[str(path.relative_to(out))] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    (out / 'runtime-files.json').write_text(json.dumps({'files': records, 'symlinks': links,
        'external_libraries': external, 'source_unresolved_optional_libraries': source_unresolved,
        'ldd_checked_files': len(seen)}, indent=2) + '\n')
    print(json.dumps({'files': len(records), 'symlinks': len(links), 'bytes': sum(r['bytes'] for r in records.values()),
                      'external_libraries': external}, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Validate compact QA regeneration with no original FOD/input/source mounts.

The completed published subject is read-only at a relocated container path.
Only a fresh evidence directory and private scratch are writable. The full
source inventory is verified before and after regeneration; rendering review
is deliberately separate from the PDF page-count and guide-text checks here.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from acceptance import CHECK_PDFS, files, sha, write_json


REPORTS = {'00_overview.pdf', '01_sheets.pdf', '02_lamellarity.pdf', '03_dispersion.pdf'}


def inventory_digest(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verify_public(root, state, inventory):
    if state.get('status') != 'complete' or not state.get('public_files'):
        raise ValueError('Subject must be completely published before QA validation')
    for relative, expected in state['public_files'].items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or inventory.get(relative) != expected:
            raise ValueError('Public inventory mismatch: '+relative)
    if any(path.is_symlink() for path in root.rglob('*')):
        raise ValueError('Published compact source must not contain symlinks')


class Runner:
    def __init__(self, args):
        self.subject, self.image, self.out = (Path(p).resolve() for p in (args.subject, args.image, args.out))
        self.scratch = Path(args.scratch).resolve()
        self.owner = self.scratch.with_name(self.scratch.name+'.owner.json')
        self.run_id = str(uuid.uuid4())
        if self.out.exists():
            raise FileExistsError('Use a fresh evidence directory')
        if self.out == Path('/nfs2') or Path('/nfs2') in self.out.parents:
            raise ValueError('/nfs2 is read-only')
        if self.out == self.subject or self.subject in self.out.parents or self.out in self.subject.parents:
            raise ValueError('Evidence must be separate from the read-only subject')
        if Path('/tmp').resolve() not in self.scratch.parents or self.scratch.exists() or self.owner.exists():
            raise ValueError('Use a fresh owned node-local /tmp scratch directory and ownership marker')
        if any(self.scratch == root or self.scratch in root.parents or root in self.scratch.parents
               for root in (self.subject, self.out)):
            raise ValueError('Scratch must be separate from inputs and persistent evidence')
        for path in (self.subject, self.image, self.out, self.scratch):
            if any(character in str(path) for character in (':', ',', '\n')):
                raise ValueError('Unsafe container bind path')
        state = json.loads((self.subject/'run.json').read_text())
        if state.get('status') != 'complete':
            raise ValueError('Wait until subject publication is complete')
        if (self.subject/'work').exists() or (self.subject/'extras/fod').exists():
            raise ValueError('Compact portability check requires removed work and no retained FOD')
        if not self.image.is_file():
            raise FileNotFoundError(self.image)
        self.runtime = shutil.which('apptainer') or shutil.which('singularity')
        if not self.runtime:
            raise FileNotFoundError('Apptainer/Singularity required')
        self.out.mkdir(parents=True)
        (self.out/'logs').mkdir()
        with self.owner.open('x') as stream:
            stream.write(json.dumps(dict(run_id=self.run_id, scratch=str(self.scratch), evidence=str(self.out)))+'\n')
        self.scratch.mkdir()
        for name in ('jobs', 'cache'):
            (self.scratch/name).mkdir()
        self.record = dict(status='running', run_id=self.run_id, started_utc=datetime.now(timezone.utc).isoformat(),
            subject=str(self.subject), image=dict(path=str(self.image), sha256=sha(self.image)),
            scratch=str(self.scratch), scratch_owner=str(self.owner), scratch_cleanup='pending; failures preserve owned scratch',
            purpose='Compact portable QA regeneration after publication; no measurement rerun',
            mounts='Read-only completed subject at /relocated-subject; fresh evidence and scratch only',
            commands=[], visual_review='Pending exact rendered-page comparison or independent review')
        self.save()

    def save(self):
        write_json(self.out/'portable_qa.json', self.record)

    def cleanup_owned_scratch(self):
        owner = json.loads(self.owner.read_text())
        if owner != dict(run_id=self.run_id, scratch=str(self.scratch), evidence=str(self.out)) or self.scratch.is_symlink():
            raise ValueError('Scratch ownership changed; refusing cleanup')
        shutil.rmtree(self.scratch)
        self.owner.unlink()
        self.record['scratch_cleanup'] = 'complete'

    def container(self, label, args):
        scratch = self.scratch/'jobs'/label
        scratch.mkdir()
        argv = ['nice', '-n', '10', self.runtime, 'exec', '--cleanenv', '--containall', '--no-home',
                '--no-mount', 'cwd,hostfs,bind-paths', '--pwd', '/']
        for source, destination, mode in [(self.subject, '/relocated-subject', 'ro'),
                                          (self.out, '/qa-evidence', 'rw'), (scratch, '/scratch', 'rw')]:
            argv += ['--bind', f'{source}:{destination}:{mode}']
        container_env = dict(TMPDIR='/scratch', MPLCONFIGDIR='/scratch/mpl', NUMBA_CACHE_DIR='/scratch/numba',
            XDG_CACHE_HOME='/scratch/cache', PYTHONDONTWRITEBYTECODE='1', NUMBA_NUM_THREADS='2',
            ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS='2', ANTS_RANDOM_SEED='1', OPENBLAS_NUM_THREADS='1',
            MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')
        for key, value in container_env.items():
            argv += ['--env', f'{key}={value}']
        argv += [str(self.image), *args]
        environment = {key: value for key, value in os.environ.items()
            if not key.startswith(('APPTAINER_', 'SINGULARITY_', 'APPTAINERENV_', 'SINGULARITYENV_'))}
        environment.update(APPTAINER_CACHEDIR=str(self.scratch/'cache'), SINGULARITY_CACHEDIR=str(self.scratch/'cache'),
                           APPTAINER_TMPDIR=str(scratch), SINGULARITY_TMPDIR=str(scratch), TMPDIR=str(scratch))
        log = self.out/'logs'/(label+'.log')
        command = dict(label=label, argv=argv, log=str(log.relative_to(self.out)))
        self.record['commands'].append(command)
        self.save()
        started = time.monotonic()
        with log.open('x') as stream:
            result = subprocess.run(argv, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=3600)
        command.update(returncode=result.returncode, seconds=time.monotonic()-started)
        self.save()
        if result.returncode:
            raise RuntimeError(label+' failed; see '+str(log))

    def run(self):
        try:
            state = json.loads((self.subject/'run.json').read_text())
            before = files(self.subject)
            verify_public(self.subject, state, before)
            write_json(self.out/'source_inventory_before.json', before)
            self.record['source_inventory_before_sha256'] = inventory_digest(before)
            self.record['original_public_inventory_verified'] = True
            self.record['source_files'] = len(before)
            self.record['source_bytes'] = sum(value['bytes'] for value in before.values())
            self.save()
            print('Published inventory verified; regenerating QA from relocated read-only compact source.', flush=True)
            self.container('regenerate', ['geometric-organization', 'qa', '/relocated-subject',
                                         '--out', '/qa-evidence/reports'])
            for label, folder in [('original-pdfs', '/relocated-subject/QA'), ('regenerated-pdfs', '/qa-evidence/reports')]:
                self.container(label, ['geo-python', '-c', CHECK_PDFS, folder, '/qa-evidence/'+label+'.json'])
            original = json.loads((self.out/'original-pdfs.json').read_text())['files']
            regenerated = json.loads((self.out/'regenerated-pdfs.json').read_text())['files']
            self.record['four_reports_exact'] = set(original) == set(regenerated) == REPORTS
            self.record['page_counts_exact'] = {key: value['pages'] for key, value in original.items()} == {
                key: value['pages'] for key, value in regenerated.items()}
            self.record['reports'] = {name: dict(original=original.get(name), regenerated=regenerated.get(name),
                regenerated_sha256=sha(self.out/'reports'/name)) for name in sorted(regenerated)}
            after = files(self.subject)
            verify_public(self.subject, state, after)
            write_json(self.out/'source_inventory_after.json', after)
            self.record['source_inventory_after_sha256'] = inventory_digest(after)
            self.record['source_inventory_unchanged'] = before == after
            self.record['status'] = 'passed' if all(self.record[key] for key in
                ('four_reports_exact', 'page_counts_exact', 'source_inventory_unchanged')) else 'failed'
            if self.record['status'] == 'passed':
                self.cleanup_owned_scratch()
        except BaseException as error:
            self.record.update(status='failed', error=type(error).__name__+': '+str(error))
            raise
        finally:
            self.record['completed_utc'] = datetime.now(timezone.utc).isoformat()
            self.save()
        print(json.dumps({key: self.record.get(key) for key in ('status', 'four_reports_exact',
              'page_counts_exact', 'source_inventory_unchanged', 'source_files', 'source_bytes')}, indent=2))
        return 0 if self.record['status'] == 'passed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True, help='Fresh owned node-local /tmp directory')
    return Runner(parser.parse_args()).run()


if __name__ == '__main__':
    raise SystemExit(main())

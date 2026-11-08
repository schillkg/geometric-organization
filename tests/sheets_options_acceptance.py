#!/usr/bin/env python3
"""Exact installed Sheets parity for the release's five advanced options.

Read a completed integrated subject and its exact prepared inputs. Run the
frozen full CLI with the same settings, then compare every scientific image
and original summary math. Inputs and subject are read-only container mounts.
All tolerances are fixed at zero; failure preserves owned node-local scratch.
The host runner requires only Python's standard library.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from acceptance import CHECK_IMAGES, sha, write_json


SETTINGS = dict(radius_mm=5.25, draws=80, seed=42, qref=.05, sample_batch_size=7)
PREFIXES = ('diagnostics/', 'tensors/', 'display/', 'legacy_SPI/')


REGIONAL = r'''
import json,sys
from pathlib import Path
sys.path[:0]=['/opt/sheets-q/Measurement','/opt/sheets-q/Code/spi_python']
import sheets_single
baseline=Path('/owned/baseline'); subject=Path('/integrated')
mapping=json.loads((subject/'provenance/layout.json').read_text())['sheets']
for kind,name in [('native','registration/JHU_labels.nii.gz'),('eroded','registration/JHU_labels_eroded.nii.gz')]:
    if name not in mapping: continue
    out=Path('/owned')/('regions-'+kind);out.mkdir()
    for item in ('q.nii.gz','wm_mask.nii.gz','diagnostics','legacy_SPI'):
        (out/item).symlink_to(baseline/item,target_is_directory=(baseline/item).is_dir())
    (out/'registration').mkdir()
    (out/'registration/JHU_labels.nii.gz').symlink_to(subject/mapping[name])
    (out/'registration/JHU_labels.csv').symlink_to(subject/mapping['registration/JHU_labels.csv'])
    meta=json.loads((baseline/'run.json').read_text());meta['labels_file']='registration/JHU_labels.nii.gz'
    sheets_single.write_summary(out,meta)
'''


def image_names(mapping):
    return {name: target for name, target in mapping.items()
            if name.endswith(('.nii', '.nii.gz'))
            and (name.startswith(PREFIXES) or name in ('q.nii.gz', 'wm_mask.nii.gz'))}


def csv_rows(path, scope=None):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    return [row for row in rows if scope is None or row['scope'] == scope]


class Runner:
    def __init__(self, args):
        self.args = args
        self.subject, self.image, self.out, self.scratch = (
            Path(value).expanduser().resolve() for value in
            (args.integrated, args.image, args.out, args.scratch))
        self.baseline_image = Path(args.sheets_image).resolve() if args.sheets_image else self.image
        fixture = Path(args.fixture).resolve()
        self.inputs = dict(fod=fixture/'reconstruction/WM_FOD.nii.gz',
                           brain_mask=fixture/'anatomy/brain_mask.nii.gz', fa=fixture/'anatomy/FA.nii.gz')
        if not self.inputs['fod'].is_file():
            self.inputs = dict(fod=fixture/'fod.nii.gz', brain_mask=fixture/'brain_mask.nii.gz', fa=fixture/'FA.nii.gz')
        self.inputs['wm_mask'] = Path(args.wm_mask).resolve() if args.wm_mask else fixture/'wm_mask.nii.gz'
        self.settings = {key: getattr(args, key) for key in SETTINGS}
        self.owner = self.scratch.with_name(self.scratch.name+'.owner.json')
        self.run_id = str(uuid.uuid4())
        if self.out.exists() or self.scratch.exists() or self.owner.exists():
            raise FileExistsError('Choose fresh evidence and node-local scratch paths')
        if Path('/tmp').resolve() not in self.scratch.parents:
            raise ValueError('Scratch must be a new directory under node-local /tmp')
        for output in (self.out, self.scratch):
            if output == Path('/nfs2') or Path('/nfs2') in output.parents:
                raise ValueError('/nfs2 is read-only')
            for source in (self.subject, fixture):
                if output == source or source in output.parents or output in source.parents:
                    raise ValueError('Output and scratch must be outside read-only input trees')
        for path in (self.subject, self.out, self.scratch, *self.inputs.values()):
            if any(c in str(path) for c in ':,\n'):
                raise ValueError('Unsafe container mount path')
        for path in (self.image, self.baseline_image, *self.inputs.values()):
            if not path.is_file(): raise FileNotFoundError(path)
        self.runtime = shutil.which('apptainer') or shutil.which('singularity')
        if not self.runtime: raise FileNotFoundError('Apptainer/Singularity required')
        state = json.loads((self.subject/'run.json').read_text())
        if state['status'] != 'complete': raise ValueError('Wait for integrated publication to complete')
        self.public = state['public_files']
        self.mapping = json.loads((self.subject/'provenance/layout.json').read_text())['sheets']
        metadata = json.loads((self.subject/self.mapping['run.json']).read_text())
        if metadata['adapter']['measurement_settings'] != self.settings:
            raise ValueError('Integrated settings do not match the requested baseline settings')
        self.input_records = {key: dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
                              for key, path in self.inputs.items()}
        for key, entry in self.input_records.items():
            if entry['sha256'] != metadata['adapter']['shared_inputs'][key]['sha256']:
                raise ValueError('Provide the exact prepared bytes used by Sheets: '+key)
        self.metadata = metadata
        self.out.mkdir(parents=True)
        (self.out/'logs').mkdir()
        self.ownership = dict(run_id=self.run_id, scratch=str(self.scratch), evidence=str(self.out))
        with self.owner.open('x') as stream: json.dump(self.ownership, stream)
        self.scratch.mkdir()
        (self.scratch/'cache').mkdir()
        self.record = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
            settings=self.settings, threads=2, subject=str(self.subject), inputs=self.input_records,
            image=dict(path=str(self.image), sha256=sha(self.image)),
            standalone_image=dict(path=str(self.baseline_image), sha256=sha(self.baseline_image)),
            runner_sha256=sha(__file__), comparator_sha256=sha(Path(__file__).with_name('acceptance.py')),
            scratch=str(self.scratch), ownership=self.ownership, scratch_cleanup='pending',
            tolerances=dict(absolute=0, relative=0, affine=0, discrete='exact', missingness='exact'),
            commands=[], checks=[], visual_review='Separate visual review required for baseline QA')
        self.save()

    def save(self):
        write_json(self.out/'sheets_options.json', self.record)

    def check(self, name, condition, details=None):
        self.record['checks'].append(dict(name=name, status='passed' if condition else 'failed', details=details))
        self.save()
        if not condition: raise AssertionError(name)

    def container(self, label, arguments, baseline=False):
        temporary = self.scratch/('tmp-'+label); temporary.mkdir()
        mounts = [(self.subject, '/integrated', 'ro'), (self.out, '/evidence', 'rw'),
                  (self.scratch, '/owned', 'rw'), (temporary, '/scratch', 'rw')]
        mounts += [(path, '/inputs/'+key+'.nii.gz', 'ro') for key, path in self.inputs.items()]
        argv = ['nice', '-n', '10', self.runtime, 'exec', '--cleanenv', '--containall', '--no-home',
                '--no-mount', 'cwd,hostfs,bind-paths', '--pwd', '/']
        for source, dest, mode in mounts: argv += ['--bind', f'{source}:{dest}:{mode}']
        for key, value in dict(TMPDIR='/scratch', MPLCONFIGDIR='/scratch/mpl', NUMBA_CACHE_DIR='/scratch/numba',
            XDG_CACHE_HOME='/scratch/cache', PYTHONDONTWRITEBYTECODE='1', NUMBA_NUM_THREADS='2',
            ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
            OMP_NUM_THREADS='1', MRTRIX_NTHREADS='2').items(): argv += ['--env', key+'='+value]
        argv += [str(self.baseline_image if baseline else self.image), *arguments]
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(('APPTAINER_', 'SINGULARITY_', 'APPTAINERENV_', 'SINGULARITYENV_'))}
        environment.update(APPTAINER_CACHEDIR=str(self.scratch/'cache'), SINGULARITY_CACHEDIR=str(self.scratch/'cache'),
                           APPTAINER_TMPDIR=str(temporary), SINGULARITY_TMPDIR=str(temporary), TMPDIR=str(temporary))
        log = self.out/'logs'/(label+'.log')
        command = dict(label=label, argv=argv, log=str(log.relative_to(self.out)))
        self.record['commands'].append(command); self.save()
        started = time.monotonic()
        with log.open('x') as stream:
            result = subprocess.run(argv, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
        command.update(returncode=result.returncode, seconds=time.monotonic()-started); self.save()
        if result.returncode: raise RuntimeError(label+' failed; see '+str(log))

    def verify_public(self):
        return all((self.subject/name).is_file() and (self.subject/name).stat().st_size == entry['bytes']
                   and sha(self.subject/name) == entry['sha256'] for name, entry in self.public.items())

    def run(self):
        try:
            self.check('integrated public inventory verified', self.verify_public())
            python = '/opt/venvs/sheets/bin/python' if self.args.sheets_image else 'sheets-python'
            argv = [python, '/opt/sheets-q/Measurement/sheets_single.py', '--fod', '/inputs/fod.nii.gz',
                '--mask', '/inputs/brain_mask.nii.gz', '--wm-mask', '/inputs/wm_mask.nii.gz', '--fa', '/inputs/fa.nii.gz',
                '--out', '/owned/baseline', '--subject', self.metadata['subject'], '--native-spacing', '--no-jhu', '--threads', '2']
            for key, value in self.settings.items(): argv += ['--'+key.replace('_', '-'), str(value)]
            self.container('standalone-sheets', argv, baseline=True)
            baseline = self.scratch/'baseline'
            baseline_meta = json.loads((baseline/'run.json').read_text())
            self.check('frozen scientific sources exact', baseline_meta['scientific_code_sha256'] == self.metadata['scientific_code_sha256'])
            self.check('standalone recipe exact', all(baseline_meta['recipe'][key] == value for key, value in self.settings.items()))
            left = image_names(self.mapping)
            right = image_names({str(path.relative_to(baseline)): str(path.relative_to(baseline))
                                 for path in baseline.rglob('*') if path.is_file()})
            self.check('all17 scientific image names exact', set(left) == set(right) and len(left) == 17,
                       dict(integrated=sorted(left), standalone=sorted(right)))
            job = dict(tolerances=self.record['tolerances'], pairs=[dict(module='sheets', relative=name,
                left='/integrated/'+left[name], right='/owned/baseline/'+name) for name in sorted(left)])
            write_json(self.out/'comparison-input.json', job)
            self.container('compare', ['geo-python', '-c', CHECK_IMAGES, '/evidence/comparison-input.json', '/evidence/comparison.json'])
            comparison = json.loads((self.out/'comparison.json').read_text())
            self.check('all17 scientific images exact', comparison['status'] == 'passed')
            q_comparison = next(row for row in comparison['comparisons'] if row['relative'] == 'q.nii.gz')
            self.check('custom radius exercises finite q', q_comparison.get('finite_values', 0) > 0,
                       dict(finite_values=q_comparison.get('finite_values', 0)))
            self.check('whole WM summary exact', csv_rows(self.subject/self.mapping['summary.csv'], 'whole_WM') == csv_rows(baseline/'summary.csv'))
            if 'registration/JHU_labels.nii.gz' in self.mapping:
                self.container('original-regional-math', [python, '-c', REGIONAL], baseline=True)
                for kind, name in [('native', 'summary.csv'), ('eroded', 'summary_eroded.csv')]:
                    if name not in self.mapping: continue
                    source = self.scratch/('regions-'+kind)/'summary.csv'
                    self.check(kind+' regional summary exact', source.read_bytes() == (self.subject/self.mapping[name]).read_bytes())
                    shutil.copy2(source, self.out/('baseline-'+kind+'-summary.csv'))
            self.check('read-only inputs unchanged', all(sha(path) == self.input_records[key]['sha256'] for key, path in self.inputs.items()))
            self.check('integrated public inventory unchanged', self.verify_public())
            saved = self.out/'baseline'; saved.mkdir()
            for name in [*sorted(right), 'summary.csv', 'run.json', 'verification.json', 'qa.pdf']:
                destination = saved/name; destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(baseline/name, destination)
            self.record['status'] = 'passed'
            if json.loads(self.owner.read_text()) != self.ownership or self.scratch.is_symlink():
                raise ValueError('Scratch ownership changed; refusing cleanup')
            shutil.rmtree(self.scratch); self.owner.unlink()
            self.record['scratch_cleanup'] = 'complete'
        except BaseException as error:
            self.record.update(status='failed', error=type(error).__name__+': '+str(error))
            raise
        finally:
            self.record['completed_utc'] = datetime.now(timezone.utc).isoformat(); self.save()
        print(json.dumps(dict(status=self.record['status'], settings=self.settings, images=17)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('image', 'fixture', 'integrated', 'out', 'scratch'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--wm-mask', type=Path)
    parser.add_argument('--sheets-image', type=Path)
    for key, default in SETTINGS.items():
        parser.add_argument('--'+key.replace('_', '-'), type=type(default), default=default)
    Runner(parser.parse_args()).run()


if __name__ == '__main__':
    main()

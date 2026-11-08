#!/usr/bin/env python3
"""Verify pinned engines and isolated Python environments without host sources."""
import hashlib
import json
from pathlib import Path
import subprocess


def command(args):
    completed = subprocess.run(args, check=True, text=True, capture_output=True)
    return completed.stdout.strip()


def main():
    roots = {'sheets': Path('/opt/sheets-q'), 'laminarity': Path('/opt/laminarity'),
             'dispersion': Path('/opt/dispersion')}
    records = {}
    for name, root in roots.items():
        manifest = json.loads((root / 'runtime_manifest.json').read_text())
        count = 0
        for row in manifest['files']:
            path = root / row['path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError(f'Pinned source changed: {path}')
            count += 1
        records[name] = {'source_files_verified': count}
    code = '''import importlib, importlib.metadata as m, json, platform, sys
names = NAMES
for name in names: importlib.import_module(name)
print(json.dumps({'python':platform.python_version(),'executable':sys.executable,'packages':{name:m.version(name) for name in PACKAGES},'imports':{name:importlib.import_module(name).__file__ for name in names}}))
'''
    environments = {
        'sheets': ('sheets-python', ['sheets_single', 'sheets_prepare', 'sheets_qa', 'verify_single_scan', 'native_grid', 'estimator', 'spi_lie', 'fodpmf_sheet_pipeline', 'sheet_roi_extract'],
                   {'numpy':'2.4.6','scipy':'1.18.0','nibabel':'5.4.2','numba':'0.66.0','matplotlib':'3.11.1','dipy':'1.12.1'}, '3.12.13'),
        'laminarity': ('laminarity-python', ['run_subject', 'pipeline', 'run', 'prepare_dwi', 'save_outputs', 'qa_report'],
                       {'numpy':'1.26.4','scipy':'1.11.4','nibabel':'5.2.1','numba':'0.59.0','matplotlib':'3.10.3'}, '3.11.7'),
        'dispersion': ('dispersion-python', ['dispersion', 'compute', 'regions', 'qa_report', 'dipy.reconst.bingham'],
                       {'numpy':'1.26.4','scipy':'1.11.4','nibabel':'5.2.1','dipy':'1.12.0'}, '3.11.7'),
    }
    for name, (wrapper, modules, versions, python) in environments.items():
        result = json.loads(command([wrapper, '-c', code.replace('NAMES', repr(modules)).replace('PACKAGES', repr(list(versions)))]))
        if result['python'] != python or result['packages'] != versions:
            raise ValueError(f'Python environment changed: {name}: {result}')
        records[name].update(result)
    for wrapper, script in [('sheets-python','/opt/sheets-q/Measurement/sheets_single.py'),
                            ('laminarity-python','/opt/laminarity/transverse/run_subject.py'),
                            ('dispersion-python','/opt/dispersion/pipeline/dispersion.py')]:
        command([wrapper, script, '--help'])
    mrtrix = command(['mrinfo', '-version'])
    ants = command(['antsRegistration', '--version'])
    sheets_peaks = command(['sheets-python', '-c', "import subprocess; print(subprocess.check_output(['sh2peaks','-version'],text=True,stderr=subprocess.STDOUT))"])
    if '3.0.4' not in mrtrix or '2.6.5' not in ants:
        raise ValueError('Shared executable versions changed')
    if '3.0.8' not in sheets_peaks:
        raise ValueError('Frozen Sheets peak-extraction version changed')
    records.update(status='passed', mrtrix=mrtrix, ants=ants, sheets_peak_extraction=sheets_peaks,
                   note='One shared MRtrix3.0.4 reconstruction/ANTs2.6.5 registration; original Sheets Python/source and MRtrix3.0.8 peak extraction retained.')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()

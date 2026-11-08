#!/usr/bin/env python3
"""Run unchanged Laminarity/Dispersion stages on one shared native extraction.

This internal adapter neither reconstructs FODs nor registers an atlas. The
subject driver owns resume, publication, QA and cleanup; all stage work stays
available until that driver has checked the complete subject.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def file_record(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))


def inventory(folder):
    return {str(p.relative_to(folder)): dict(bytes=p.stat().st_size, sha256=sha256(p))
            for p in sorted(folder.rglob('*')) if p.is_file() and not p.is_symlink()
            and p.name not in ('manifest.json', 'output_manifest.json')}


def sources(engine):
    default = '/opt/laminarity/transverse' if engine == 'laminarity' else '/opt/dispersion/pipeline'
    return Path(os.environ.get('GEO_' + engine.upper() + '_SOURCE', default)).resolve()


def python_for(engine):
    fallback = Path('/opt/laminarity/venv/bin/python')
    return os.environ.get('GEO_' + engine.upper() + '_PYTHON',
                          str(fallback) if fallback.exists() else sys.executable)


def environment(engine, work, threads):
    scratch = work/'scratch'
    scratch.mkdir()
    env = os.environ.copy()
    env.update(TMPDIR=str(scratch), MRTRIX_TMPFILE_DIR=str(scratch),
               MPLCONFIGDIR=str(scratch/'matplotlib'), NUMBA_CACHE_DIR=str(scratch/'numba'),
               XDG_CACHE_HOME=str(scratch/'cache'), MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1',
               MRTRIX_NTHREADS=str(threads), NUMBA_NUM_THREADS=str(threads),
               ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=str(threads), OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    # Never let Dispersion's dependency overlay leak into Laminarity.
    overlay = os.environ.get('GEO_DISPERSION_DEPS', '/opt/dispersion/deps')
    env['PYTHONPATH'] = str(sources(engine))
    if engine == 'dispersion' and Path(overlay).is_dir():
        env['PYTHONPATH'] += os.pathsep + overlay
    return env


def require_fresh(path):
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'Use a fresh stage destination: {path}')


def require_files(paths):
    for path in paths:
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)


def native_records(native):
    names = ('fixels/index.nii', 'fixels/directions.nii', 'fixels/mass.nii', 'fixels/peak_amplitude.nii',
             'brain_mask.nii.gz', 'total_mass.nii.gz', 'summary.json')
    require_files([native/name for name in names])
    return {name: dict(bytes=(native/name).stat().st_size, sha256=sha256(native/name)) for name in names}


def validate_native(native, fod, brain):
    """Require linkage to this exact FOD/mask and untouched extraction files."""
    record = json.loads((native/'integration_extract.json').read_text())
    if record['status'] != 'complete':
        raise ValueError('Shared native extraction is not complete')
    for key, path in (('fod', fod), ('brain_mask', brain)):
        if record['inputs'][key]['sha256'] != sha256(path):
            raise ValueError(f'Shared extraction {key} differs from the supplied input')
    if record['files'] != native_records(native):
        raise ValueError('Shared native extraction files have changed')
    return record


def resolve_laminarity_scales(min_scale_mm=None, scales_mm=None, fixed_scale_mm=None):
    """Resolve only the requested scale override; None preserves the frozen config."""
    defaults = (2.5, 3., 4., 6., 8.)
    if sum(value is not None for value in (min_scale_mm, scales_mm, fixed_scale_mm)) > 1:
        raise ValueError('Choose one scale option, not both minimum/list/fixed options')
    if fixed_scale_mm is not None:
        scales_mm = [fixed_scale_mm]
    if min_scale_mm is not None:
        if not math.isfinite(min_scale_mm) or min_scale_mm not in defaults:
            raise ValueError('--min-scale-mm must be one of 2.5, 3, 4, 6, 8 mm')
        return [value for value in defaults if value >= min_scale_mm]
    if scales_mm is None:
        return None
    scales = list(scales_mm)
    if not scales or any(not math.isfinite(value) or value <= 0 for value in scales):
        raise ValueError('--scales-mm requires finite positive integration sigmas')
    if len(set(scales)) != len(scales):
        raise ValueError('--scales-mm requires unique integration sigmas')
    # The unchanged engine uses these names for per-scale output directories.
    if len({f'{value:g}' for value in scales}) != len(scales):
        raise ValueError('--scales-mm values collide in the frozen engine scale filenames')
    return sorted(scales)


DISPERSION_DEFAULTS = dict(fit_angle=30., min_separation=35., peak_threshold=.1,
                          max_lobes=3, directions=2896, match_angle=20., match_margin=5., axis_gap=.1)


def validate_dispersion_options(**options):
    """Match the frozen public Dispersion CLI bounds, without importing its runtime."""
    if set(options) - DISPERSION_DEFAULTS.keys():
        raise ValueError('Unknown Dispersion fitting option')
    values = {**DISPERSION_DEFAULTS, **options}
    for name in ('fit_angle', 'min_separation', 'match_angle'):
        value = values[name]
        if not math.isfinite(value) or not 0 < value <= 90 or (name == 'fit_angle' and value == 90):
            raise ValueError(f'{name} must be in (0,90] degrees, with fit_angle below 90')
    if not (0 < values['peak_threshold'] <= 1 and 0 <= values['axis_gap'] < 1
            and 0 <= values['match_margin'] <= 90):
        raise ValueError('Use peak_threshold in (0,1], axis_gap in [0,1), and match_margin in [0,90]')
    if any(not isinstance(values[name], int) for name in ('max_lobes', 'directions')) or not (
            1 <= values['max_lobes'] <= 10 and values['directions'] >= 100):
        raise ValueError('Use integer max_lobes in 1-10 and directions of at least 100')
    return values


def run_stage(args):
    engine = 'laminarity' if args.stage == 'extract' else args.stage
    root = sources(engine)
    if args.stage == 'laminarity':
        resolve_laminarity_scales(getattr(args, 'min_scale_mm', None), getattr(args, 'scales_mm', None),
                                 getattr(args, 'fixed_scale_mm', None))
    elif args.stage == 'dispersion':
        validate_dispersion_options(**{name: getattr(args, name, default) for name, default in DISPERSION_DEFAULTS.items()})
    require_files([args.fod, args.brain_mask, root/'prepare_inputs.py'])
    if args.threads < 1:
        raise ValueError('Threads must be positive')
    require_fresh(args.work)
    if args.stage == 'extract':
        require_fresh(args.native)
        destination = args.native
    else:
        require_fresh(args.out)
        require_files([args.fa, args.labels, args.template_fa, args.regional_wm_mask, args.preparation_json])
        if args.template_fa and not args.labels:
            raise ValueError('--template-fa requires shared --labels')
        if args.regional_wm_mask and not args.labels:
            raise ValueError('Regional WM support requires shared labels')
        validate_native(args.native, args.fod, args.brain_mask)
        destination = args.out
    # A private stage workspace must not contain, or be inside, its outputs or inputs.
    for path in (destination, args.native, args.fod, args.brain_mask):
        if path == args.work or args.work in path.parents or path in args.work.parents:
            raise ValueError('Stage work must be separate from native inputs and outputs')
    if args.stage != 'extract':
        for path in (args.fod, args.brain_mask, args.fa, args.native, args.labels,
                     args.template_fa, args.regional_wm_mask, args.preparation_json):
            if path and (path == args.out or args.out in path.parents):
                raise ValueError('Inputs must be outside the compact module output')
    args.work.mkdir(parents=True)
    env = environment(engine, args.work, args.threads)
    state = dict(adapter='geometric-organization/native_modules.py', stage=args.stage,
                 status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                 inputs=dict(fod=file_record(args.fod), brain_mask=file_record(args.brain_mask)),
                 commands=[], source_root=str(root))
    status_path = args.work/'stage.json'
    log = args.work/'commands.log'

    def command(cmd):
        cmd = list(map(str, cmd))
        state['commands'].append(cmd)
        write_json(status_path, state)
        with log.open('a') as stream:
            stream.write('\n' + shlex.join(cmd) + '\n'); stream.flush()
            result = subprocess.run(cmd, cwd=args.work, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{Path(cmd[0]).name} failed ({result.returncode}); see {log}')

    python = python_for(engine)
    try:
        if args.stage == 'extract':
            command([python, root/'prepare_inputs.py', '--fod', args.fod, '--mask', args.brain_mask,
                     '--out', args.native, '--threads', args.threads])
            state.update(status='complete', files=native_records(args.native),
                         source_sha256=sha256(root/'prepare_inputs.py'))
            write_json(args.native/'integration_extract.json', state)
        else:
            args.out.mkdir(parents=True)
            provenance = args.out/'provenance'; provenance.mkdir()
            snapshot = provenance/'source'; snapshot.mkdir()
            for path in sorted(root.glob('*')):
                if path.is_file() and (path.suffix == '.py' or path.name in
                                      ('final_config.json', 'requirements.txt', 'reference_sources.json')):
                    shutil.copy2(path, snapshot/path.name)
            state['source_sha256'] = {p.name: sha256(p) for p in sorted(snapshot.iterdir())}
            state['inputs']['fa'] = file_record(args.fa)
            state['native'] = file_record(args.native/'integration_extract.json')
            state['native_files'] = native_records(args.native)
            for key in ('labels', 'template_fa', 'regional_wm_mask', 'preparation_json'):
                if getattr(args, key): state['inputs'][key] = file_record(getattr(args, key))
            anatomy = args.out/'anatomy'; anatomy.mkdir()
            shutil.copy2(args.fa, anatomy/'FA.nii.gz')
            shutil.copy2(args.brain_mask, anatomy/'brain_mask.nii.gz')
            # Shared preparation supplies these as NIfTI. Check its geometry
            # before spending time on either estimator, even when JHU is off.
            grid_code = ('import sys; import nibabel as n; import numpy as np; '
                         'ref=n.load(sys.argv[1]); fa=n.load(sys.argv[2]); brain=n.load(sys.argv[3]); '
                         'assert fa.shape==brain.shape==ref.shape[:3], "Shared anatomy shape mismatch"; '
                         'assert np.allclose(fa.affine,ref.affine,atol=1e-5,rtol=0) and '
                         'np.allclose(brain.affine,ref.affine,atol=1e-5,rtol=0), "Shared anatomy affine mismatch"; '
                         'mask=np.asarray(brain.dataobj); '
                         'assert np.isfinite(mask).all() and np.any(mask>0), "Invalid shared brain mask"; '
                         'assert np.isfinite(np.asarray(fa.dataobj)[mask>0]).all(), "Nonfinite observed FA"')
            command([python, '-c', grid_code, args.native/'fixels/index.nii',
                     anatomy/'FA.nii.gz', anatomy/'brain_mask.nii.gz'])
            if args.preparation_json:
                reconstruction = provenance/'reconstruction'; reconstruction.mkdir()
                shutil.copy2(args.preparation_json, reconstruction/'reconstruction.json')
            write_json(provenance/'shared_inputs.json', state['inputs'])
            write_json(provenance/'shared_native.json', json.loads((args.native/'integration_extract.json').read_text()))
            version_code = "import json,sys,importlib.metadata as m; from pathlib import Path; v={'python':sys.version}; v.update({p:m.version(p) for p in sys.argv[2:]}); Path(sys.argv[1]).write_text(json.dumps(v,indent=2)+'\\n')"
            packages = ['numpy', 'scipy', 'nibabel', 'matplotlib', 'reportlab']
            if engine == 'dispersion': packages.append('dipy')
            command([python, '-c', version_code, provenance/'software_versions.json', *packages])
            if engine == 'laminarity':
                laminarity(args, root, python, command)
            else:
                state['parameter_retention'] = dispersion(args, root, python, command)
            shutil.copy2(log, provenance/'commands.log')
            state.update(status='complete', subject=args.subject or args.out.name,
                         qa_status='pending subject-level QA', shared_registration=bool(args.labels))
            write_json(provenance/'integration_stage.json', state)
            if engine == 'laminarity':
                # Preserve its compact schema and original public-file integrity format.
                command([python, '-c', 'from pathlib import Path; import sys; from run_subject import finalize_public_manifest; finalize_public_manifest(Path(sys.argv[1]))', args.out])
            else:
                records = inventory(args.out)
                write_json(args.out/'output_manifest.json', dict(files=records, file_count=len(records),
                           total_bytes=sum(x['bytes'] for x in records.values())))
            state['output_manifest_sha256'] = sha256(args.out/'output_manifest.json')
            write_json(args.out/'manifest.json', state)
        state['completed_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(status_path, state)
        return state
    except Exception as error:
        state.update(status='failed', error=str(error))
        write_json(status_path, state)
        if args.stage != 'extract' and args.out.exists(): write_json(args.out/'manifest.json', state)
        raise


def laminarity(args, root, python, command):
    analysis = args.work/'analysis'
    config_path = root/'final_config.json'
    config = json.loads(config_path.read_text())
    scales = resolve_laminarity_scales(getattr(args, 'min_scale_mm', None), getattr(args, 'scales_mm', None),
                                     getattr(args, 'fixed_scale_mm', None))
    if scales is not None:
        base = file_record(config_path)
        config['scales'] = scales
        config['workers'] = min(2, args.threads)
        config_path = args.out/'provenance/laminarity_configuration.json'
        write_json(config_path, config)
        write_json(args.out/'provenance/laminarity_scale_settings.json', dict(
            requested=dict(min_scale_mm=getattr(args, 'min_scale_mm', None),
                           scales_mm=getattr(args, 'scales_mm', None),
                           fixed_scale_mm=getattr(args, 'fixed_scale_mm', None)),
            resolved_scales_mm=scales, scale_mode='fixed' if len(scales) == 1 else 'max-selected',
            base_configuration=base, resolved_configuration=dict(
                path=config_path.name, bytes=config_path.stat().st_size, sha256=sha256(config_path)),
            interpretation='Gaussian integration sigma in mm; response scale, not anatomical thickness.'))
    command([python, root/'pipeline.py', '--input', args.native, '--out', analysis,
             '--config', config_path, '--workers', min(2, args.threads),
             '--subject', args.subject or args.out.name])
    cmd = [python, root/'save_outputs.py', '--input', analysis, '--prepared-input', args.native,
           '--out', args.out]
    if args.save_all_scales: cmd.append('--save-all-scales')
    if args.save_native_fixels: cmd.append('--save-native-fixels')
    command(cmd)
    if args.labels:
        cmd = [python, root/'regions.py', '--input', analysis, '--fa', args.fa,
               '--brain-mask', args.brain_mask, '--out', args.out/'regions', '--atlas-dir', root/'atlas',
               '--native-labels', args.labels,
               '--subject', args.subject or args.out.name, '--threads', args.threads]
        if args.template_fa: cmd += ['--template-in-native', args.template_fa]
        if args.regional_wm_mask: cmd += ['--wm-mask', args.regional_wm_mask]
        command(cmd)
    for name in ('summary.json', 'pipeline_manifest.json', 'configuration.json'):
        shutil.copy2(analysis/name, args.out/'provenance'/('analysis_' + name))
    scales = config['scales']
    scale_text = (f'L and T at fixed integration sigma {scales[0]:g} mm; no scale selection.\n'
                  if len(scales) == 1 else
                  'Independent L/T maxima over ' + ', '.join(f'{value:g}' for value in scales) +
                  ' mm; paired T at the L-selected scale.\n')
    (args.out/'README.txt').write_text('Laminarity 1.2.0 compact outputs from shared preparation.\n'
        + scale_text +
        'No hard WM calculation mask. Native IDs belong to the shared extraction.\n'
        'Shared inputs and engine settings are recorded in provenance.\n'
        'Subject-level QA and publication are performed by the integration driver.\n')
    code = ('from pathlib import Path; from types import SimpleNamespace; import sys; '
            'from run_subject import update_guide; '
            'update_guide(Path(sys.argv[1]), SimpleNamespace(save_fod=False, '
            'no_jhu=sys.argv[2]=="1", no_report=True))')
    command([python, '-c', code, args.out, '0' if args.labels else '1'])


def dispersion(args, root, python, command):
    options = validate_dispersion_options(**{name: getattr(args, name, default)
                                            for name, default in DISPERSION_DEFAULTS.items()})
    cmd = [python, root/'compute.py', '--fod', args.fod, '--brain-mask', args.brain_mask,
           '--native-fixels', args.native/'fixels', '--out', args.out, '--threads', args.threads,
           '--save-parameters']
    if args.save_native_fixels: cmd.append('--save-native-fixels')
    for name, value in options.items():
        if value != DISPERSION_DEFAULTS[name]: cmd += ['--' + name.replace('_', '-'), value]
    if options != DISPERSION_DEFAULTS:
        write_json(args.out/'provenance/dispersion_settings.json', options)
    command(cmd)
    retention = retain_parameters(args.out, args.work, args.save_parameters)
    if args.labels:
        code = ('from pathlib import Path; import sys; from regions import run; '
                'run(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4], '
                'Path(sys.argv[5]), Path(sys.argv[6]) if sys.argv[6] else None, False)')
        command([python, '-c', code, args.out, root/'atlas', args.threads, args.subject or args.out.name,
                 args.labels, args.regional_wm_mask or ''])
        if args.template_fa:
            shutil.copy2(args.template_fa, args.out/'regions/JHU_template_FA_native.nii.gz')
            path = args.out/'regions/regions.json'; record = json.loads(path.read_text())
            record['registration']['shared_warped_template_FA'] = file_record(args.template_fa)
            write_json(path, record)
    (args.out/'README.txt').write_text('Dispersion 1.0.0 compact outputs from shared preparation.\n'
        'DIPY 1.12 Bingham fits; published ODI, DA_B, FS and angular half-widths.\n'
        'Original native AFD weights and extraction IDs; no hard WM fitting mask.\n'
        'Undefined transverse axes remain missing while circular cones retain their widths.\n'
        'Subject-level QA and publication are performed by the integration driver.\n')
    return retention


def retain_parameters(out, work, explicit):
    # Measure the actual compressed bundle. The frozen engine still computes
    # the same primary measures; this changes only optional file retention.
    parameters = out/'parameters'
    if not parameters.is_dir():
        raise FileNotFoundError('Frozen Dispersion compute did not produce its optional parameter bundle')
    parameter_bytes = sum(p.stat().st_size for p in parameters.rglob('*') if p.is_file())
    retained = explicit or parameter_bytes <= 10_000_000
    retention = dict(mode='explicit' if explicit else 'automatic',
                     compressed_bytes=parameter_bytes, automatic_limit_bytes=10_000_000,
                     retained=retained, scope='Entire optional Bingham parameter/diagnostic bundle',
                     retained_path='parameters/' if retained else None)
    if not retained:
        parameters.rename(work/'optional_parameters')
    write_json(out/'provenance/parameter_retention.json', retention)
    return retention


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='stage', required=True)
    for name in ('extract', 'laminarity', 'dispersion'):
        stage = sub.add_parser(name)
        for flag in ('fod', 'brain-mask', 'native', 'work'):
            stage.add_argument('--' + flag, type=Path, required=True)
        stage.add_argument('--threads', type=int, default=4)
        if name == 'extract': continue
        stage.add_argument('--out', type=Path, required=True)
        stage.add_argument('--fa', type=Path, required=True)
        stage.add_argument('--subject')
        for flag in ('labels', 'template-fa', 'regional-wm-mask', 'preparation-json'):
            stage.add_argument('--' + flag, type=Path)
        stage.add_argument('--save-native-fixels', action='store_true')
        stage.add_argument('--save-all-scales' if name == 'laminarity' else '--save-parameters', action='store_true')
        if name == 'laminarity':
            scale = stage.add_mutually_exclusive_group()
            scale.add_argument('--min-scale-mm', type=float, choices=(2.5, 3., 4., 6., 8.),
                               help='Lower limit of the frozen integration-scale sweep, in mm')
            scale.add_argument('--scales-mm', type=float, nargs='+',
                               help='Explicit positive integration sigmas in mm; one value gives fixed-scale output')
            scale.add_argument('--fixed-scale-mm', type=float,
                               help='Calculate one positive integration sigma, without scale selection')
        else:
            for option, default in DISPERSION_DEFAULTS.items():
                stage.add_argument('--' + option.replace('_', '-'), type=type(default), default=default)
    return p


def main():
    args = parser().parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path): setattr(args, name, value.expanduser().resolve())
    state = run_stage(args)
    print(json.dumps(dict(stage=state['stage'], status=state['status'])))


if __name__ == '__main__':
    main()

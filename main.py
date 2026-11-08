#!/usr/bin/env python3
"""One preprocessed DWI (or prepared FOD), three measures of geometric organization."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import nibabel as nib
import numpy as np

import publish
import shared

ROOT = Path(__file__).resolve().parent
VERSION = '0.1.1'
MODULES = ('sheets', 'laminarity', 'dispersion')
PYTHONS = {name: os.environ.get('GEO_'+name.upper()+'_PYTHON', '/usr/local/bin/'+name+'-python')
           for name in MODULES}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--dwi', type=Path, help='Preprocessed DWI NIfTI')
    source.add_argument('--fod', type=Path, help='Prepared isotropic Tournier07 lmax6/8 FOD; grid is preserved')
    p.add_argument('--bval', type=Path); p.add_argument('--bvec', type=Path)
    p.add_argument('--brain-mask', type=Path, help='Aligned input-grid observation mask; generated in DWI mode if omitted')
    wm = p.add_mutually_exclusive_group()
    wm.add_argument('--wm-mask', type=Path, help='Input-grid WM mask for Sheets centers only')
    wm.add_argument('--freesurfer', type=Path, help='Existing reconstruction with mri/T1.mgz and mri/aseg.mgz')
    p.add_argument('--fa', type=Path, help='Aligned FA, required with prepared FOD')
    p.add_argument('--b0', type=Path, help='Optional aligned mean b0 for prepared FOD')
    p.add_argument('--out', '--out-dir', type=Path, required=True, help='Fresh subject output folder')
    p.add_argument('--modules', default=','.join(MODULES), help='Comma-separated selection: sheets,laminarity,dispersion')
    p.add_argument('--subject', help='Label; defaults to output folder name')
    p.add_argument('--threads', type=int, default=4)
    grid = p.add_mutually_exclusive_group()
    grid.add_argument('--spacing', '--resample', type=float, help='Shared DWI isotropic grid in mm (default 1.25); prepared FOD must already match')
    grid.add_argument('--native-spacing', action='store_true', help='Use the input DWI isotropic spacing; prepared FOD always keeps its grid')
    p.add_argument('--csd-mode', choices=('auto','msmt','sscsd'), default='auto', help='Shared reconstruction; explicit modes must agree with the acquired shells')
    p.add_argument('--no-jhu', action='store_true', help='Skip the shared atlas registration and regional summaries')
    p.add_argument('--jhu-labels', type=Path, help='Reuse aligned JHU 1–50 labels on the final grid for all three measures')
    p.add_argument('--no-report', action='store_true', help='Skip PDFs; regenerate them later from the compact results with the qa command')
    p.add_argument('--resume', action='store_true', help='Resume identical interrupted run or verify completed output')
    sheets = p.add_argument_group('Sheets only — spatial q fitting and displays')
    sheets.add_argument('--sheets-radius-mm', '--sheets-length-mm', type=float, default=6.875, help='Physical neighborhood radius in mm; independent of voxel spacing')
    sheets.add_argument('--sheets-draws', '--sheets-samples', type=int, default=100, help='Number of structural FOD draws')
    sheets.add_argument('--sheets-seed', type=int, default=20261015, help='Nonnegative random seed')
    sheets.add_argument('--sheets-qref', type=float, default=.03, help='Reference q in mm^-1 for displays and tensor weights; does not change q')
    sheets.add_argument('--sheets-sample-batch-size', type=int, default=4, help='Number of draws sampled together; retain for exact reproducibility')
    laminarity = p.add_argument_group('Laminarity only — spatial integration sigma')
    scales = laminarity.add_mutually_exclusive_group()
    scales.add_argument('--laminarity-min-scale-mm', '--laminarity-min-scale', type=float, choices=(2.5,3.,4.,6.,8.), help='Lower limit of the default 2.5,3,4,6,8-mm sweep')
    scales.add_argument('--laminarity-fixed-scale-mm', '--laminarity-fixed-scale', type=float, help='Use one positive integration sigma in mm')
    scales.add_argument('--laminarity-scales-mm', type=float, nargs='+', help='Explicit positive, unique integration sigmas in mm; one value gives fixed-scale output')
    dispersion = p.add_argument_group('Dispersion only — angular Bingham fitting and matching')
    for name, default, kind, description in (
        ('fit-angle',30.,float,'Independent peak fitting cap, degrees, below 90'),
        ('min-separation',35.,float,'Minimum detected peak separation, degrees'),
        ('peak-threshold',.1,float,'Peak amplitude relative to the largest detected peak'),
        ('max-lobes',3,int,'Maximum fitted lobes; reported native slots remain strongest three'),
        ('directions',2896,int,'Hemisphere sampling directions'),
        ('match-angle',20.,float,'Maximum axial fit-to-native matching angle, degrees'),
        ('match-margin',5.,float,'Required unique-match margin in both directions, degrees'),
        ('axis-gap',.10,float,'Relative concentration gap for reporting transverse axes')):
        dispersion.add_argument('--dispersion-'+name,type=kind,default=default,help=description)
    save = p.add_argument_group('Optional retained files')
    for name, help_text in [('save-fod', 'Keep the one shared FOD'),
                            ('save-native-fixels', 'Keep the one original all-native extraction'),
                            ('save-registration', 'Keep registration transforms and intermediate images'),
                            ('sheets-save-draws', 'Keep full Sheets draw arrays'),
                            ('laminarity-save-all-scales', 'Keep full L/T scale volumes'),
                            ('dispersion-save-parameters', 'Keep Bingham parameter bundle regardless of size'),
                            ('keep-work', 'Keep all successful-run scratch for debugging')]:
        save.add_argument('--'+name, action='store_true', help=help_text)
    p.add_argument('--regional-wm-mask', type=Path, help='Final-grid support for Laminarity/Dispersion regional summaries only; does not restrict fitting or Sheets')
    p.add_argument('--version', action='version', version='Geometric organization '+VERSION)
    return p


def inputs_and_recipe(args):
    paths = {name: getattr(args, name) for name in ('dwi', 'fod', 'bval', 'bvec', 'brain_mask',
             'wm_mask', 'fa', 'b0', 'regional_wm_mask', 'jhu_labels') if getattr(args, name) is not None}
    if args.freesurfer:
        paths.update(freesurfer_t1=args.freesurfer/'mri/T1.mgz',
                     freesurfer_aseg=args.freesurfer/'mri/aseg.mgz')
    for name, path in paths.items():
        if not path.is_file(): raise ValueError(f'Missing {name}: {path}')
        if args.out == path or args.out in path.parents:
            raise ValueError('Inputs must be outside the output directory.')
    records = {k: dict(path=str(v), bytes=v.stat().st_size, sha256=publish.sha256(v)) for k, v in paths.items()}
    settings = {k: v for k, v in vars(args).items() if k not in {*paths, 'freesurfer', 'out', 'resume'}}
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in settings.items()}
    engines = {}
    for name, path in [('sheets', Path('/opt/sheets-q/runtime_manifest.json')),
                       ('laminarity', shared.LAMINARITY.parent/'runtime_manifest.json'),
                       ('dispersion', shared.DISPERSION.parent/'runtime_manifest.json')]:
        if path.exists(): engines[name] = dict(path=str(path), sha256=publish.sha256(path))
    return dict(version=VERSION, inputs=records, settings=settings,
                sources={p.name: publish.sha256(p) for p in sorted(ROOT.glob('*.py'))}, engines=engines)


def check_outputs(prepared, module_dirs, region_dir):
    reference = nib.load(prepared/'anatomy/brain_mask.nii.gz')
    checked = 0
    for name, root in module_dirs.items():
        for path in root.rglob('*.nii.gz'):
            if 'work' in path.relative_to(root).parts: continue
            image = nib.load(path)
            # Original sparse MRtrix optional exports have a different layout.
            if 'mrtrix_fixels' in path.parts: continue
            if image.shape[:3] != reference.shape[:3] or not np.allclose(image.affine, reference.affine, atol=1e-5, rtol=0):
                raise ValueError(f'{name} output grid differs from the shared analysis grid: {path}')
            checked += 1
    native = {}
    for module in ('laminarity', 'dispersion'):
        if module not in module_dirs: continue
        folder = module_dirs[module]/'fixel_maps'
        ids = np.asarray(nib.load(folder/'native_fixel_id.nii.gz').dataobj)
        afd = np.asarray(nib.load(folder/'afd.nii.gz').dataobj)
        direction = np.asarray(nib.load(folder/'fiber_direction.nii.gz').dataobj)
        if (ids.ndim != 4 or ids.shape[-1] != 3 or afd.shape != ids.shape or
                direction.shape != ids.shape+(3,) or not np.issubdtype(ids.dtype,np.integer) or
                np.any(ids < -1)):
            raise ValueError(f'Invalid reported native identity layout: {module}')
        present = np.flatnonzero(ids.ravel() >= 0)
        rows = ids.ravel()[present]
        mass = afd.ravel()[present]
        axes = direction.reshape(-1,3)[present]
        if (len(np.unique(rows)) != len(rows) or not np.isfinite(mass).all() or np.any(mass <= 0) or
                not np.isfinite(axes).all() or not np.allclose(np.linalg.norm(axes,axis=1),1,atol=1e-4,rtol=0)):
            raise ValueError(f'Invalid native row IDs, AFD or directions: {module}')
        native[module] = (rows,mass,axes,present//3)
    identity = dict(shared_reported_maps=True, modules=list(native))
    if len(native) == 2:
        left,right = native['laminarity'],native['dispersion']
        common,li,ri = np.intersect1d(left[0],right[0],assume_unique=True,return_indices=True)
        left_axes,right_axes=left[2][li],right[2][ri]
        left_axes=left_axes/np.linalg.norm(left_axes,axis=1,keepdims=True)
        right_axes=right_axes/np.linalg.norm(right_axes,axis=1,keepdims=True)
        signs=np.where(np.sum(left_axes*right_axes,axis=1)<0,-1,1)
        if (not np.array_equal(left[1][li],right[1][ri]) or
                not np.array_equal(left[3][li],right[3][ri]) or
                not np.allclose(left_axes,right_axes*signs[:,None],atol=1e-6,rtol=0)):
            raise ValueError('Same native ID has inconsistent AFD, axial direction or voxel across modules')
        equal = all(publish.equal_image(module_dirs['laminarity']/'fixel_maps'/name,
                                       module_dirs['dispersion']/'fixel_maps'/name) for name in publish.IDENTITY)
        identity.update(shared_reported_maps=equal, common_native_fixels=len(common),
                        laminarity_only_fixels=len(left[0])-len(common), dispersion_only_fixels=len(right[0])-len(common))
        if not equal:
            provenance = [json.loads((module_dirs[name]/'provenance/shared_native.json').read_text()) for name in native]
            signatures = [dict(files=record['files'],inputs={key:record['inputs'][key]['sha256']
                          for key in ('fod','brain_mask')}) for record in provenance]
            if any(record.get('status') != 'complete' for record in provenance) or signatures[0] != signatures[1]:
                raise ValueError('Different reported sets require one verified shared native extraction')
            identity['note'] = ('Frozen engines can differ at the 10% AFD boundary because their selection arithmetic '
                                'uses different precision. Original reported maps remain separate; common IDs agree. '
                                'No slot-based correspondence or estimator change is imposed.')
    return dict(status='passed', analysis_shape=list(reference.shape),
                affine=reference.affine.tolist(), spacing_mm=list(map(float, reference.header.get_zooms())),
                aligned_images_checked=checked,
                native_identity=identity,
                shared_fod=True, shared_JHU=region_dir is not None,
                interpretation='Software/grid integrity, not visual or biological validation')


def finalize(out, state, keep_work):
    work = out/'work'
    records = state['public_files']
    staged = work/'public'
    for relative, record in records.items():
        target = out/relative
        if target.exists():
            publish.verify(out, {relative: record})
        else:
            publish.verify(staged, {relative: record})
            target.parent.mkdir(parents=True, exist_ok=True)
            (staged/relative).replace(target)
    publish.verify(out, records)
    cleanup = out/('.cleanup-'+state['run_id'])
    cleanup_marker = cleanup.with_name(cleanup.name+'.json')
    ownership = dict(run_id=state['run_id'], directory=cleanup.name)
    if cleanup_marker.exists() and json.loads(cleanup_marker.read_text()) != ownership:
        raise ValueError('Cleanup ownership changed; refusing deletion.')
    if not keep_work and work.exists():
        marker = work/'.owner.json'
        if not marker.is_file() or json.loads(marker.read_text()).get('run_id') != state['run_id']:
            raise ValueError('Work ownership changed; refusing cleanup.')
        if cleanup.exists(): raise ValueError('Both active and cleanup work exist; inspect before resuming.')
        write_json(cleanup_marker, ownership)
        work.rename(cleanup)
    if not keep_work and cleanup.exists():
        if not cleanup_marker.is_file() or json.loads(cleanup_marker.read_text()) != ownership:
            raise ValueError('Cleanup ownership marker is absent or changed; refusing deletion.')
        shutil.rmtree(cleanup)
    if not keep_work and cleanup_marker.exists(): cleanup_marker.unlink()
    state.update(status='complete', completed_utc=state.get('completed_utc', now()), work_retained=keep_work)
    state.pop('error', None)
    for step in state['stages'].values(): step.pop('files', None)
    write_json(out/'run.json', state)


def _run(args, stack):
    validate_measurement_options(args)
    args.out = args.out.expanduser().resolve()
    if args.out == Path('/nfs2') or Path('/nfs2') in args.out.parents:
        raise ValueError('/nfs2 is read-only; choose an output location elsewhere.')
    args.modules = [name for name in MODULES if name in args.modules.split(',')]
    for key, value in vars(args).items():
        if isinstance(value, Path): setattr(args, key, value.expanduser().resolve())
    args.subject = args.subject or args.out.name
    if len(args.subject) > 100 or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.subject):
        raise ValueError('Subject must be at most 100 characters, start with a letter/digit and contain only letters, digits, _, . or -; use --subject if the folder name differs.')
    out = args.out
    recipe = inputs_and_recipe(args)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError('Choose a fresh output directory or use --resume.')
    if args.resume and not (out/'run.json').is_file(): raise ValueError('--resume requires this pipeline\'s run.json.')
    out.mkdir(parents=True, exist_ok=True)
    lock = stack.enter_context((out/'.run.lock').open('a'))
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.resume:
        state = json.loads((out/'run.json').read_text())
        if state['recipe'] != recipe: raise ValueError('Inputs, settings, or source changed; choose a fresh output directory.')
        if state.get('publication_ready'):
            finalize(out, state, args.keep_work)
            print(f'Complete; saved files verified: {out}', flush=True)
            return
    else:
        state = dict(version=VERSION, run_id=str(uuid.uuid4()), subject=args.subject,
                     recipe=recipe, started_utc=now(), status='running', stages={})
    work = out/'work'
    if work.exists():
        marker = work/'.owner.json'
        if not marker.is_file() or json.loads(marker.read_text()).get('run_id') != state['run_id']:
            raise ValueError('Existing work does not belong to this run.')
    else:
        work.mkdir(); write_json(work/'.owner.json', {'run_id': state['run_id']})
    temp = work/'tmp'; temp.mkdir(exist_ok=True)
    os.environ.update(TMPDIR=str(temp), MPLCONFIGDIR=str(temp/'matplotlib'),
                      NUMBA_CACHE_DIR=str(temp/'numba'), XDG_CACHE_HOME=str(temp/'cache'),
                      MRTRIX_TMPFILE_DIR=str(temp),
                      PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      OMP_NUM_THREADS='1', NUMBA_NUM_THREADS=str(args.threads),
                      ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=str(args.threads), ANTS_RANDOM_SEED='1')
    log = work/'commands.log'

    def record(): write_json(out/'run.json', state)

    def command(argv, cwd=None, env=None):
        argv = list(map(str, argv))
        start = time.monotonic()
        with log.open('a') as f:
            f.write(json.dumps(dict(time=now(), command=argv, cwd=str(cwd or work)))+'\n'); f.flush()
            result = subprocess.run(argv, cwd=cwd or work, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True)
            f.write(result.stdout); f.write(result.stderr)
            f.write(json.dumps(dict(returncode=result.returncode, seconds=time.monotonic()-start))+'\n')
        if result.returncode:
            raise RuntimeError(f'Command failed ({result.returncode}): {argv[0]}\n{result.stderr[-3000:]}\nSee {log}')
        return result.stdout

    def stage(name, destinations, callback):
        prior = state['stages'].get(name)
        if prior and prior.get('status') == 'complete':
            publish.verify(work, prior['files'])
            print(name+': verified', flush=True)
            return
        # Preserve interrupted products while retrying only this run's stage.
        for path in destinations:
            if path.exists():
                failed = work/'failures'/f'{name}-{time.time_ns()}'/path.name
                failed.parent.mkdir(parents=True, exist_ok=True); path.rename(failed)
        state['stages'][name] = dict(status='running', started_utc=now()); record()
        print(name+': running', flush=True); start = time.monotonic()
        callback()
        files = {}
        for path in destinations:
            for relative, entry in publish.inventory(path).items():
                files[str(path.relative_to(work)/relative)] = entry
        if not files: raise RuntimeError(name+' produced no files.')
        state['stages'][name].update(status='complete', seconds=time.monotonic()-start, files=files)
        record(); print(name+': complete', flush=True)

    prepared = work/'prepared'
    region_dir = None if args.no_jhu else work/'regions'
    module_dirs = {name: work/'modules'/name for name in args.modules}
    try:
        state['status'] = 'running'; record()
        stage('prepare', [prepared, work/'preparation_work'],
              lambda: shared.prepare(args, prepared, work/'preparation_work', command))
        state['preparation'] = json.loads((prepared/'reconstruction/reconstruction.json').read_text()); record()
        if 'sheets' in args.modules:
            stage('WM_alignment', [work/'wm'], lambda: shared.wm_domain(args, prepared, work/'wm', command))
        if region_dir:
            stage('JHU_registration', [region_dir], lambda: shared.jhu(prepared, region_dir, args.threads, args.jhu_labels))
        common = ['--fod', prepared/'reconstruction/WM_FOD.nii.gz',
                  '--brain-mask', prepared/'anatomy/brain_mask.nii.gz', '--threads', args.threads]
        if any(m in args.modules for m in ('laminarity', 'dispersion')):
            stage('native_extraction', [work/'native', work/'extract_work'],
                  lambda: command([PYTHONS['laminarity'], ROOT/'native_modules.py', 'extract', *common,
                                   '--native', work/'native', '--work', work/'extract_work']))
        for name in args.modules:
            argv = [PYTHONS[name], ROOT/('sheets_module.py' if name == 'sheets' else 'native_modules.py')]
            if name != 'sheets': argv += [name]
            argv += [*common, '--fa', prepared/'anatomy/FA.nii.gz', '--out', module_dirs[name],
                     '--subject', args.subject, '--preparation-json', prepared/'reconstruction/reconstruction.json']
            if name == 'sheets':
                argv += ['--wm-mask', work/'wm/wm_mask.nii.gz']
                for public, internal in [('radius_mm','radius-mm'),('draws','draws'),('seed','seed'),('qref','qref'),('sample_batch_size','sample-batch-size')]:
                    argv += ['--'+internal, getattr(args,'sheets_'+public)]
                if (prepared/'anatomy/mean_b0.nii.gz').exists(): argv += ['--b0', prepared/'anatomy/mean_b0.nii.gz']
                if args.sheets_save_draws: argv += ['--save-draws']
            else:
                argv += ['--native', work/'native', '--work', work/(name+'_work')]
                if args.regional_wm_mask: argv += ['--regional-wm-mask', args.regional_wm_mask]
                if name == 'laminarity' and args.laminarity_save_all_scales: argv += ['--save-all-scales']
                if name == 'laminarity':
                    if args.laminarity_min_scale_mm is not None: argv += ['--min-scale-mm',args.laminarity_min_scale_mm]
                    if args.laminarity_fixed_scale_mm is not None: argv += ['--fixed-scale-mm',args.laminarity_fixed_scale_mm]
                    if args.laminarity_scales_mm is not None: argv += ['--scales-mm',*args.laminarity_scales_mm]
                if name == 'dispersion' and args.dispersion_save_parameters: argv += ['--save-parameters']
                if name == 'dispersion':
                    from native_modules import DISPERSION_DEFAULTS
                    for key in DISPERSION_DEFAULTS: argv += ['--'+key.replace('_','-'),getattr(args,'dispersion_'+key)]
            if region_dir:
                argv += ['--labels', region_dir/'JHU_labels_native.nii.gz']
                if (region_dir/'JHU_template_FA_native.nii.gz').is_file(): argv += ['--template-fa',region_dir/'JHU_template_FA_native.nii.gz']
                if name == 'sheets': argv += ['--eroded-labels', region_dir/'JHU_labels_eroded_1voxel.nii.gz', '--labels-table', region_dir/'labels.csv']
            destinations = [module_dirs[name]] + ([] if name == 'sheets' else [work/(name+'_work')])
            stage(name, destinations, lambda argv=argv: command(argv))
        checks = check_outputs(prepared, module_dirs, region_dir)
        from qa import build_reports
        if args.no_report:
            stage('QA', [work/'reports'], lambda: write_json(work/'reports/omitted.json',dict(status='skipped',reason='--no-report',regenerate='geometric-organization qa SUBJECT --out NEW_QA_FOLDER')))
        else:
            stage('QA', [work/'reports'], lambda: build_reports(module_dirs, prepared, region_dir,
                  work/'reports', args.subject, state, command))
        stage('publish', [work/'public'], lambda: publish.build(args, work, work/'public', module_dirs, work/'reports'))
        write_json(work/'public/checks.json', checks)
        shutil.copy2(log, work/'public/provenance/commands.log')
        # Recheck immutable inputs immediately before public completion.
        for item in recipe['inputs'].values():
            if publish.sha256(item['path']) != item['sha256']: raise ValueError('An input changed during the run: '+item['path'])
        state.update(status='finalizing', publication_ready=True, public_files=publish.inventory(work/'public'))
        record(); finalize(out, state, args.keep_work)
        print(f'Complete: {out}', flush=True)
    except BaseException as exc:
        state.update(status='failed', error=type(exc).__name__+': '+str(exc), failed_utc=now())
        record()
        raise


def run(args):
    with ExitStack() as stack:
        return _run(args, stack)


def validate_measurement_options(args):
    from native_modules import DISPERSION_DEFAULTS, resolve_laminarity_scales, validate_dispersion_options
    chosen = args.modules.split(',') if isinstance(args.modules,str) else args.modules
    defaults = dict(radius_mm=6.875,draws=100,seed=20261015,qref=.03,sample_batch_size=4)
    for name, default in defaults.items():
        value=getattr(args,'sheets_'+name)
        if 'sheets' not in chosen and value != default: raise ValueError('--sheets-'+name.replace('_','-')+' requires the Sheets module')
    if any(not math.isfinite(v) or v<=0 for v in (args.sheets_radius_mm,args.sheets_qref)):
        raise ValueError('Sheets radius and qref must be finite and positive')
    if args.sheets_draws<1 or args.sheets_sample_batch_size<1 or args.sheets_seed<0:
        raise ValueError('Sheets draws and batch size must be positive; seed must be nonnegative')
    resolved=resolve_laminarity_scales(args.laminarity_min_scale_mm,args.laminarity_scales_mm,args.laminarity_fixed_scale_mm)
    if resolved is not None and 'laminarity' not in chosen: raise ValueError('Laminarity scale options require the Laminarity module')
    if args.laminarity_scales_mm is not None: args.laminarity_scales_mm=resolved
    options={name:getattr(args,'dispersion_'+name) for name in DISPERSION_DEFAULTS}
    validate_dispersion_options(**options)
    if 'dispersion' not in chosen and options!=DISPERSION_DEFAULTS: raise ValueError('Dispersion fitting options require the Dispersion module')


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'qa':
        from qa import regenerate
        p = argparse.ArgumentParser(description='Regenerate reports from a read-only compact subject folder')
        p.add_argument('input', type=Path); p.add_argument('--out', type=Path, required=True)
        a = p.parse_args(sys.argv[2:]); regenerate(a.input, a.out); return
    p = parser(); args = p.parse_args()
    chosen = args.modules.split(',')
    if not chosen or len(set(chosen)) != len(chosen) or any(x not in MODULES for x in chosen): p.error('Choose each requested module once from sheets,laminarity,dispersion.')
    if args.threads < 1 or (args.spacing is not None and (not math.isfinite(args.spacing) or not 0 < args.spacing < 20)): p.error('Use positive threads and spacing between 0 and 20 mm.')
    if args.dwi and (not args.bval or not args.bvec): p.error('--dwi requires --bval and --bvec.')
    if args.fod and (not args.brain_mask or not args.fa): p.error('--fod requires --brain-mask and --fa.')
    if args.fod and (args.bval or args.bvec or args.freesurfer): p.error('Prepared FOD uses aligned masks; omit gradients and --freesurfer.')
    if args.dwi and (args.fa or args.b0): p.error('DWI mode generates FA and b0; these inputs are for prepared FOD only.')
    if 'sheets' in chosen and not (args.wm_mask or args.freesurfer): p.error('Sheets requires --wm-mask or --freesurfer.')
    if 'sheets' not in chosen and (args.wm_mask or args.freesurfer): p.error('WM input is only used by Sheets; omit it for this selection.')
    for option, name in [(args.sheets_save_draws, 'sheets'), (args.laminarity_save_all_scales, 'laminarity'), (args.dispersion_save_parameters, 'dispersion')]:
        if option and name not in chosen: p.error('A save option refers to an unselected module: '+name)
    if args.save_native_fixels and not {'laminarity','dispersion'}.intersection(chosen): p.error('--save-native-fixels needs Laminarity or Dispersion.')
    if args.no_jhu and args.regional_wm_mask: p.error('--regional-wm-mask requires JHU summaries.')
    if args.no_jhu and args.jhu_labels: p.error('--jhu-labels requires JHU summaries.')
    if args.fod and args.csd_mode!='auto': p.error('Prepared FOD is already reconstructed; --csd-mode applies to DWI.')
    try: validate_measurement_options(args)
    except ValueError as error: p.error(str(error))
    run(args)


if __name__ == '__main__':
    main()

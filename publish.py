"""Compact publication and reversible views for the original QA readers."""
import hashlib
import json
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np

IDENTITY = {'afd.nii.gz', 'fiber_direction.nii.gz', 'native_fixel_id.nii.gz'}
REGIONS = {'JHU_labels_native.nii.gz', 'JHU_labels_eroded_1voxel.nii.gz',
           'JHU_template_FA_native.nii.gz'}
SHEETS_REGIONS = {'JHU_labels.nii.gz': 'JHU_labels_native.nii.gz',
                 'JHU_labels.csv': 'labels.csv',
                 'JHU_labels_eroded.nii.gz': 'JHU_labels_eroded_1voxel.nii.gz',
                 'template_in_native_FA.nii.gz': 'JHU_template_FA_native.nii.gz'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def inventory(root):
    root = Path(root)
    return {str(p.relative_to(root)): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
            for p in sorted(root.rglob('*')) if p.is_file() and not p.is_symlink()}


def verify(root, records):
    root = Path(root)
    for name, entry in records.items():
        p = root/name
        if p.is_symlink() or not p.is_file() or p.stat().st_size != entry['bytes'] or sha256(p) != entry['sha256']:
            raise ValueError(f'Missing or changed saved file: {p}')


def equal_image(left, right):
    a, b = nib.load(left), nib.load(right)
    return (a.shape == b.shape and np.allclose(a.affine, b.affine, atol=1e-5, rtol=0)
            and np.array_equal(np.asarray(a.dataobj), np.asarray(b.dataobj), equal_nan=True))


def copy_checked(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        equivalent = equal_image(source, destination) if source.name.endswith(('.nii', '.nii.gz')) else sha256(source) == sha256(destination)
        if not equivalent:
            raise ValueError(f'Files designated as shared differ: {source} and {destination}')
    else:
        shutil.copy2(source, destination)


def build(args, work, destination, module_dirs, qa_dir):
    work, destination, qa_dir = map(Path, (work, destination, qa_dir))
    destination.mkdir(parents=True)
    layout = {}
    native_modules = [name for name in ('laminarity','dispersion') if name in module_dirs]
    share_identity = (len(native_modules) < 2 or
                      all(equal_image(Path(module_dirs['laminarity'])/'fixel_maps'/name,
                                      Path(module_dirs['dispersion'])/'fixel_maps'/name) for name in IDENTITY))
    for source in (work/'prepared/anatomy').iterdir():
        if source.is_file():
            copy_checked(source, destination/'shared/anatomy'/source.name)
    if (work/'wm/T1_in_diffusion.nii.gz').exists():
        copy_checked(work/'wm/T1_in_diffusion.nii.gz', destination/'shared/anatomy/T1_in_diffusion.nii.gz')
    if (work/'regions').exists():
        for source in (work/'regions').iterdir():
            if source.is_file(): copy_checked(source, destination/'shared/regions'/source.name)
    for name, folder in module_dirs.items():
        folder = Path(folder)
        mapping = layout[name] = {}
        for source in sorted(folder.rglob('*')):
            if not source.is_file(): continue
            relative = source.relative_to(folder)
            group = relative.parts[0]
            if group in {'work', 'QA', 'qa_work', '.cache'} or source.name in {'qa.pdf', '.run.lock', 'JHU_QA.png'}:
                continue
            if group == 'anatomy':
                target = Path('shared/anatomy')/relative.name
            elif group == 'fixel_maps' and relative.name in IDENTITY:
                target = (Path('shared/fixels') if share_identity else Path(name)/'fixel_maps')/relative.name
            elif group == 'regions' and relative.name in REGIONS:
                target = Path('shared/regions')/relative.name
            elif group == 'regions' and relative.name == 'template_in_native_FA.nii.gz':
                target = Path('shared/regions/JHU_template_FA_native.nii.gz')
            elif name == 'sheets' and group == 'registration' and relative.name in SHEETS_REGIONS:
                target = Path('shared/regions')/SHEETS_REGIONS[relative.name]
            elif group in {'parameters', 'scales', 'mrtrix_fixels', 'draws'}:
                target = Path('extras')/name/relative
            elif name == 'sheets' and group == 'audit':
                target = Path('extras/sheets/draws')/relative.name
            elif group in {'provenance', 'logs', 'reconstruction'} or relative.suffix == '.json' or relative.name == 'README.txt':
                target = Path('provenance/modules')/name/relative
            else:
                target = Path(name)/relative
            copy_checked(source, destination/target)
            mapping[str(relative)] = str(target)
    reconstruction = work/'prepared/reconstruction'
    for source in reconstruction.rglob('*'):
        if source.is_file() and source.suffix in {'.txt', '.json', '.bval', '.bvec'}:
            copy_checked(source, destination/'provenance/reconstruction'/source.relative_to(reconstruction))
    if (work/'wm/alignment.json').exists():
        copy_checked(work/'wm/alignment.json', destination/'provenance/WM_alignment.json')
    for source in qa_dir.glob('*.pdf'):
        copy_checked(source, destination/'QA'/source.name)
    for source in qa_dir.glob('*.json'):
        copy_checked(source, destination/'provenance/QA'/source.name)
    if args.save_fod:
        copy_checked(reconstruction/'WM_FOD.nii.gz', destination/'extras/fod/WM_FOD.nii.gz')
    if args.save_native_fixels:
        shutil.copytree(work/'native/fixels', destination/'extras/native_fixels')
    if args.save_registration:
        for key, source in [('JHU', work/'regions/registration'), ('T1', work/'wm')]:
            if source.exists(): shutil.copytree(source, destination/'extras/registration'/key)
    readme='''Geometric organization | one FOD, three complementary measurements

Start with QA/00_overview.pdf, then the report for the measure of interest.
shared/ contains anatomy, native fixel identity/AFD/directions, and one shared JHU atlas.
sheets/ contains q (mm^-1; lower is more sheet-compatible), all three fixed pairs,
reference-plane tensors, availability, support, displays and historical SPI.
laminarity/ contains lamellarity/tubularity, selected scales, geometry and coverage.
dispersion/ contains ODI, dispersion anisotropy, angular widths, FS and validity.
Module regions/ tables use the same JHU labels, with their own meaningful denominators.
Sheets uses its WM domain; the other calculations use brain observations.
Sheets pair12 is primary, never a minimum-q winner. Native fixel ranks are local;
they do not identify tracts or correspond automatically to Sheets reference peaks.
NaN is missing; zero may be a genuine measurement. Orientations are physical/world
axes. Sheets reference-plane tensors and lamellarity tensors are distinct measures.
FOD draws describe angular structure, not acquisition bootstrap uncertainty.

run.json lists the actual inputs, settings, engine pins, stages and saved-file hashes.
checks.json records common grid/identity and output checks. provenance/ preserves
original engine records and the map from their paths to this compact layout.
Original engine records contain historical work paths; run.json owns current paths.
All default QA can be regenerated after moving this entire folder, without the FOD:
  geometric-organization qa SUBJECT_FOLDER --out NEW_QA_FOLDER
extras/ contains only requested large exports or a Bingham parameter bundle <=10 MB.
Supplied inputs are never deleted. Failed work is retained for inspection/resume.
'''
    if getattr(args,'no_report',False):
        readme=readme.replace('Start with QA/00_overview.pdf, then the report for the measure of interest.',
            'PDFs were skipped with --no-report. Regenerate them with the qa command below.')
    (destination/'README.txt').write_text(readme)
    (destination/'provenance').mkdir(exist_ok=True)
    (destination/'provenance/layout.json').write_text(json.dumps(layout, indent=2)+'\n')
    return layout


def restore(root, module_name, destination):
    """Temporary read-only file links; no persistent links leave the subject."""
    root, destination = Path(root).resolve(), Path(destination)
    layout = json.loads((root/'provenance/layout.json').read_text())[module_name]
    destination.mkdir(parents=True)
    for old, new in layout.items():
        source = (root/new).resolve()
        if root not in source.parents or not source.is_file():
            raise ValueError(f'Invalid compact-layout reference: {new}')
        target = destination/old
        if destination.resolve() not in target.resolve().parents:
            raise ValueError(f'Invalid engine-relative path: {old}')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
    return destination

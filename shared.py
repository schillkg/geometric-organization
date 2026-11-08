"""One preparation and one registration for the three unchanged estimators."""
import csv
import importlib.util
import json
import os
from argparse import Namespace
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np

LAMINARITY = Path(os.environ.get('GEO_LAMINARITY_SOURCE', '/opt/laminarity/transverse'))
DISPERSION = Path(os.environ.get('GEO_DISPERSION_SOURCE', '/opt/dispersion/pipeline'))
FS_LABELS = [2, 7, 16, 41, 46, 251, 252, 253, 254, 255]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def save(path, values, reference):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # FreeSurfer inputs use MGH headers; convert them before applying NIfTI
    # scaling fields. The supplied affine remains the spatial reference.
    header = nib.Nifti1Header.from_header(reference.header)
    header.set_data_dtype(values.dtype)
    header.set_slope_inter(None, None)
    nib.save(nib.Nifti1Image(values, reference.affine, header), path)


def on_grid(path, reference, binary=False):
    image = nib.load(path)
    if image.shape != reference.shape[:3] or not np.allclose(image.affine, reference.affine, atol=1e-5, rtol=0):
        raise ValueError(f'Image does not match its required grid: {path}')
    values = np.asarray(image.dataobj)
    if not np.isfinite(values).all():
        raise ValueError(f'Nonfinite values in {path}')
    if binary and (not np.isin(values, [0, 1]).all() or not values.any()):
        raise ValueError(f'A nonempty binary mask is required: {path}')
    return values


def prepare(args, out, work, command):
    """Use the frozen Laminarity/Dispersion preparation exactly once."""
    out, work = Path(out), Path(work)
    out.mkdir(parents=True)
    work.mkdir(parents=True)
    engine = module('geo_frozen_preparation', LAMINARITY/'prepare_dwi.py')
    spacing = args.spacing
    native_spacing = getattr(args, 'native_spacing', False)
    if native_spacing and spacing is not None:
        raise ValueError('Choose --spacing or --native-spacing, not both')
    if args.fod:
        converted = work/'supplied_fod.nii.gz'
        command(['mrconvert', args.fod, converted, '-nthreads', args.threads, '-quiet'])
        image = nib.load(converted)
        grid = engine.validate_grid(image)
        zoom = np.asarray(grid['spacing_mm'])
        if not np.allclose(zoom, zoom[0], atol=1e-5, rtol=0):
            raise ValueError('Prepared FOD must be isotropic; supply a ready analysis-grid FOD.')
        if spacing is not None and not np.allclose(zoom, spacing, atol=1e-5, rtol=0):
            raise ValueError('--spacing conflicts with the prepared FOD grid; no SH interpolation is performed.')
        spacing = float(zoom[0])
    else:
        if native_spacing:
            grid = engine.validate_grid(nib.load(args.dwi))
            zoom = np.asarray(grid['spacing_mm'])
            if not np.allclose(zoom, zoom[0], atol=1e-5, rtol=0):
                raise ValueError('--native-spacing requires an isotropic input DWI grid')
            spacing = float(zoom[0])
        else:
            spacing = 1.25 if spacing is None else spacing
    request = Namespace(fod=args.fod, dwi=args.dwi, bval=args.bval, bvec=args.bvec,
                        mask=args.brain_mask, fa=args.fa, spacing=spacing,
                        threads=args.threads, csd_mode=getattr(args, 'csd_mode', 'auto'))
    profile = engine.prepare(request, out, work, command)
    image = nib.load(out/'reconstruction/WM_FOD.nii.gz')
    mask = on_grid(out/'anatomy/brain_mask.nii.gz', image, binary=True) > 0
    if not np.isfinite(np.asarray(image.dataobj)[mask]).all():
        raise ValueError('FOD has nonfinite coefficients inside the brain observation mask.')
    if args.b0:
        save(out/'anatomy/mean_b0.nii.gz', on_grid(args.b0, image).astype(np.float32), image)
    profile.update(shared_by=args.modules,
                   integration_policy='One FOD/grid; MRtrix 3.0.4; no module-specific reconstruction',
                   supplied_fod_grid_preserved=bool(args.fod))
    if native_spacing:
        profile['native_spacing_requested'] = True
        profile['native_spacing_note'] = 'Use the isotropic input voxel spacing; supplied FOD is preserved and DWI follows the frozen grid preparation.'
    (out/'reconstruction/reconstruction.json').write_text(json.dumps(profile, indent=2)+'\n')
    return profile


def wm_domain(args, prepared, out, command):
    """Align a supplied WM mask or FreeSurfer mask once; never alter FOD support."""
    prepared, out = Path(prepared), Path(out)
    out.mkdir(parents=True)
    fod = prepared/'reconstruction/WM_FOD.nii.gz'
    reference = nib.load(fod)
    brain = on_grid(prepared/'anatomy/brain_mask.nii.gz', reference, binary=True) > 0
    aligned = out/'wm_mask.nii.gz'
    if args.wm_mask:
        if args.dwi:
            on_grid(args.wm_mask, nib.load(args.dwi), binary=True)
            command(['mrgrid', args.wm_mask, 'regrid', '-template', fod,
                     '-interp', 'nearest', aligned, '-nthreads', args.threads, '-quiet'])
        else:
            values = on_grid(args.wm_mask, reference, binary=True)
            save(aligned, values.astype(np.uint8), reference)
        record = dict(method='Supplied input-grid WM mask; nearest-neighbor grid alignment only',
                      input=str(args.wm_mask))
    else:
        fs = Path(args.freesurfer)/'mri'
        t1_path, aseg_path = fs/'T1.mgz', fs/'aseg.mgz'
        t1, aseg = nib.load(t1_path), nib.load(aseg_path)
        if t1.shape != aseg.shape or not np.allclose(t1.affine, aseg.affine, atol=1e-5):
            raise ValueError('FreeSurfer T1.mgz and aseg.mgz must share their grid.')
        t1_values = np.asarray(t1.dataobj, dtype=np.float32)
        labels = np.asarray(aseg.dataobj)
        if not np.isfinite(t1_values).all() or not np.isfinite(labels).all():
            raise ValueError('FreeSurfer inputs contain nonfinite values.')
        moving = out/'T1_brain.nii.gz'
        save(moving, np.where(labels > 0, t1_values, 0).astype(np.float32), t1)
        source_wm = out/'WM_structural.nii.gz'
        save(source_wm, np.isin(labels, FS_LABELS).astype(np.uint8), aseg)
        b0 = on_grid(prepared/'anatomy/mean_b0.nii.gz', reference)
        fixed = out/'b0_brain.nii.gz'
        save(fixed, np.where(brain, b0, 0).astype(np.float32), reference)
        # Relative filenames avoid ANTs bracket parsing problems with spaces.
        command(['antsRegistration', '--dimensionality', '3', '--float', '1',
                 '--output', '[T1_to_diffusion_,T1_in_diffusion.nii.gz]',
                 '--interpolation', 'Linear', '--winsorize-image-intensities', '[0.005,0.995]',
                 '--use-histogram-matching', '0', '--random-seed', '1',
                 '--initial-moving-transform', '[b0_brain.nii.gz,T1_brain.nii.gz,1]',
                 '--transform', 'Rigid[0.1]',
                 '--metric', 'MI[b0_brain.nii.gz,T1_brain.nii.gz,1,32,Regular,0.25]',
                 '--convergence', '[1000x500x250x0,1e-6,10]', '--shrink-factors', '8x4x2x1',
                 '--smoothing-sigmas', '3x2x1x0vox'], cwd=out)
        command(['antsApplyTransforms', '-d', '3', '--float', '1', '-i', source_wm,
                 '-r', fixed, '-o', aligned, '-n', 'NearestNeighbor',
                 '-t', out/'T1_to_diffusion_0GenericAffine.mat'])
        record = dict(method='One ANTs rigid MI alignment of FreeSurfer T1 brain to mean b0',
                      source_t1=str(t1_path), source_aseg=str(aseg_path),
                      labels=FS_LABELS, includes_whole_brainstem=True,
                      interpolation='NearestNeighbor for mask; linear for T1', random_seed=1)
    values = on_grid(aligned, reference, binary=True) > 0
    effective = values & brain
    if not effective.any():
        raise ValueError('WM domain does not intersect the brain observation mask.')
    save(aligned, effective.astype(np.uint8), reference)
    record.update(aligned_wm_voxels=int(values.sum()), effective_voxels=int(effective.sum()),
                  brain_voxels=int(brain.sum()), applies_to='Sheets measurement centers only')
    (out/'alignment.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


def jhu(prepared, out, threads, native_labels=None):
    """Register once, or reuse supplied final-grid labels, for every module."""
    prepared, out = Path(prepared), Path(out)
    out.mkdir(parents=True)
    engine = module('geo_atlas_registration', DISPERSION/'atlas_registration.py')
    atlas = DISPERSION/'atlas'
    fa, brain = prepared/'anatomy/FA.nii.gz', prepared/'anatomy/brain_mask.nii.gz'
    if native_labels is None:
        labels, record = engine.register_labels(fa, brain, atlas, out, threads)
        image = nib.load(labels)
        values = np.asarray(image.dataobj)
    else:
        native_labels = Path(native_labels)
        image = nib.load(fa)
        values = on_grid(native_labels, image)
        labels = out/'JHU_labels_native.nii.gz'
        record = dict(method='Provided labels on the exact shared analysis grid; no registration',
                      input=str(native_labels), sha256=engine.sha256(native_labels),
                      registration_performed=False, warped_template_available=False,
                      interpolation='None; supplied final-grid labels used directly')
    names = engine.atlas_names(atlas)
    if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all() or not set(np.unique(values)).issubset({0, *names}) or not np.any(values):
        raise ValueError('Invalid shared JHU label values.')
    save(labels, values.astype(np.uint16), image)
    save(out/'JHU_labels_eroded_1voxel.nii.gz', engine.erode_labels(values.astype(np.uint16)), image)
    if native_labels is None:
        shutil.copy2(out/'registration/template_in_native_FA.nii.gz', out/'JHU_template_FA_native.nii.gz')
        shutil.copy2(out/'registration/commands.log', out/'commands.log')
    else:
        (out/'commands.log').write_text('No registration commands: supplied final-grid JHU labels reused.\n')
    shutil.copy2(atlas/'atlas_manifest.json', out/'atlas_manifest.json')
    with (out/'labels.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['id', 'name']); writer.writerows(sorted(names.items()))
    record.update(shared_by='all selected modules', regions=len(names),
                  replaces_in_sheets='Historical FLIRT/ENIGMA 48-label registration; q is unaffected',
                  erosion='Each parcel eroded independently by one six-connected analysis voxel')
    (out/'registration.json').write_text(json.dumps(record, indent=2)+'\n')
    return record

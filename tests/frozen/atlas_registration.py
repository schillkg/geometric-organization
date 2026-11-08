"""Unchanged Laminarity 1.2 registration helpers; downstream labels only."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load_on_grid(path, reference):
    image = nib.load(path)
    if image.shape[:3] != reference.shape[:3] or not np.allclose(image.affine, reference.affine, atol=1e-5):
        raise ValueError(f'Image does not match the native FA grid: {path}')
    return np.asarray(image.dataobj)

def save_image(path, data, reference):
    header = reference.header.copy()
    header.set_data_dtype(data.dtype)
    header.set_slope_inter(None, None)
    nib.save(nib.Nifti1Image(data, reference.affine, header), path)

def atlas_names(atlas_dir):
    # This is a Label atlas, with an explicit index 0 for unclassified.
    # Unlike probabilistic maxprob atlases, its XML indexes need no +1 offset.
    tree = ET.parse(Path(atlas_dir) / 'JHU-labels.xml')
    if tree.findtext('./header/type') != 'Label':
        raise ValueError('Expected the JHU Label atlas, not a probabilistic atlas')
    return {int(e.attrib['index']): e.text.strip() for e in tree.findall('./data/label')
            if int(e.attrib['index']) > 0}

def erode_labels(labels, iterations=1):
    """Erode each label by one six-connected native voxel per iteration."""
    if not isinstance(iterations, (int, np.integer)) or iterations < 0:
        raise ValueError('Erosion iterations must be a nonnegative integer')
    if iterations == 0:
        return labels.copy()
    result = np.zeros_like(labels)
    # scipy's default 3D structure is six-connected. Outside is background.
    for label in np.unique(labels):
        if label == 0:
            continue
        xyz = np.where(labels == label)
        box = tuple(slice(int(x.min()), int(x.max()) + 1) for x in xyz)
        eroded = binary_erosion(labels[box] == label, iterations=iterations, border_value=0)
        view = result[box]
        view[eroded] = label
    return result

def run_command(command, folder, env, log):
    # Array arguments avoid shell quoting problems in paths containing spaces.
    with log.open('a') as f:
        f.write(json.dumps([str(x) for x in command]) + '\n')
        f.flush()
        subprocess.run([str(x) for x in command], cwd=folder, env=env,
                       stdout=f, stderr=subprocess.STDOUT, check=True)

def register_labels(fa_path, brain_mask_path, atlas_dir, out, threads):
    """Rigid+affine+SyN FA registration; template is moving, native FA fixed."""
    for executable in ('antsRegistration', 'antsApplyTransforms'):
        if shutil.which(executable) is None:
            raise RuntimeError(f'{executable} is required for JHU registration')
    registration = out / 'registration'
    registration.mkdir()
    fa = nib.load(fa_path)
    mask = load_on_grid(brain_mask_path, fa) > 0
    values = np.asarray(fa.dataobj, dtype=np.float32)
    if not mask.any() or not np.isfinite(values[mask]).all():
        raise ValueError('Native FA or brain mask is empty/nonfinite')
    save_image(registration / 'native_FA.nii.gz', np.where(mask, values, 0).astype(np.float32), fa)
    save_image(registration / 'native_brain_mask.nii.gz', mask.astype(np.uint8), fa)
    template = nib.load(atlas_dir / 'FMRIB58_FA_1mm.nii.gz')
    atlas_manifest = json.loads((atlas_dir / 'atlas_manifest.json').read_text())
    template_values = np.asarray(template.dataobj, dtype=np.float32) * atlas_manifest['template_FA_scale_to_unit']
    save_image(registration / 'template_FA.nii.gz', template_values, template)
    save_image(registration / 'template_support.nii.gz', (template_values > 0).astype(np.uint8), template)
    # Relative paths keep ANTs bracket syntax simple, including when out has spaces.
    fixed, moving = 'native_FA.nii.gz', 'template_FA.nii.gz'
    command = ['antsRegistration', '--dimensionality', '3', '--float', '1',
               '--output', '[template_to_native_,template_in_native_FA.nii.gz]',
               '--interpolation', 'Linear', '--winsorize-image-intensities', '[0.005,0.995]',
               '--use-histogram-matching', '0', '--random-seed', '1',
               '--initial-moving-transform', f'[{fixed},{moving},1]',
               '--masks', '[native_brain_mask.nii.gz,template_support.nii.gz]']
    # Fixed schedule based on ANTs SyNQuick reproducible mode; no shell wrapper.
    for transform in ('Rigid[0.1]', 'Affine[0.1]'):
        command += ['--transform', transform, '--metric', f'GC[{fixed},{moving},1,1,Regular,0.25]',
                    '--convergence', '[1000x500x250x0,1e-6,10]',
                    '--shrink-factors', '8x4x2x1', '--smoothing-sigmas', '3x2x1x0vox']
    command += ['--transform', 'SyN[0.2,3,0]', '--metric', f'CC[{fixed},{moving},1,2]',
                '--convergence', '[100x70x50x0,1e-6,10]', '--shrink-factors', '8x4x2x1',
                '--smoothing-sigmas', '3x2x1x0vox', '--verbose', '1']
    env = os.environ.copy()
    env.update(ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=str(threads), ANTS_RANDOM_SEED='1')
    log = registration / 'commands.log'
    run_command(command, registration, env, log)
    labels_path = out / 'JHU_labels_native.nii.gz'
    apply = ['antsApplyTransforms', '-d', '3', '--float', '1',
             '-i', atlas_dir / 'JHU-ICBM-labels-1mm.nii.gz', '-r', fixed,
             '-o', labels_path, '-n', 'NearestNeighbor',
             '-t', 'template_to_native_1Warp.nii.gz', '-t', 'template_to_native_0GenericAffine.mat']
    run_command(apply, registration, env, log)
    version = subprocess.run(['antsRegistration', '--version'], capture_output=True, text=True, check=True)
    return labels_path, dict(method='ANTs rigid + affine + SyN; template FA to native FA',
                             interpolation='NearestNeighbor', random_seed=1, threads=threads,
                             version=version.stdout.strip(), commands=[list(map(str, command)), list(map(str, apply))])

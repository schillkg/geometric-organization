#!/usr/bin/env python3
"""Create a three-population software-parity fixture; no biological claim.

Run with the installed Dispersion Python/MRtrix runtime. Its unchanged sh_matrix
and verify_matrix helpers establish the actual Tournier07 coefficient convention.
Only the new fixture directory and owned scratch are written.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import nibabel as nib
import numpy as np


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''): h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args(); a.out=a.out.resolve(); a.work=a.work.resolve()
    for path in (a.out,a.work):
        if path==Path('/nfs2') or Path('/nfs2') in path.parents: p.error('/nfs2 is read-only')
        if path.exists() and any(path.iterdir()): p.error('Use a new or empty owned destination: '+str(path))
        path.mkdir(parents=True,exist_ok=True)
    if a.out==a.work or a.out in a.work.parents or a.work in a.out.parents:
        p.error('Fixture and scratch must be separate')
    source=Path(os.environ.get('GEO_DISPERSION_SOURCE','/opt/dispersion/pipeline'))
    sys.path.insert(0,str(source))
    from compute import sphere_directions,sh_matrix,verify_matrix
    started=time.monotonic(); commands=[]
    bank=sphere_directions(4096)
    matrix,direction_file=sh_matrix(45,bank,a.work,commands)
    x,y,z=bank.T
    # Three axial angular Gaussians: x has unequal transverse spread; y/z circular.
    targets=np.column_stack((np.exp(-6*y*y-14*z*z),
                             np.exp(-10*(x*x+z*z)),np.exp(-12*(x*x+y*y))))
    coefficients,residuals,rank,singular=np.linalg.lstsq(matrix.T.astype(np.float64),targets,rcond=None)
    if rank!=45: raise ValueError('The MRtrix lmax8 design matrix is rank-deficient')
    lobe_coefficients=coefficients.T.astype(np.float32)
    shape=(16,18,20); spacing=2.
    axes=[(np.arange(n)-(n-1)/2)*spacing for n in shape]
    xx,yy,zz=np.meshgrid(*axes,indexing='ij')
    amplitudes=np.stack((1+.25*np.tanh(yy/5), .85*(1+.20*np.tanh(zz/5)),
                         .70*(1+.15*np.sin(xx/8)*np.cos(yy/7))),axis=-1).astype(np.float32)
    fod=np.einsum('...l,lc->...c',amplitudes,lobe_coefficients,dtype=np.float32)
    affine=np.diag([spacing,spacing,spacing,1.]); affine[:3,3]=[v[0] for v in axes]
    def save(name,values):
        image=nib.Nifti1Image(values,affine); image.set_qform(affine,1); image.set_sform(affine,1)
        nib.save(image,a.out/name)
    save('fod.nii.gz',fod)
    save('brain_mask.nii.gz',np.ones(shape,np.uint8))
    save('FA.nii.gz',(.45+.18*np.exp(-(xx*xx+yy*yy+zz*zz)/200)).astype(np.float32))
    samples=fod.reshape(-1,45)[np.linspace(0,np.prod(shape)-1,12,dtype=int)]
    parity=verify_matrix(samples,matrix,direction_file,a.work,commands)
    peaks=a.work/'reference_peaks.nii.gz'
    argv=['sh2peaks',str(a.out/'fod.nii.gz'),str(peaks),'-num','3','-threshold','0.05',
          '-mask',str(a.out/'brain_mask.nii.gz'),'-nthreads',str(a.threads),'-quiet']
    commands.append(argv); subprocess.run(argv,check=True)
    vectors=np.asarray(nib.load(peaks).dataobj).reshape(shape+(3,3))
    peak_amplitudes=np.linalg.norm(vectors,axis=-1)
    if not np.isfinite(vectors).all() or not np.all(peak_amplitudes>.05):
        raise ValueError('Fixture does not have three strong reference peaks throughout the brain')
    unit=vectors/peak_amplitudes[...,None]
    nearest=np.argmax(np.abs(unit),axis=-1)
    if not np.all(np.sort(nearest,axis=-1)==np.arange(3)):
        raise ValueError('Reference peaks do not represent all three orthogonal populations')
    angular_error=np.rad2deg(np.arccos(np.clip(np.max(np.abs(unit),axis=-1),0,1)))
    evaluated=samples@matrix
    record=dict(purpose='Software parity and nonempty crossing-population control; no histological validation',
        shape=list(shape),spacing_mm=spacing,affine=affine.tolist(),basis='MRtrix Tournier07, lmax8, 45 coefficients',
        matrix_method='Frozen Dispersion compute.sh_matrix: sh2amp of identity coefficient volumes',
        matrix_parity=parity,hemisphere_directions=len(bank),design_rank=int(rank),
        angular_lobes=['exp(-6*y^2-14*z^2), axis x','exp(-10*(x^2+z^2)), axis y',
                       'exp(-12*(x^2+y^2)), axis z'],
        amplitudes=['1+0.25*tanh(Y/5)','0.85*(1+0.20*tanh(Z/5))',
                    '0.70*(1+0.15*sin(X/8)*cos(Y/7))'],
        amplitude_coordinates='Scanner RAS mm at voxel centers',coefficient_dtype=str(fod.dtype),
        lobe_coefficients_float32=lobe_coefficients.tolist(),
        fit_relative_errors=(np.linalg.norm(matrix.T@lobe_coefficients.T-targets,axis=0)/np.linalg.norm(targets,axis=0)).tolist(),
        sampled_minimum_FOD=float(evaluated.min()),sampled_negative_fraction=float(np.mean(evaluated<0)),
        reference_peaks=dict(all_voxels_have_three=True,minimum_amplitude=float(peak_amplitudes.min()),
                             maximum_axis_error_deg=float(angular_error.max()),threshold=.05),
        commands=commands,compute_source_sha256=sha256(source/'compute.py'),generator_sha256=sha256(__file__),
        numpy=np.__version__,mrtrix=subprocess.check_output(['mrinfo','-version'],text=True,stderr=subprocess.STDOUT).strip(),
        files={name:dict(bytes=(a.out/name).stat().st_size,sha256=sha256(a.out/name))
               for name in ('fod.nii.gz','brain_mask.nii.gz','FA.nii.gz')},elapsed_seconds=time.monotonic()-started)
    (a.out/'fixture.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='complete',output=str(a.out),peaks=record['reference_peaks'],matrix_parity=parity)))


if __name__=='__main__': main()

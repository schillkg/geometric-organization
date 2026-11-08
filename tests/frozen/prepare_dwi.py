"""Prepare a preprocessed DWI for L/T using the same CSD recipes as Sheets.

This module is self-contained; the original Sheet software is not imported or
modified. Motion/distortion correction must already be complete, with matching
rotated FSL b-vectors. No additional denoising or WM analysis mask is applied.
"""
from pathlib import Path
import json
import shutil

import nibabel as nib
import numpy as np
from scipy.special import gammaln, lpmv


def angular_rank(directions, lmax):
    directions=np.asarray(directions,float)
    norms=np.linalg.norm(directions,axis=1)
    if not np.isfinite(directions).all() or np.any(norms<=1e-8):
        raise ValueError('Invalid nonzero-shell gradient direction')
    xyz=directions/norms[:,None]
    phi=np.arctan2(xyz[:,1],xyz[:,0]);z=np.clip(xyz[:,2],-1,1)
    columns=[]
    for degree in range(0,lmax+1,2):
        for order in range(-degree,degree+1):
            m=abs(order)
            scale=np.sqrt((2*degree+1)/(4*np.pi)*np.exp(gammaln(degree-m+1)-gammaln(degree+m+1)))
            value=scale*lpmv(m,degree,z)
            if order<0:value*=np.sqrt(2)*np.sin(m*phi)
            elif order>0:value*=np.sqrt(2)*np.cos(m*phi)
            columns.append(value)
    return int(np.linalg.matrix_rank(np.column_stack(columns)))


def reconstruction_profile(gradient,means,sizes,indices,bzero_means,requested='auto'):
    """Use MRtrix shell membership; never silently discard a diffusion shell."""
    gradient=np.asarray(gradient,float);means=np.asarray(means,float)
    if gradient.ndim!=2 or gradient.shape[1]!=4 or not np.isfinite(gradient).all():
        raise ValueError('Invalid MRtrix gradient table')
    if len(means)!=len(sizes) or len(means)!=len(indices) or len(bzero_means)!=1:
        raise ValueError('Ambiguous MRtrix shell classification')
    flat=[i for shell in indices for i in shell]
    if sorted(flat)!=list(range(len(gradient))) or list(map(len,indices))!=list(sizes):
        raise ValueError('Shell indices do not partition the diffusion volumes')
    bzero=np.flatnonzero(np.isclose(means,bzero_means[0],atol=.05,rtol=1e-5))
    if len(bzero)!=1:raise ValueError('Could not identify the b0 shell')
    nonzero=[i for i in range(len(means)) if i!=int(bzero[0])]
    if not nonzero:raise ValueError('No diffusion-weighted shell')
    mode='sscsd' if len(nonzero)==1 else 'msmt'
    if requested not in ('auto',mode):
        raise ValueError(f'{requested} conflicts with {len(nonzero)} nonzero shells; no shell is silently discarded')
    lmax=6 if mode=='sscsd' else 8
    required=(lmax+1)*(lmax+2)//2
    ranks={str(i):angular_rank(gradient[indices[i],:3],lmax) for i in nonzero}
    highest=max(nonzero,key=lambda i:means[i])
    if ranks[str(highest)]!=required:
        raise ValueError(f'Highest shell cannot identify lmax {lmax}: angular rank {ranks[str(highest)]}/{required}')
    return dict(mode=mode,wm_lmax=lmax,shell_means=means.tolist(),shell_sizes=list(map(int,sizes)),
                shell_indices=indices,bzero_shell_index=int(bzero[0]),angular_rank=ranks,
                single_shell_value=float(means[nonzero[0]]) if mode=='sscsd' else None,
                basis='MRtrix Tournier07',normalization='No mtnormalise; no extra response-mask erosion')


def validate_grid(image):
    affine=np.asarray(image.affine,float)
    spacing=np.linalg.norm(affine[:3,:3],axis=0)
    if not np.isfinite(affine).all() or np.any(spacing<=0):raise ValueError('Invalid image geometry')
    frame=affine[:3,:3]/spacing
    if not np.allclose(frame.T@frame,np.eye(3),atol=1e-5):
        raise ValueError('Sheared grids are unsupported; supply a preprocessed orthogonal image')
    return dict(shape=list(image.shape),affine=affine.tolist(),spacing_mm=spacing.tolist())


def prepare(args,out,work,command):
    """Run commands through the subject logger; preserve scientific inputs."""
    anatomy=out/'anatomy';reconstruction=out/'reconstruction'
    anatomy.mkdir(exist_ok=True);reconstruction.mkdir(exist_ok=True)
    flags=['-nthreads',str(args.threads),'-quiet','-force']
    brain=anatomy/'brain_mask.nii.gz';fa=anatomy/'FA.nii.gz';fod=reconstruction/'WM_FOD.nii.gz'
    if args.fod:
        # Standardize the spatial grid before fixel extraction. Pure spacing
        # changes leave the scanner-coordinate SH basis/directions unchanged.
        native_fod=work/'input_FOD.nii.gz'
        command(['mrconvert',args.fod,native_fod]+flags)
        reference=nib.load(native_fod);grid=validate_grid(reference)
        if reference.ndim!=4 or reference.shape[3] not in (28,45):
            raise ValueError('Direct FOD input requires MRtrix Tournier07 lmax 6 or 8 coefficients')
        resampled=not np.allclose(grid['spacing_mm'],args.spacing,atol=1e-6,rtol=1e-6)
        if resampled:
            command(['mrgrid',native_fod,'regrid','-voxel',str(args.spacing),'-interp','cubic',fod]+flags)
        else:
            shutil.copy2(native_fod,fod)
        for source,dest,interpolation in [(args.mask,brain,'nearest'),(args.fa,fa,'linear')]:
            if source is None:continue
            native_scalar=work/('input_'+dest.name)
            command(['mrconvert',source,native_scalar]+flags)
            scalar=nib.load(native_scalar)
            if scalar.shape!=reference.shape[:3] or not np.allclose(scalar.affine,reference.affine,atol=1e-4):
                raise ValueError('Direct-FOD brain mask and FA must match its native grid')
            if resampled:
                command(['mrgrid',native_scalar,'regrid','-template',fod,'-interp',interpolation,dest]+flags)
            else:
                shutil.copy2(native_scalar,dest)
        profile=dict(mode='provided_fod',basis='User-asserted MRtrix Tournier07',wm_lmax=6 if reference.shape[3]==28 else 8,
                     resampled=resampled,input_grid=grid,spacing_mm=args.spacing)
    else:
        original=nib.load(args.dwi);grid=validate_grid(original)
        if original.ndim!=4:raise ValueError('DWI must be a 4D NIfTI')
        bvals=np.loadtxt(args.bval).ravel();bvecs=np.loadtxt(args.bvec)
        if bvals.shape!=(original.shape[3],) or bvecs.shape!=(3,original.shape[3]):
            raise ValueError('FSL gradients require N b-values and a 3-by-N b-vector matrix matching the DWI')
        if not np.isfinite(bvals).all() or not np.isfinite(bvecs).all() or (bvals<0).any():
            raise ValueError('Gradients must be finite and b-values nonnegative')
        imported=work/'dwi_import.mif';dwi=work/'dwi_fit.mif';bzero=work/'bzero.mif'
        command(['mrconvert',args.dwi,imported,'-fslgrad',args.bvec,args.bval]+flags)
        command(['mrgrid',imported,'regrid','-voxel',str(args.spacing),dwi]+flags)
        command(['dwiextract',dwi,bzero,'-bzero']+flags)
        b0=anatomy/'mean_b0.nii.gz'
        command(['mrmath',bzero,'mean',b0,'-axis','3']+flags)
        if args.mask:
            mask=nib.load(args.mask)
            if mask.shape!=original.shape[:3] or not np.allclose(mask.affine,original.affine,atol=1e-4):
                raise ValueError('Supplied brain mask must match the input DWI grid')
            command(['mrgrid',args.mask,'regrid','-template',b0,'-interp','nearest',brain]+flags)
        else:command(['dwi2mask',dwi,brain]+flags)
        gradient=reconstruction/'grad_mrtrix.txt'
        command(['mrinfo',dwi,'-export_grad_mrtrix',gradient,'-force'])
        means=list(map(float,command(['mrinfo',dwi,'-shell_bvalues']).split()))
        sizes=list(map(int,command(['mrinfo',dwi,'-shell_sizes']).split()))
        indices=[list(map(int,x.split(','))) for x in command(['mrinfo',dwi,'-shell_indices']).split()]
        bzero_means=list(map(float,command(['mrinfo',bzero,'-shell_bvalues']).split()))
        profile=reconstruction_profile(np.loadtxt(gradient),means,sizes,indices,bzero_means,args.csd_mode)
        profile.update(input_grid=grid,resampled=True,spacing_mm=args.spacing)
        responses=reconstruction/'responses';responses.mkdir(exist_ok=True)
        wm=responses/'wm.txt';wm_fod=work/'WM_FOD.mif'
        if profile['mode']=='msmt':
            gm,csf=responses/'gm.txt',responses/'csf.txt'
            command(['dwi2response','dhollander',dwi,wm,gm,csf,'-mask',brain,'-scratch',work/'scratch']+flags)
            command(['dwi2fod','msmt_csd',dwi,wm,wm_fod,gm,work/'GM_FOD.mif',csf,work/'CSF_FOD.mif',
                     '-mask',brain,'-lmax','8,0,0']+flags)
        else:
            shell=format(profile['single_shell_value'],'.12g')
            command(['dwi2response','tournier',dwi,wm,'-mask',brain,'-shells',shell,'-lmax','6',
                     '-voxels',reconstruction/'response_voxels.nii.gz','-scratch',work/'scratch']+flags)
            command(['dwi2fod','csd',dwi,wm,wm_fod,'-mask',brain,'-shells',shell,'-lmax','6']+flags)
        command(['mrconvert',wm_fod,fod]+flags)
        tensor=work/'diffusion_tensor.mif'
        command(['dwi2tensor',dwi,tensor,'-mask',brain]+flags)
        command(['tensor2metric',tensor,'-fa',fa,'-mask',brain]+flags)
        shutil.copy2(args.bval,reconstruction/'input.bval');shutil.copy2(args.bvec,reconstruction/'input.bvec')
    mask=np.asarray(nib.load(brain).dataobj)
    if not np.isfinite(mask).all() or not np.any(mask>0):raise ValueError('Brain observation mask is empty or invalid')
    profile.update(output_grid=validate_grid(nib.load(fod)),fod=str(fod),brain_mask=str(brain),
                   fa=str(fa) if fa.exists() else None,
                   input_requirement='Motion/distortion-corrected DWI with corresponding rotated FSL b-vectors',
                   measurement_mask='Brain observation mask only; no WM mask')
    profile['grid_standardization']=dict(target_spacing_mm=args.spacing,
        interpolation='Cubic DWI/FOD, nearest-neighbor mask, linear FA; matching direct-FOD grids are copied without interpolation.',
        native_resolution_note='Resampling standardizes the analysis grid; it does not recover spatial detail absent from the input acquisition.')
    (reconstruction/'reconstruction.json').write_text(json.dumps(profile,indent=2)+'\n')
    return profile

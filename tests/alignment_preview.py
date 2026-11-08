#!/usr/bin/env python3
"""Read-only, multi-slice alignment audit from captured prepared images."""
import argparse
from pathlib import Path
import json
import hashlib

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('Choose a fresh audit output folder')
    if args.input.resolve() in args.out.resolve().parents:parser.error('Keep audit output outside input')
    names=['FA.nii.gz','brain_mask.nii.gz','wm_mask.nii.gz','mean_b0.nii.gz',
           'JHU_labels_native.nii.gz','JHU_template_FA_native.nii.gz','T1_in_diffusion.nii.gz']
    images={name:nib.load(args.input/name) for name in names}
    reference=images[names[0]]
    for name,image in images.items():
        if image.shape!=reference.shape or not np.allclose(image.affine,reference.affine,rtol=0,atol=1e-5):
            raise ValueError('Alignment audit requires one shared grid: '+name)
    canonical={name:nib.as_closest_canonical(image) for name,image in images.items()}
    data={name:np.asarray(image.dataobj,dtype=np.float32) for name,image in canonical.items()}
    brain=data['brain_mask.nii.gz']>0
    coordinates=np.argwhere(brain)
    slices=np.rint(np.percentile(coordinates,[10,30,50,70,90],axis=0)).astype(int).T
    affine=canonical[names[0]].affine
    args.out.mkdir(parents=True)
    audit={}
    for mode in ('wm_on_fa','t1_b0_checker','jhu_fa_checker'):
        fig,axes=plt.subplots(3,5,figsize=(18,13),layout='constrained')
        for axis in range(3):
            for column,index in enumerate(slices[axis]):
                ax=axes[axis,column]
                plane=lambda name:np.take(data[name],index,axis=axis).T
                background=plane('mean_b0.nii.gz' if mode=='t1_b0_checker' else 'FA.nii.gz')
                upper=np.percentile(background[background>0],99) if mode=='t1_b0_checker' else .8
                ax.imshow(background,origin='lower',cmap='gray',vmin=0,vmax=upper)
                if mode=='wm_on_fa':
                    wm=plane('wm_mask.nii.gz')
                    if np.any(wm>0) and np.any(wm==0):ax.contour(wm,levels=[.5],colors=['#ffbd40'],linewidths=.55)
                else:
                    overlay=plane('T1_in_diffusion.nii.gz' if mode=='t1_b0_checker' else 'JHU_template_FA_native.nii.gz')
                    yy,xx=np.indices(overlay.shape)
                    masked=np.ma.masked_where((xx//12+yy//12)%2==0,overlay)
                    upper=np.percentile(overlay[overlay>0],99) if mode=='t1_b0_checker' else .8
                    ax.imshow(masked,origin='lower',cmap='gray',vmin=0,vmax=upper)
                    if mode=='jhu_fa_checker':
                        labels=plane('JHU_labels_native.nii.gz')
                        boundary=np.zeros(labels.shape,bool)
                        for dim in (0,1):boundary|=(labels!=np.roll(labels,1,dim))&(labels>0)
                        rgba=np.zeros(labels.shape+(4,));rgba[boundary]=[1,.74,.25,.9]
                        ax.imshow(rgba,origin='lower')
                position=float(affine[axis,axis]*index+affine[axis,3])
                ax.set_title(f'{("Sagittal R","Coronal A","Axial S")[axis]} = {position:.1f} mm',fontsize=10)
                left,right=('P','A') if axis==0 else ('L','R')
                for x,label,ha in ((.02,left,'left'),(.98,right,'right')):
                    ax.text(x,.03,label,transform=ax.transAxes,color='white',ha=ha,fontsize=8)
                ax.set_xticks([]);ax.set_yticks([])
        title={'wm_on_fa':'Shared FreeSurfer WM boundary on diffusion FA',
               't1_b0_checker':'Shared rigid T1 / mean b0 alignment; independent grayscale windows',
               'jhu_fa_checker':'Shared JHU borders and warped-template / subject FA checkerboard'}[mode]
        fig.suptitle(title+'\n10th, 30th, 50th, 70th and 90th brain-coordinate percentiles',fontsize=14)
        target=args.out/(mode+'.png');fig.savefig(target,dpi=160);plt.close(fig)
        audit[mode]=dict(path=target.name,slice_indices=slices.tolist())
    audit['input_sha256']={name:hashlib.sha256((args.input/name).read_bytes()).hexdigest() for name in names}
    audit['purpose']='Visual alignment inspection only; no source values or transforms changed'
    (args.out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':main()

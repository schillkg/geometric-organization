"""Shared-preparation contracts against the frozen routing/copy implementation."""
import argparse
import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import nibabel as nib
import numpy as np

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import shared

SOURCE=ROOT/'tests/frozen/prepare_dwi.py'
spec=importlib.util.spec_from_file_location('frozen_prepare_for_test',SOURCE)
frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
spec=importlib.util.spec_from_file_location('frozen_atlas_for_test',ROOT/'tests/frozen/atlas_registration.py')
frozen_atlas=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen_atlas)


class SharedPreparation(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.shape=(3,4,5)
        self.commands=[]

    def image(self,name,values,spacing=2):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        nib.save(nib.Nifti1Image(values,np.diag([spacing,spacing,spacing,1.])),path)
        return path

    def args(self,mode='fod',spacing=None):
        fod=self.image('fod.nii.gz',np.ones(self.shape+(28,),np.float32))
        brain=self.image('brain.nii.gz',np.ones(self.shape,np.uint8))
        fa=self.image('FA.nii.gz',np.full(self.shape,.6,np.float32))
        return argparse.Namespace(fod=fod if mode=='fod' else None,dwi=fod if mode=='dwi' else None,
            bval=None,bvec=None,brain_mask=brain,fa=fa if mode=='fod' else None,b0=None,
            spacing=spacing,threads=2,modules=['sheets','laminarity','dispersion'])

    def command(self,argv,**kwargs):
        self.commands.append(list(map(str,argv)))
        if argv[0]!='mrconvert':raise AssertionError('Unexpected external command: '+str(argv))
        shutil.copy2(argv[1],argv[2])
        return ''

    def test_direct_two_mm_uses_frozen_no_interpolation_branch_exactly_once(self):
        args=self.args()
        with mock.patch.object(shared,'module',return_value=frozen), \
                mock.patch.object(frozen,'prepare',wraps=frozen.prepare) as prepare:
            result=shared.prepare(args,self.root/'out',self.root/'work',self.command)
        prepare.assert_called_once()
        self.assertEqual(prepare.call_args.args[0].spacing,2)
        self.assertFalse(result['resampled'])
        self.assertTrue(result['supplied_fod_grid_preserved'])
        self.assertTrue(all(command[0]=='mrconvert' for command in self.commands))
        original=nib.load(args.fod);saved=nib.load(self.root/'out/reconstruction/WM_FOD.nii.gz')
        np.testing.assert_array_equal(original.affine,saved.affine)
        np.testing.assert_array_equal(np.asarray(original.dataobj),np.asarray(saved.dataobj))

    def test_conflicting_prepared_spacing_stops_before_frozen_preparation(self):
        args=self.args(spacing=1.25)
        with mock.patch.object(shared,'module',return_value=frozen), \
                mock.patch.object(frozen,'prepare') as prepare:
            with self.assertRaisesRegex(ValueError,'conflicts with the prepared FOD grid'):
                shared.prepare(args,self.root/'out',self.root/'work',self.command)
        prepare.assert_not_called()
        self.assertEqual(len(self.commands),1)

    def test_dwi_default_and_override_pass_one_explicit_reconstruction_request(self):
        for index,(requested,expected,native_spacing,csd_mode) in enumerate(
                ((None,1.25,False,'auto'),(1.,1.,False,'msmt'),(None,2.,True,'sscsd'))):
            with self.subTest(spacing=requested,native_spacing=native_spacing,csd_mode=csd_mode):
                args=self.args(mode='dwi',spacing=requested)
                args.native_spacing=native_spacing;args.csd_mode=csd_mode
                destination=self.root/f'dwi-{index}'
                captured=[]
                def prepared(request,out,work,command):
                    captured.append(request)
                    reference=nib.Nifti1Image(np.ones(self.shape+(28,),np.float32),
                                             np.diag([request.spacing]*3+[1.]))
                    shared.save(out/'reconstruction/WM_FOD.nii.gz',np.asarray(reference.dataobj),reference)
                    shared.save(out/'anatomy/brain_mask.nii.gz',np.ones(self.shape,np.uint8),reference)
                    shared.save(out/'anatomy/FA.nii.gz',np.ones(self.shape,np.float32),reference)
                    return {'mode':'fixture-only-routing-spy'}
                with mock.patch.object(shared,'module',return_value=frozen), \
                        mock.patch.object(frozen,'prepare',side_effect=prepared) as prepare:
                    result=shared.prepare(args,destination,destination.with_name(destination.name+'-work'),self.command)
                prepare.assert_called_once()
                self.assertEqual(captured[0].spacing,expected)
                self.assertEqual(captured[0].csd_mode,csd_mode)
                self.assertIsNone(captured[0].fod)
                self.assertEqual(captured[0].mask,args.brain_mask)
                self.assertFalse(result['supplied_fod_grid_preserved'])
                self.assertEqual(result.get('native_spacing_requested',False),native_spacing)

    def test_native_spacing_rejects_anisotropy_shear_and_conflicting_target(self):
        for index,affine in enumerate((np.diag([1.,1.25,1.,1.]),
                                      np.array([[1.,.2,0,0],[0,1.,0,0],[0,0,1.,0],[0,0,0,1.]]))):
            args=self.args(mode='dwi');args.native_spacing=True
            image=nib.load(args.dwi);nib.save(nib.Nifti1Image(np.asarray(image.dataobj),affine),args.dwi)
            with mock.patch.object(shared,'module',return_value=frozen), mock.patch.object(frozen,'prepare') as prepare:
                with self.assertRaises(ValueError):
                    shared.prepare(args,self.root/f'bad-{index}',self.root/f'bad-work-{index}',self.command)
            prepare.assert_not_called()
        args=self.args(mode='dwi',spacing=1.);args.native_spacing=True
        with mock.patch.object(shared,'module',return_value=frozen), mock.patch.object(frozen,'prepare') as prepare:
            with self.assertRaisesRegex(ValueError,'not both'):
                shared.prepare(args,self.root/'conflict',self.root/'conflict-work',self.command)
        prepare.assert_not_called()

    def test_native_spacing_supplied_fod_still_preserves_grid(self):
        args=self.args();args.native_spacing=True
        with mock.patch.object(shared,'module',return_value=frozen):
            record=shared.prepare(args,self.root/'out',self.root/'work',self.command)
        self.assertFalse(record['resampled'])
        self.assertEqual(record['output_grid']['spacing_mm'],[2.,2.,2.])

    def test_reused_jhu_labels_use_frozen_erosion_without_registration(self):
        prepared=self.root/'prepared'
        self.image('prepared/anatomy/FA.nii.gz',np.ones(self.shape,np.float32))
        self.image('prepared/anatomy/brain_mask.nii.gz',np.ones(self.shape,np.uint8))
        labels=np.ones(self.shape,np.uint16);labels[0,0,0]=50
        source=self.image('supplied_labels.nii.gz',labels);before=source.read_bytes()
        atlas=self.root/'engine/atlas';atlas.mkdir(parents=True);(atlas/'atlas_manifest.json').write_text('{}')
        names={value:f'Region {value}' for value in range(1,51)}
        with mock.patch.object(shared,'DISPERSION',atlas.parent), mock.patch.object(shared,'module',return_value=frozen_atlas), \
                mock.patch.object(frozen_atlas,'atlas_names',return_value=names), mock.patch.object(frozen_atlas,'register_labels') as register:
            result=shared.jhu(prepared,self.root/'regions',2,source)
        register.assert_not_called();self.assertEqual(source.read_bytes(),before)
        self.assertFalse(result['registration_performed']);self.assertFalse(result['warped_template_available'])
        self.assertEqual(result['sha256'],hashlib.sha256(before).hexdigest())
        self.assertFalse((self.root/'regions/registration').exists())
        self.assertFalse((self.root/'regions/JHU_template_FA_native.nii.gz').exists())
        np.testing.assert_array_equal(nib.load(self.root/'regions/JHU_labels_native.nii.gz').dataobj,labels)
        expected=np.zeros(self.shape,np.uint16);expected[1:-1,1:-1,1:-1]=1
        np.testing.assert_array_equal(nib.load(self.root/'regions/JHU_labels_eroded_1voxel.nii.gz').dataobj,expected)
        self.assertEqual(len((self.root/'regions/labels.csv').read_text().splitlines()),51)

    def test_reused_jhu_rejects_invalid_labels_and_grid(self):
        prepared=self.root/'prepared';self.image('prepared/anatomy/FA.nii.gz',np.ones(self.shape,np.float32))
        atlas=self.root/'engine/atlas';atlas.mkdir(parents=True)
        names={value:str(value) for value in range(1,51)}
        for index,(value,spacing) in enumerate(((0.,2.),(-1.,2.),(51.,2.),(1.5,2.),(float('nan'),2.),(1.,1.))):
            source=self.image(f'labels-{index}.nii.gz',np.full(self.shape,value,np.float32),spacing)
            with mock.patch.object(shared,'DISPERSION',atlas.parent), mock.patch.object(shared,'module',return_value=frozen_atlas), \
                    mock.patch.object(frozen_atlas,'atlas_names',return_value=names), mock.patch.object(frozen_atlas,'register_labels') as register:
                with self.assertRaises(ValueError):shared.jhu(prepared,self.root/f'regions-{index}',2,source)
            register.assert_not_called()

    def test_default_jhu_route_still_registers_once_and_keeps_template(self):
        prepared=self.root/'prepared';self.image('prepared/anatomy/FA.nii.gz',np.ones(self.shape,np.float32))
        atlas=self.root/'engine/atlas';atlas.mkdir(parents=True);(atlas/'atlas_manifest.json').write_text('{}')
        names={value:str(value) for value in range(1,51)}
        def registered(fa,brain,atlas,out,threads):
            labels=self.image('regions/JHU_labels_native.nii.gz',np.ones(self.shape,np.uint16))
            self.image('regions/registration/template_in_native_FA.nii.gz',np.ones(self.shape,np.float32))
            (out/'registration/commands.log').write_text('frozen registration commands\n')
            return labels,dict(method='fixture registration',threads=threads)
        with mock.patch.object(shared,'DISPERSION',atlas.parent), mock.patch.object(shared,'module',return_value=frozen_atlas), \
                mock.patch.object(frozen_atlas,'atlas_names',return_value=names), mock.patch.object(frozen_atlas,'register_labels',side_effect=registered) as register:
            record=shared.jhu(prepared,self.root/'regions',3)
        register.assert_called_once();self.assertEqual(record['method'],'fixture registration')
        self.assertTrue((self.root/'regions/JHU_template_FA_native.nii.gz').is_file())
        self.assertEqual((self.root/'regions/commands.log').read_text(),'frozen registration commands\n')

    def test_freesurfer_mgh_to_nifti_preserves_values_affine_and_requested_dtype(self):
        affine=np.array([[0,0,-1.25,15],[1.25,0,0,-20],[0,-1.25,0,30],[0,0,0,1.]])
        for dtype in (np.float32,np.uint8):
            with self.subTest(dtype=dtype):
                original=np.arange(np.prod(self.shape),dtype=np.float32).reshape(self.shape)
                source=self.root/('source-'+np.dtype(dtype).name+'.mgz')
                nib.save(nib.MGHImage(original,affine),source)
                reference=nib.load(source)
                values=original.astype(dtype)
                target=self.root/('converted-'+np.dtype(dtype).name+'.nii.gz')
                shared.save(target,values,reference)
                result=nib.load(target)
                self.assertIsInstance(result.header,nib.Nifti1Header)
                self.assertEqual(result.get_data_dtype(),np.dtype(dtype))
                np.testing.assert_array_equal(result.affine,reference.affine)
                np.testing.assert_array_equal(np.asarray(result.dataobj),values)


class FrozenShellRouting(unittest.TestCase):
    def test_portable_frozen_helpers_match_recorded_source_hashes(self):
        manifest=json.loads((ROOT/'tests/frozen/source_manifest.json').read_text())
        for name,record in manifest.items():
            payload=(ROOT/'tests/frozen'/name).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(),record['sha256'])
            self.assertEqual(len(payload),record['bytes'])

    def profile(self,shells):
        count=64;i=np.arange(count);z=1-2*(i+.5)/count;angle=i*np.pi*(3-np.sqrt(5))
        directions=np.column_stack((np.sqrt(1-z*z)*np.cos(angle),np.sqrt(1-z*z)*np.sin(angle),z))
        gradient=[np.zeros((1,4))];indices=[[0]]
        for index,b in enumerate(shells):
            gradient.append(np.column_stack((directions,np.full(count,b))))
            indices.append(list(range(1+index*count,1+(index+1)*count)))
        return np.vstack(gradient),[0,*shells],[1,*([count]*len(shells))],indices,[0]

    def test_single_shell_is_tournier_lmax6_and_multishell_keeps_every_shell(self):
        single=frozen.reconstruction_profile(*self.profile([1000]))
        self.assertEqual((single['mode'],single['wm_lmax']),('sscsd',6))
        data=self.profile([1000,2000,3000]);multi=frozen.reconstruction_profile(*data)
        self.assertEqual((multi['mode'],multi['wm_lmax']),('msmt',8))
        self.assertEqual(multi['shell_indices'],data[3])
        self.assertEqual(multi['shell_means'],[0,1000,2000,3000])
        with self.assertRaises(ValueError):frozen.reconstruction_profile(*self.profile([1000]),requested='msmt')
        self.assertEqual(frozen.reconstruction_profile(*self.profile([1000]),requested='sscsd')['mode'],'sscsd')
        self.assertEqual(frozen.reconstruction_profile(*self.profile([1000,2000]),requested='msmt')['mode'],'msmt')
        with self.assertRaises(ValueError):frozen.reconstruction_profile(*self.profile([1000,2000]),requested='sscsd')

    def test_insufficient_highest_shell_rank_is_rejected_without_order_reduction(self):
        gradient,means,sizes,indices,bzero=self.profile([1000,2000])
        gradient=gradient[:1+64+20];indices[-1]=indices[-1][:20];sizes[-1]=20
        with self.assertRaisesRegex(ValueError,'cannot identify lmax 8'):
            frozen.reconstruction_profile(gradient,means,sizes,indices,bzero)


if __name__=='__main__':unittest.main()

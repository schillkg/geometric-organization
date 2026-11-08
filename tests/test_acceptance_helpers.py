"""Exercise isolated comparison/preflight helpers without launching a container."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

spec=importlib.util.spec_from_file_location('acceptance',Path(__file__).with_name('acceptance.py'))
acceptance=importlib.util.module_from_spec(spec);spec.loader.exec_module(acceptance)


class AcceptanceHelpers(unittest.TestCase):
    def test_exact_compare_preserves_nan_zero_and_detects_small_float_change(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);affine=np.eye(4)
            values=np.asarray([np.nan,0.,1.],dtype=np.float32).reshape(1,1,3)
            for key in ('left','right'): nib.save(nib.Nifti1Image(values,affine),root/(key+'.nii.gz'))
            job=dict(tolerances={'absolute':0,'relative':0},pairs=[dict(module='sheets',relative='q.nii.gz',
                left=str(root/'left.nii.gz'),right=str(root/'right.nii.gz'))])
            (root/'job.json').write_text(json.dumps(job))
            def compare():
                subprocess.run([sys.executable,'-c',acceptance.CHECK_IMAGES,root/'job.json',root/'result.json'],
                               check=True,capture_output=True,text=True)
                return json.loads((root/'result.json').read_text())
            self.assertEqual(compare()['status'],'passed')
            changed=values.copy();changed[0,0,2]=np.nextafter(np.float32(1),np.float32(2))
            nib.save(nib.Nifti1Image(changed,affine),root/'right.nii.gz')
            result=compare()
            self.assertEqual(result['status'],'failed')
            self.assertEqual(result['comparisons'][0]['unequal_finite_values'],1)
            changed=values.copy();changed[0,0,0]=0
            nib.save(nib.Nifti1Image(changed,affine),root/'right.nii.gz')
            self.assertFalse(compare()['comparisons'][0]['nan_mask_equal'])

    def test_fixture_preflight_is_bounded_to_the_agreed_small_grid(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);affine=np.diag([2.,2.,2.,1.])
            for key,shape in [('fod',(16,18,20,28)),('brain',(16,18,20)),('fa',(16,18,20))]:
                nib.save(nib.Nifti1Image(np.ones(shape,np.float32),affine),root/(key+'.nii.gz'))
            command=[sys.executable,'-c',acceptance.CHECK_FIXTURE,
                     root/'fod.nii.gz',root/'brain.nii.gz',root/'fa.nii.gz',root/'result.json']
            subprocess.run(command,check=True,capture_output=True,text=True)
            self.assertEqual(json.loads((root/'result.json').read_text())['spacing_mm'],[2,2,2])
            nib.save(nib.Nifti1Image(np.ones((16,18,20,28),np.float32),np.eye(4)),root/'fod.nii.gz')
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)

    def test_empty_q_cannot_pass_while_a_genuine_zero_measurement_can(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);affine=np.eye(4)
            paths=['sheets/q.nii.gz','laminarity/fixel_maps/lamellarity.nii.gz',
                   'laminarity/fixel_maps/tubularity.nii.gz','dispersion/fixel_maps/odi_total.nii.gz']
            for relative in paths:
                path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
                values=np.ones((1,1,1,3),np.float32)
                if relative==paths[0]:values[:]=np.nan
                nib.save(nib.Nifti1Image(values,affine),path)
            command=[sys.executable,'-c',acceptance.CHECK_COVERAGE,root,root/'result.json']
            subprocess.run(command,check=True,capture_output=True,text=True)
            self.assertEqual(json.loads((root/'result.json').read_text())['status'],'failed')
            q=np.full((1,1,1,3),np.nan,np.float32);q[...,0]=0
            nib.save(nib.Nifti1Image(q,affine),root/paths[0])
            subprocess.run(command,check=True,capture_output=True,text=True)
            result=json.loads((root/'result.json').read_text())
            self.assertEqual(result['status'],'passed')
            self.assertEqual(result['q']['range'],[0,0])


if __name__=='__main__': unittest.main()

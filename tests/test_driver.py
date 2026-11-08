"""Local lifecycle/publication checks; external estimators are not exercised.

Scientific acceptance still requires a full original `sheets --fod SHARED_FOD
--mask SHARED_BRAIN --wm-mask SHARED_WM --fa SHARED_FA --no-jhu` run with the same
threads/settings as sheets_module.py, then exact q/missingness/count comparison.
Shared 50-label regional outputs intentionally replace standalone registration;
test their unchanged statistics using the exact shared native/eroded label maps.
"""
import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import nibabel as nib
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import publish
import shared

spec = importlib.util.spec_from_file_location("geometric_driver", ROOT / "main.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class TemporaryProject(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        environment = mock.patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        self.root = Path(temporary.name)
        self.affine = np.diag([1.25, 1.25, 1.25, 1.])
        self.shape = (3, 4, 5)

    def image(self, path, values=None, affine=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if values is None:
            values = np.ones(self.shape, np.float32)
        nib.save(nib.Nifti1Image(values, self.affine if affine is None else affine), path)
        return path

    def text(self, path, text="fixture"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path


class PublicationTests(TemporaryProject):
    def fixture(self):
        work = self.root / "work"
        self.image(work / "prepared/anatomy/brain_mask.nii.gz", np.ones(self.shape, np.uint8))
        self.image(work / "prepared/anatomy/FA.nii.gz")
        self.image(work / "prepared/reconstruction/WM_FOD.nii.gz",
                   np.ones(self.shape + (28,), np.float32))
        self.text(work / "prepared/reconstruction/reconstruction.json", '{}')
        self.text(work / "prepared/reconstruction/responses/wm.txt", '1 0.5')
        self.text(work / "native/fixels/index.mif", 'original sparse indices')
        dirs = {name: work / "modules" / name for name in ("sheets", "laminarity", "dispersion")}
        scalar = np.zeros(self.shape, np.float32)
        scalar[0, 0, 0] = np.nan
        for name, folder in dirs.items():
            self.image(folder / "anatomy/brain_mask.nii.gz", np.ones(self.shape, np.uint8))
            self.image(folder / "anatomy/FA.nii.gz")
            self.text(folder / "run.json", json.dumps({'module': name}))
            self.text(folder / "work/huge-intermediate.bin", 'owned scratch')
            self.text(folder / "qa.pdf", 'discard duplicate module report')
            self.image(folder / ("q.nii.gz" if name == "sheets" else "voxel_maps/measure.nii.gz"), scalar)
            if name != "sheets":
                for filename in publish.IDENTITY:
                    values = np.ones(self.shape + ((3, 3) if filename == 'fiber_direction.nii.gz' else (3,)), np.float32)
                    if filename == 'native_fixel_id.nii.gz':
                        values = np.arange(np.prod(self.shape)*3,dtype=np.int32).reshape(self.shape+(3,))
                    elif filename == 'fiber_direction.nii.gz':
                        values[...,1:] = 0
                    self.image(folder / "fixel_maps" / filename, values)
                self.text(folder / 'provenance/shared_native.json', json.dumps(dict(
                    status='complete',files={'extraction':'same frozen source'},
                    inputs={'fod':{'sha256':'same-fod'},'brain_mask':{'sha256':'same-mask'}})))
        reports = work / "reports"
        self.text(reports / "00_overview.pdf", 'existing report artifact')
        args = argparse.Namespace(save_fod=False, save_native_fixels=False, save_registration=False)
        return args, work, dirs, reports

    def test_compact_publish_deduplicates_identity_and_preserves_missingness(self):
        args, work, dirs, reports = self.fixture()
        before = publish.inventory(work)
        destination = self.root / 'published'
        layout = publish.build(args, work, destination, dirs, reports)
        self.assertEqual(before, publish.inventory(work))
        self.assertEqual(len(list((destination / 'shared/fixels').glob('*.nii.gz'))), 3)
        self.assertFalse((destination / 'laminarity/fixel_maps/afd.nii.gz').exists())
        self.assertFalse((destination / 'extras').exists())
        self.assertFalse(any('work' in p.relative_to(destination).parts for p in destination.rglob('*')))
        q = np.asarray(nib.load(destination / 'sheets/q.nii.gz').dataobj)
        self.assertTrue(np.isnan(q[0, 0, 0]))
        self.assertEqual(q[1, 0, 0], 0)
        self.assertEqual(layout['laminarity']['fixel_maps/afd.nii.gz'],
                         layout['dispersion']['fixel_maps/afd.nii.gz'])
        restored = self.root / 'restored'
        publish.restore(destination, 'dispersion', restored)
        self.assertTrue((restored / 'fixel_maps/afd.nii.gz').is_symlink())
        self.assertTrue(publish.equal_image(restored / 'voxel_maps/measure.nii.gz',
                                            dirs['dispersion'] / 'voxel_maps/measure.nii.gz'))

    def test_same_name_cannot_hide_nan_zero_or_grid_difference(self):
        a = self.image(self.root / 'a.nii.gz')
        values = np.ones(self.shape, np.float32)
        values[0, 0, 0] = np.nan
        b = self.image(self.root / 'b.nii.gz', values)
        destination = self.root / 'saved/map.nii.gz'
        publish.copy_checked(a, destination)
        with self.assertRaises(ValueError):
            publish.copy_checked(b, destination)
        changed_affine = self.affine.copy()
        changed_affine[0, 3] = .1
        self.image(b, affine=changed_affine)
        with self.assertRaises(ValueError):
            publish.copy_checked(b, destination)
        self.assertTrue(publish.equal_image(a, destination))

    def test_cross_module_identity_and_grid_checks_fail_before_publication(self):
        args, work, dirs, reports = self.fixture()
        result = driver.check_outputs(work / 'prepared', dirs, None)
        self.assertEqual(result['status'], 'passed')
        for filename in ('afd.nii.gz','fiber_direction.nii.gz','native_fixel_id.nii.gz'):
            with self.subTest(filename=filename):
                path = dirs['dispersion'] / 'fixel_maps' / filename
                values = np.asarray(nib.load(path).dataobj).copy()
                if filename == 'afd.nii.gz': values[0,0,0,0] = 2
                elif filename == 'fiber_direction.nii.gz': values[0,0,0,0] = [0,1,0]
                else: values[0,0,0,0],values[1,0,0,0] = values[1,0,0,0],values[0,0,0,0]
                self.image(path, values)
                with self.assertRaisesRegex(ValueError, 'Same native ID has inconsistent'):
                    driver.check_outputs(work / 'prepared', dirs, None)
                shutil.copy2(dirs['laminarity'] / 'fixel_maps' / filename,path)
        shifted = self.affine.copy(); shifted[1, 3] = .2
        self.image(dirs['sheets'] / 'q.nii.gz', affine=shifted)
        with self.assertRaisesRegex(ValueError, 'output grid differs'):
            driver.check_outputs(work / 'prepared', dirs, None)

    def test_distinct_reported_sets_preserve_each_module_identity_maps(self):
        args, work, dirs, reports = self.fixture()
        for filename in publish.IDENTITY:
            path = dirs['dispersion'] / 'fixel_maps' / filename
            values = np.asarray(nib.load(path).dataobj).copy()
            values[0,0,0,0] = -1 if filename=='native_fixel_id.nii.gz' else np.nan
            self.image(path,values)
        result = driver.check_outputs(work / 'prepared',dirs,None)
        self.assertFalse(result['native_identity']['shared_reported_maps'])
        self.assertEqual(result['native_identity']['laminarity_only_fixels'],1)
        destination = self.root/'published'
        layout = publish.build(args,work,destination,dirs,reports)
        self.assertNotEqual(layout['laminarity']['fixel_maps/afd.nii.gz'],
                            layout['dispersion']['fixel_maps/afd.nii.gz'])
        self.assertFalse((destination/'shared/fixels').exists())
        for module in ('laminarity','dispersion'):
            for filename in publish.IDENTITY:
                self.assertTrue(publish.equal_image(dirs[module]/'fixel_maps'/filename,
                                  destination/layout[module]['fixel_maps/'+filename]))

    def test_retention_flags_save_original_fod_and_native_extraction(self):
        args, work, dirs, reports = self.fixture()
        args.save_fod = args.save_native_fixels = True
        destination = self.root / 'published'
        publish.build(args, work, destination, dirs, reports)
        self.assertEqual(publish.sha256(work / 'prepared/reconstruction/WM_FOD.nii.gz'),
                         publish.sha256(destination / 'extras/fod/WM_FOD.nii.gz'))
        self.assertEqual((destination / 'extras/native_fixels/index.mif').read_text(), 'original sparse indices')

    def test_changed_file_or_symlink_cannot_pass_inventory(self):
        folder = self.root / 'files'
        path = self.text(folder / 'value.txt', 'original')
        records = publish.inventory(folder)
        path.write_text('modified')
        with self.assertRaises(ValueError):
            publish.verify(folder, records)
        original = self.text(self.root / 'original.txt', 'original')
        path.unlink()
        path.symlink_to(original)
        with self.assertRaises(ValueError):
            publish.verify(folder, records)

    def test_restored_view_rejects_source_or_destination_escape(self):
        folder = self.root / 'published'
        self.text(folder / 'safe.txt')
        self.text(self.root / 'outside.txt')
        for number, mapping in enumerate(({'value.txt': '../outside.txt'}, {'../escape.txt': 'safe.txt'})):
            self.text(folder / 'provenance/layout.json', json.dumps({'sheets': mapping}))
            with self.assertRaises(ValueError):
                publish.restore(folder, 'sheets', self.root / f'restored{number}')
        self.assertFalse((self.root / 'escape.txt').exists())


class FinalizationTests(TemporaryProject):
    def fixture(self):
        out = self.root / 'subject'
        work = out / 'work'
        self.text(work / '.owner.json', json.dumps({'run_id': 'owned-run'}))
        for name in ('a.txt', 'maps/b.txt', 'maps/c.txt'):
            self.text(work / 'public' / name, name)
        state = dict(run_id='owned-run', stages={'publish': {'status': 'complete', 'files': {}}},
                     public_files=publish.inventory(work / 'public'), status='finalizing', publication_ready=True)
        return out, state

    def test_interrupted_publication_resumes_without_overwriting_completed_files(self):
        out, state = self.fixture()
        original_replace = Path.replace
        calls = []
        def interrupted_replace(path, target):
            calls.append(str(target))
            if str(target).endswith('maps/b.txt'):
                raise OSError('simulated interrupted move')
            return original_replace(path, target)
        with mock.patch.object(Path, 'replace', interrupted_replace):
            with self.assertRaises(OSError):
                driver.finalize(out, state, False)
        self.assertTrue((out / 'a.txt').is_file())
        self.assertTrue((out / 'work/public/maps/b.txt').is_file())
        original_time = (out / 'a.txt').stat().st_mtime_ns
        driver.finalize(out, state, False)
        self.assertEqual((out / 'a.txt').stat().st_mtime_ns, original_time)
        self.assertEqual(state['status'], 'complete')
        self.assertFalse((out / 'work').exists())
        publish.verify(out, state['public_files'])
        driver.finalize(out, state, False)
        self.assertEqual(state['status'], 'complete')

    def test_modified_public_target_is_not_overwritten_or_cleaned(self):
        out, state = self.fixture()
        self.text(out / 'a.txt', 'unrelated result')
        with self.assertRaises(ValueError):
            driver.finalize(out, state, False)
        self.assertEqual((out / 'a.txt').read_text(), 'unrelated result')
        self.assertTrue((out / 'work/.owner.json').exists())

    def test_foreign_work_owner_prevents_cleanup(self):
        out, state = self.fixture()
        self.text(out / 'work/.owner.json', json.dumps({'run_id': 'other-run'}))
        with self.assertRaises(ValueError):
            driver.finalize(out, state, False)
        self.assertTrue((out / 'work').exists())

    def test_partial_cleanup_can_resume_after_inner_owner_is_removed(self):
        out, state = self.fixture()
        def interrupted_cleanup(path, *args, **kwargs):
            (Path(path) / '.owner.json').unlink()
            raise OSError('simulated partial recursive cleanup')
        with mock.patch.object(driver.shutil, 'rmtree', interrupted_cleanup):
            with self.assertRaises(OSError):
                driver.finalize(out, state, False)
        cleanup = out / '.cleanup-owned-run'
        marker = out / '.cleanup-owned-run.json'
        self.assertTrue(cleanup.is_dir())
        self.assertFalse((cleanup / '.owner.json').exists())
        self.assertTrue(marker.is_file())
        self.assertEqual(json.loads(marker.read_text())['run_id'], 'owned-run')
        driver.finalize(out, state, False)
        self.assertFalse(cleanup.exists())
        self.assertFalse(marker.exists())
        self.assertEqual(state['status'], 'complete')

    def test_arbitrary_cleanup_tree_without_matching_external_owner_survives(self):
        for bad_marker in (None, {'run_id': 'other-run', 'directory': '.cleanup-owned-run'}):
            with self.subTest(marker=bad_marker):
                out = self.root / ('missing-marker' if bad_marker is None else 'foreign-marker')
                self.text(out / '.cleanup-owned-run/unrelated.txt', 'preserve')
                if bad_marker:
                    self.text(out / '.cleanup-owned-run.json', json.dumps(bad_marker))
                state = dict(run_id='owned-run', stages={}, public_files={}, status='finalizing')
                with self.assertRaises(ValueError):
                    driver.finalize(out, state, False)
                self.assertEqual((out / '.cleanup-owned-run/unrelated.txt').read_text(), 'preserve')


class DriverResumeTests(TemporaryProject):
    def arguments(self, extra=()):
        source = self.root / 'inputs'
        for name in ('fod.nii.gz', 'brain.nii.gz', 'FA.nii.gz'):
            if not (source / name).exists():
                self.image(source / name)
        return driver.parser().parse_args([
            '--fod', str(source / 'fod.nii.gz'), '--brain-mask', str(source / 'brain.nii.gz'),
            '--fa', str(source / 'FA.nii.gz'), '--out', str(self.root / 'subject'),
            '--modules', 'dispersion', '--no-jhu', *extra])

    def test_failure_preserves_owned_work_and_changed_recipe_cannot_resume(self):
        fake_source = self.root / 'source'
        self.text(fake_source / 'adapter.py', '# original source\n')
        args = self.arguments()
        original_input = publish.sha256(args.fod)
        def fail_prepare(args, destination, work, command):
            self.text(destination / 'interrupted.txt', 'failure evidence')
            self.text(work / 'command.log', 'partial command')
            raise RuntimeError('fixture preparation failure')
        with mock.patch.object(driver, 'ROOT', fake_source), mock.patch.object(shared, 'prepare', fail_prepare):
            with self.assertRaises(RuntimeError):
                driver.run(args)
            state = json.loads((args.out / 'run.json').read_text())
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(json.loads((args.out / 'work/.owner.json').read_text())['run_id'], state['run_id'])
            self.assertEqual(publish.sha256(args.fod), original_input)
            with self.assertRaisesRegex(ValueError, 'settings, or source changed'):
                driver.run(self.arguments(['--resume', '--threads', '2']))
            self.text(fake_source / 'adapter.py', '# changed source\n')
            with self.assertRaisesRegex(ValueError, 'settings, or source changed'):
                driver.run(self.arguments(['--resume']))
            self.text(fake_source / 'adapter.py', '# original source\n')
            self.image(args.fod, np.full(self.shape, 2, np.float32))
            with self.assertRaisesRegex(ValueError, 'settings, or source changed'):
                driver.run(self.arguments(['--resume']))
        self.assertEqual((args.out / 'work/prepared/interrupted.txt').read_text(), 'failure evidence')

    def test_resume_reuses_verified_preparation_and_preserves_failed_native_stage(self):
        fake_source = self.root / 'source'
        self.text(fake_source / 'adapter.py', '# source\n')
        calls = []
        def prepare(args, destination, work, command):
            calls.append('prepare')
            self.text(destination / 'reconstruction/reconstruction.json', '{}')
            self.text(work / 'preparation.log', 'prepared')
        def native_failure(argv, **kwargs):
            native = Path(argv[argv.index('--native') + 1])
            self.text(native / 'interrupted.txt', 'native failure evidence')
            return subprocess.CompletedProcess(argv, 1, '', 'native fixture failure')
        with mock.patch.object(driver, 'ROOT', fake_source), mock.patch.object(shared, 'prepare', prepare), \
                mock.patch.object(driver.subprocess, 'run', native_failure):
            with self.assertRaises(RuntimeError):
                driver.run(self.arguments())
            with self.assertRaises(RuntimeError):
                driver.run(self.arguments(['--resume']))
            self.text(self.root / 'subject/work/prepared/reconstruction/reconstruction.json', '{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'Missing or changed saved file'):
                driver.run(self.arguments(['--resume']))
        out = self.root / 'subject'
        state = json.loads((out / 'run.json').read_text())
        self.assertEqual(calls, ['prepare'])
        self.assertEqual(state['stages']['prepare']['status'], 'complete')
        self.assertTrue(list((out / 'work/failures').rglob('interrupted.txt')))
        self.assertEqual((out / 'work/native/interrupted.txt').read_text(), 'native failure evidence')

    def test_cli_rejects_incompatible_modes_and_unselected_retention(self):
        examples = [
            ['--modules', 'unknown'],
            ['--modules', 'dispersion,dispersion'],
            ['--modules', 'sheets'],
            ['--modules', 'dispersion', '--sheets-save-draws'],
            ['--modules', 'dispersion', '--threads', '0'],
            ['--modules', 'dispersion', '--spacing', 'nan'],
            ['--modules', 'dispersion', '--bval', 'bval'],
        ]
        for suffix in examples:
            with self.subTest(arguments=suffix), mock.patch.object(sys, 'argv', [
                'geometric-organization', '--fod', 'fod.nii.gz', '--brain-mask', 'mask.nii.gz',
                '--fa', 'FA.nii.gz', '--out', 'output', *suffix]), \
                    mock.patch.object(driver, 'run') as run, mock.patch('sys.stderr'):
                with self.assertRaises(SystemExit) as error:
                    driver.main()
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()

    def test_mask_validation_rejects_nonfinite_nonbinary_and_shifted_grids(self):
        reference = nib.Nifti1Image(np.ones(self.shape + (28,), np.float32), self.affine)
        path = self.root / 'mask.nii.gz'
        bad = np.ones(self.shape, np.float32)
        bad[0, 0, 0] = np.nan
        self.image(path, bad)
        with self.assertRaises(ValueError): shared.on_grid(path, reference, binary=True)
        bad[0, 0, 0] = 2
        self.image(path, bad)
        with self.assertRaises(ValueError): shared.on_grid(path, reference, binary=True)
        shifted = self.affine.copy(); shifted[2, 3] = .2
        self.image(path, np.ones(self.shape, np.uint8), shifted)
        with self.assertRaises(ValueError): shared.on_grid(path, reference, binary=True)


if __name__ == '__main__':
    unittest.main()

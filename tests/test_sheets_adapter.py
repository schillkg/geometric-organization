"""Shared-region/compact-report checks; no estimator is replaced or rerun."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sheets_adapter", ROOT / "sheets_module.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class SharedRegionsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.out = self.root / "sheets"
        self.out.mkdir()
        (self.out / "work").mkdir()
        self.engine = adapter.engine()
        self.affine = np.diag([1.25, 1.25, 1.25, 1.])
        self.shape = (6, 6, 6)
        brain = np.ones(self.shape, np.uint8)
        q = np.broadcast_to(np.arange(6, dtype=np.float32)[:, None, None, None] * .01,
                            self.shape + (3,)).copy()
        peaks = np.broadcast_to(np.eye(3, dtype=np.float32).reshape(9), self.shape + (9,)).copy()
        tensor, available = self.engine.tensor_from_peaks(q, peaks, .03)
        fields = {
            "q.nii.gz": q, "wm_mask.nii.gz": brain, "anatomy/brain_mask.nii.gz": brain,
            "anatomy/FA.nii.gz": np.full(self.shape, .6, np.float32),
            "diagnostics/reference_peaks.nii.gz": peaks,
            "diagnostics/peak_count.nii.gz": np.full(self.shape, 3, np.uint8),
            "diagnostics/valid_draws.nii.gz": np.full(q.shape, 100, np.uint32),
            "diagnostics/available.nii.gz": np.ones(q.shape, np.uint8),
            "diagnostics/q_iqr.nii.gz": np.zeros(q.shape, np.float32),
            "diagnostics/support.nii.gz": np.ones(q.shape, np.float32),
            "legacy_SPI/valid_draws.nii.gz": np.full(q.shape, 100, np.uint32),
            "legacy_SPI/SPI_0008.nii.gz": np.full(q.shape, .4, np.float32),
            "legacy_SPI/SPI_0024.nii.gz": np.full(q.shape, .8, np.float32),
            "display/Srelative.nii.gz": 1 - q / .03,
            "display/Sbounded.nii.gz": .03 / (.03 + q), "tensors/available.nii.gz": available,
        }
        for i, pair in enumerate(("12", "13", "23")):
            fields[f"tensors/pair{pair}.nii.gz"] = tensor[..., i, :]
        for name, values in fields.items():
            self.engine.save(values, self.affine, self.out / name)
        labels = np.ones(self.shape, np.int16)
        labels[3:] = 50
        eroded = labels.copy()
        eroded[[0, 5]] = 0
        self.labels = self.root / "labels.nii.gz"
        self.eroded = self.root / "eroded.nii.gz"
        self.engine.save(labels, self.affine, self.labels)
        self.engine.save(eroded, self.affine, self.eroded)
        self.table = self.root / "labels.csv"
        with self.table.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "name"])
            writer.writerows((i, f"Region {i}") for i in range(1, 51))
        self.args = argparse.Namespace(labels=self.labels, eroded_labels=self.eroded,
                                       template_fa=None, labels_table=self.table)
        self.meta = dict(qref=.03, draws=100, radius_mm=6.875, pair_order=["12", "13", "23"],
                         subject="shared-region-fixture", outputs={}, labels_file=None)
        self.engine.write_summary(self.out, self.meta)

    def test_native_and_eroded_use_original_statistics_and_all_50_names(self):
        adapter.attach_shared_regions(self.out, self.args, adapter.label_names(self.table), self.meta)
        adapter.write_json(self.out / "run.json", self.meta)
        from verify_single_scan import verify
        self.assertEqual(verify(self.out)["status"], "passed")
        with (self.out / "summary.csv").open() as stream:
            native = list(csv.DictReader(stream))
        with (self.out / "summary_eroded.csv").open() as stream:
            eroded = list(csv.DictReader(stream))
        self.assertEqual(len(native), 153)
        self.assertEqual(len(eroded), 153)
        n = next(r for r in native if r["label"] == "1" and r["pair"] == "12")
        e = next(r for r in eroded if r["label"] == "1" and r["pair"] == "12")
        self.assertEqual(int(n["n_centers"]), 108)
        self.assertEqual(int(e["n_centers"]), 72)
        self.assertAlmostEqual(float(n["q_median"]), .01)
        self.assertAlmostEqual(float(e["q_median"]), .015)
        empty = next(r for r in native if r["label"] == "2")
        self.assertEqual(empty["n_centers"], "0")
        self.assertEqual(empty["coverage_fraction"], "")
        self.assertEqual(empty["q_median"], "")

    def test_shared_registered_labels_preserve_original_registration_metadata(self):
        self.args.template_fa = self.out / "anatomy/FA.nii.gz"
        adapter.attach_shared_regions(self.out, self.args, adapter.label_names(self.table), self.meta)
        self.assertEqual(self.meta["shared_registration"]["method"],
                         "Shared ANTs FMRIB58-to-diffusion registration supplied by coordinator")
        self.assertEqual(self.meta["shared_registration"]["atlas"], "JHU-ICBM labels / FMRIB58 FA")
        self.assertEqual(self.meta["jhu_registration"]["status"],
                         "Shared ANTs / FMRIB58 FA; JHU 50-label atlas; native labels; no Sheets registration rerun")
        self.assertEqual(adapter.sha(self.args.template_fa),
                         adapter.sha(self.out / "registration/template_in_native_FA.nii.gz"))

    def test_compact_qa_uses_shared_metadata_and_leaves_source_unchanged(self):
        self.meta['radius_mm'] = 5.25
        adapter.attach_shared_regions(self.out, self.args, adapter.label_names(self.table), self.meta)
        adapter.write_json(self.out / "run.json", self.meta)
        before = {str(p.relative_to(self.out)): adapter.sha(p)
                  for p in self.out.rglob("*") if p.is_file()}
        report = self.root / "new-qa/report.pdf"
        from sheets_qa import _Renderer
        original_finish = _Renderer.finish
        record = adapter.regenerate_qa(self.out, report, published_paths=True)
        after = {str(p.relative_to(self.out)): adapter.sha(p)
                 for p in self.out.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(record["page_count"], 8)
        self.assertEqual(record["shared_registration"]["named_regions"], 50)
        self.assertFalse(record["shared_registration"]["registration_rerun"])
        self.assertEqual(record["shared_registration"]["method"],
                         "Provided shared JHU labels already aligned to diffusion; no registration run")
        self.assertFalse((self.out / "registration/template_in_native_FA.nii.gz").exists())
        self.assertIs(_Renderer.finish, original_finish)
        self.assertEqual(record["guide_paths"], "published subject layout")
        self.assertGreater(report.stat().st_size, 1000)
        if importlib.util.find_spec('fitz'):
            import fitz
            with fitz.open(report) as document:
                first = document[0].get_text()
                self.assertIn('sheets/{q.nii.gz, wm_mask.nii.gz',first)
                self.assertIn('shared/anatomy/',first)
                self.assertIn('shared/regions/',first)
                self.assertIn('Statistics: sheets/summary.csv',first)
                self.assertIn('provenance/modules/sheets/',first)
                self.assertIn('5.25 mm', first)
                self.assertNotIn('Files: q + wm_mask',first)
                regional = document[-1].get_text()
                self.assertIn('sheets/summary.csv', regional)
                self.assertIn('Provided aligned JHU 50-region labels; no registration run', regional)
                self.assertNotIn('Shared ANTs', regional)
        with self.assertRaises(FileExistsError):
            adapter.regenerate_qa(self.out, report)

    def test_table_cannot_silently_drop_or_duplicate_ids(self):
        self.table.write_text("id,name\n1,one\n1,duplicate\n")
        with self.assertRaises(ValueError):
            adapter.label_names(self.table)

    def test_published_guide_does_not_advertise_absent_jhu_files(self):
        adapter.write_json(self.out / "run.json",self.meta)
        report=self.root/'no-jhu.pdf'
        record=adapter.regenerate_qa(self.out,report,published_paths=True)
        self.assertEqual(record['page_count'],7)
        if importlib.util.find_spec('fitz'):
            import fitz
            with fitz.open(report) as document:
                first=document[0].get_text()
                self.assertIn('JHU regions: not requested',first)
                self.assertNotIn('shared/regions/',first)


class MeasurementRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = {}
        for name, values in {'fod': np.ones((3, 3, 3, 45), np.float32),
                             'brain_mask': np.ones((3, 3, 3), np.uint8),
                             'wm_mask': np.ones((3, 3, 3), np.uint8),
                             'fa': np.full((3, 3, 3), .6, np.float32)}.items():
            path = self.root/(name+'.nii.gz')
            nib.save(nib.Nifti1Image(values, np.eye(4)), path)
            self.inputs[name] = path

    def args(self, name, **settings):
        return argparse.Namespace(**self.inputs, out=self.root/name, **settings)

    def interrupted_call(self, args):
        entry = Mock(side_effect=RuntimeError('controlled stop before measurement'))
        with patch.object(adapter, 'engine', return_value=SimpleNamespace(main=entry)):
            with self.assertRaisesRegex(RuntimeError, 'controlled stop'):
                adapter.run(args)
        return entry.call_args.args[0]

    def test_default_and_custom_settings_forward_to_frozen_cli_and_provenance(self):
        custom = dict(draws=80, radius_mm=5.25, qref=.05, seed=42, sample_batch_size=7)
        for name, supplied, expected in [('default', {}, adapter.DEFAULT_MEASUREMENT), ('custom', custom, custom)]:
            with self.subTest(name=name):
                args = self.args(name, **supplied)
                argv = self.interrupted_call(args)
                record = json.loads((args.out/'adapter.json').read_text())
                for key, value in expected.items():
                    flag = '--'+key.replace('_', '-')
                    self.assertEqual(float(argv[argv.index(flag)+1]), value)
                    self.assertEqual(record[key], value)
                self.assertIn('--native-spacing', argv)
                self.assertIn('--no-jhu', argv)
                self.assertNotIn('--dwi', argv)
        parsed = adapter.parser().parse_args([part for key, path in self.inputs.items()
                    for part in ('--'+key.replace('_', '-'), str(path))]+['--out', str(self.root/'parsed')])
        self.assertEqual({key: getattr(parsed, key) for key in adapter.DEFAULT_MEASUREMENT}, adapter.DEFAULT_MEASUREMENT)

    def test_each_changed_measurement_setting_refuses_resume_without_touching_state(self):
        changes = dict(draws=80, radius_mm=5.25, qref=.05, seed=42, sample_batch_size=7)
        for key, value in changes.items():
            with self.subTest(setting=key):
                args = self.args('resume-'+key)
                self.interrupted_call(args)
                state = (args.out/'adapter.json').read_bytes()
                resumed = self.args('resume-'+key, resume=True, **{key: value})
                with patch.object(adapter, 'engine') as entry:
                    with self.assertRaisesRegex(ValueError, 'resume inputs or settings differ'):
                        adapter.run(resumed)
                    entry.assert_not_called()
                self.assertEqual((args.out/'adapter.json').read_bytes(), state)

    def test_invalid_settings_and_radius_bound_fail_before_engine_or_input_hashing(self):
        invalid = [('radius_mm', x) for x in (0, -1, np.nan, np.inf, np.nextafter(16., np.inf))]
        invalid += [('draws', 0), ('draws', 1.5), ('seed', -1), ('sample_batch_size', 0)]
        invalid += [('qref', x) for x in (0, -1, np.nan, np.inf)]
        for index, (key, value) in enumerate(invalid):
            with self.subTest(setting=key, value=value):
                args = self.args('invalid-'+str(index), **{key: value})
                with patch.object(adapter, 'engine') as entry, patch.object(adapter, 'sha') as checksum:
                    with self.assertRaises(ValueError):
                        adapter.run(args)
                    entry.assert_not_called()
                    checksum.assert_not_called()
                self.assertFalse(args.out.exists())

    def test_radius_physical_bound_scales_with_the_supplied_grid(self):
        self.assertEqual(adapter.validate_radius(16, 1), 16)
        self.assertEqual(adapter.validate_radius(32, 2), 32)
        with self.assertRaisesRegex(ValueError, 'int16 neighborhood limit'):
            adapter.validate_radius(np.nextafter(32., np.inf), 2)
        argv = self.interrupted_call(self.args('largest-supported', radius_mm=16))
        self.assertEqual(argv[argv.index('--radius-mm')+1], '16.0')


if __name__ == "__main__":
    unittest.main()

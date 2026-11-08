"""Ownership, identity and retention checks that need no imaging runtime."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json
import sys
from argparse import Namespace

SPEC = importlib.util.spec_from_file_location('native_modules', Path(__file__).parents[1]/'native_modules.py')
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


class NativeAdapterTests(unittest.TestCase):
    def test_laminarity_scale_options_preserve_standalone_minimum_semantics(self):
        self.assertIsNone(native.resolve_laminarity_scales())
        for minimum, expected in [(2.5, [2.5, 3., 4., 6., 8.]),
                                  (3., [3., 4., 6., 8.]), (4., [4., 6., 8.]),
                                  (6., [6., 8.]), (8., [8.])]:
            self.assertEqual(native.resolve_laminarity_scales(minimum), expected)
        self.assertEqual(native.resolve_laminarity_scales(scales_mm=[6., 1.5, 3.]), [1.5, 3., 6.])
        self.assertEqual(native.resolve_laminarity_scales(scales_mm=[2.]), [2.])
        self.assertEqual(native.resolve_laminarity_scales(fixed_scale_mm=2.), [2.])

    def test_invalid_laminarity_scales_fail_before_engine_launch(self):
        for minimum in (0, 2., 9., float('nan'), float('inf')):
            with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                native.resolve_laminarity_scales(minimum)
        for values in ([], [0.], [-1.], [float('nan')], [float('inf')],
                       [3., 3.], [1., 1.0000001]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                native.resolve_laminarity_scales(scales_mm=values)
        with self.assertRaisesRegex(ValueError, 'not both'):
            native.resolve_laminarity_scales(4., [4., 8.])
        for fixed in (0., -1., float('nan'), float('inf')):
            with self.subTest(fixed=fixed), self.assertRaises(ValueError):
                native.resolve_laminarity_scales(fixed_scale_mm=fixed)
        for other in (dict(min_scale_mm=4.), dict(scales_mm=[4.])):
            with self.assertRaisesRegex(ValueError, 'not both'):
                native.resolve_laminarity_scales(fixed_scale_mm=4., **other)
        # Validate before requiring inputs or creating a stage workspace.
        with self.assertRaisesRegex(ValueError, 'finite positive'):
            native.run_stage(Namespace(stage='laminarity', min_scale_mm=None, scales_mm=[float('nan')]))

    def test_laminarity_config_routing_and_recorded_scale_contract(self):
        base = dict(method_version='transverse-lamellarity-v1', derivative_mm=.9375,
                    reference_mm=1.25, scales=[2.5, 3., 4., 6., 8.], workers=2,
                    min_fraction=.1, max_fixels=3, min_support=.8, no_wm_mask=True)
        cases = [(None, None, None, base['scales']), (4., None, None, [4., 6., 8.]),
                 (None, [6., 1.5, 3.], None, [1.5, 3., 6.]), (None, [2.], None, [2.]),
                 (None, None, 2., [2.])]
        for minimum, explicit, fixed, expected_scales in cases:
            with self.subTest(minimum=minimum, explicit=explicit), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); engine = root/'engine'; engine.mkdir()
                original = engine/'final_config.json'; native.write_json(original, base)
                original_bytes = original.read_bytes()
                out = root/'out'; (out/'provenance').mkdir(parents=True)
                work = root/'work'; work.mkdir(); commands = []
                args = Namespace(work=work, native=root/'native', out=out, threads=1, subject='test',
                                 min_scale_mm=minimum, scales_mm=explicit, save_all_scales=True,
                                 fixed_scale_mm=fixed, save_native_fixels=False, labels=None)

                def command(argv):
                    commands.append(argv)
                    if argv[1] == engine/'pipeline.py':
                        configuration = json.loads(Path(argv[argv.index('--config')+1]).read_text())
                        configuration['workers'] = int(argv[argv.index('--workers')+1])
                        analysis = work/'analysis'; analysis.mkdir()
                        for name, record in [('summary.json', {'options': configuration}),
                                             ('pipeline_manifest.json', {'configuration': configuration}),
                                             ('configuration.json', configuration)]:
                            native.write_json(analysis/name, record)

                native.laminarity(args, engine, 'python', command)
                config_path = commands[0][commands[0].index('--config')+1]
                expected = dict(base, scales=expected_scales, workers=1)
                self.assertEqual(json.loads((out/'provenance/analysis_configuration.json').read_text()), expected)
                self.assertIn('--save-all-scales', commands[1])
                self.assertEqual(original.read_bytes(), original_bytes)
                settings_path = out/'provenance/laminarity_scale_settings.json'
                if minimum is None and explicit is None and fixed is None:
                    self.assertEqual(config_path, original)
                    self.assertFalse(settings_path.exists())
                    self.assertIn('Independent L/T maxima over 2.5, 3, 4, 6, 8 mm;', (out/'README.txt').read_text())
                else:
                    self.assertEqual(config_path, out/'provenance/laminarity_configuration.json')
                    self.assertEqual(json.loads(config_path.read_text()), expected)
                    settings = json.loads(settings_path.read_text())
                    self.assertEqual(settings['base_configuration']['sha256'], native.sha256(original))
                    self.assertEqual(settings['resolved_configuration']['sha256'], native.sha256(config_path))
                    self.assertEqual(settings['resolved_scales_mm'], expected_scales)
                    self.assertEqual(settings['scale_mode'], 'fixed' if len(expected_scales) == 1 else 'max-selected')
                if len(expected_scales) == 1:
                    self.assertIn('fixed integration sigma 2 mm; no scale selection.', (out/'README.txt').read_text())
                else:
                    self.assertIn(', '.join(f'{value:g}' for value in expected_scales) + ' mm;', (out/'README.txt').read_text())

    def test_dispersion_public_bounds_and_early_rejection(self):
        self.assertEqual(native.validate_dispersion_options(), native.DISPERSION_DEFAULTS)
        edge = dict(fit_angle=89.9, min_separation=90., peak_threshold=1., max_lobes=10,
                    directions=100, match_angle=90., match_margin=0., axis_gap=0.)
        self.assertEqual(native.validate_dispersion_options(**edge), edge)
        invalid = dict(fit_angle=[0., 90.], min_separation=[0., 90.1], peak_threshold=[0., 1.1],
                       max_lobes=[0, 11, 1.5], directions=[99, 100.5], match_angle=[0., 90.1],
                       match_margin=[-1., 90.1], axis_gap=[-1., 1.])
        for name, cases in invalid.items():
            if name not in ('max_lobes', 'directions'): cases += [float('nan'), float('inf')]
            for value in cases:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    native.validate_dispersion_options(**{name: value})
        with self.assertRaisesRegex(ValueError, 'directions'):
            native.run_stage(Namespace(stage='dispersion', directions=99))

    def test_dispersion_default_command_and_all_custom_options(self):
        custom = dict(fit_angle=25., min_separation=40., peak_threshold=.2, max_lobes=4,
                      directions=724, match_angle=15., match_margin=7., axis_gap=.2)
        for options in (native.DISPERSION_DEFAULTS, custom):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); out = root/'out'; (out/'provenance').mkdir(parents=True)
                work = root/'work'; work.mkdir(); commands = []
                args = Namespace(fod=root/'fod', brain_mask=root/'brain', native=root/'native',
                                 out=out, work=work, threads=2, save_parameters=False,
                                 save_native_fixels=False, labels=None, **options)

                def command(argv):
                    commands.append(argv)
                    (out/'parameters').mkdir()
                    (out/'parameters/fixture.nii.gz').write_bytes(b'optional parameters')

                native.dispersion(args, root, 'python', command)
                expected = ['python', root/'compute.py', '--fod', args.fod, '--brain-mask', args.brain_mask,
                            '--native-fixels', args.native/'fixels', '--out', out, '--threads', 2,
                            '--save-parameters']
                if options == custom:
                    for name, value in custom.items(): expected += ['--' + name.replace('_', '-'), value]
                    self.assertEqual(json.loads((out/'provenance/dispersion_settings.json').read_text()), custom)
                else:
                    self.assertFalse((out/'provenance/dispersion_settings.json').exists())
                self.assertEqual(commands, [expected])

    def test_both_native_modules_accept_shared_labels_without_warped_template(self):
        for module in ('laminarity', 'dispersion'):
            with self.subTest(module=module), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);engine=root/'engine';engine.mkdir()
                native.write_json(engine/'final_config.json',dict(scales=[2.5,3.,4.,6.,8.]))
                out=root/'out';(out/'provenance').mkdir(parents=True);work=root/'work';work.mkdir()
                args=Namespace(fod=root/'fod',brain_mask=root/'brain',fa=root/'fa',native=root/'native',
                    labels=root/'labels',template_fa=None,regional_wm_mask=None,subject='labels-only',
                    out=out,work=work,threads=2,save_parameters=False,save_all_scales=False,save_native_fixels=False)
                commands=[]
                def command(argv):
                    self.assertNotIn(None,argv);commands.append(argv)
                    if argv[1]==engine/'pipeline.py':
                        analysis=work/'analysis';analysis.mkdir()
                        for name in ('summary.json','pipeline_manifest.json','configuration.json'):
                            native.write_json(analysis/name,{})
                    elif argv[1]==engine/'compute.py':
                        (out/'parameters').mkdir();(out/'parameters/fixture').write_bytes(b'parameters')
                getattr(native,module)(args,engine,'python',command)
                if module=='laminarity':
                    regional=next(argv for argv in commands if argv[1]==engine/'regions.py')
                    self.assertIn('--native-labels',regional);self.assertNotIn('--template-in-native',regional)
                else:
                    self.assertTrue(any('from regions import run' in str(argv) for argv in commands))
                self.assertFalse((out/'regions/JHU_template_FA_native.nii.gz').exists())

    def test_float32_selection_boundary_keeps_module_maps_separate(self):
        try:
            import nibabel as nib
            import numpy as np
        except ImportError:
            self.skipTest('Imaging runtime is needed for publication regression')
        if not np.__version__.startswith('1.26.'):
            self.skipTest('This precision regression targets the frozen NumPy 1.26 runtime')
        sys.path.insert(0,str(Path(__file__).parents[1]))
        import publish
        spec=importlib.util.spec_from_file_location('boundary_driver',Path(__file__).parents[1]/'main.py')
        driver=importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
        mass=np.array([.1,np.nextafter(np.float32(.9),np.float32(np.inf))],np.float32)
        total=float(np.sum(mass,dtype=np.float64))
        l_keep=mass/np.array([total,total],np.float64)>=.1
        d_keep=mass/total>=.1
        self.assertEqual(l_keep.tolist(),[False,True]); self.assertEqual(d_keep.tolist(),[True,True])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); work=root/'work'; prepared=work/'prepared'; reports=work/'reports'; reports.mkdir(parents=True)
            def save(path,values):
                path.parent.mkdir(parents=True,exist_ok=True); nib.save(nib.Nifti1Image(values,np.eye(4)),path)
            save(prepared/'anatomy/brain_mask.nii.gz',np.ones((1,1,1),np.uint8))
            save(prepared/'anatomy/FA.nii.gz',np.ones((1,1,1),np.float32))
            (prepared/'reconstruction').mkdir(); (prepared/'reconstruction/reconstruction.json').write_text('{}')
            modules={name:work/'modules'/name for name in ('laminarity','dispersion')}
            record=dict(status='complete',files={'index':{'sha256':'shared'}},
                        inputs={'fod':{'sha256':'same-fod'},'brain_mask':{'sha256':'same-mask'}})
            for name,selection in [('laminarity',[1]),('dispersion',[1,0])]:
                folder=modules[name]; ids=np.full(3,-1,np.int32); afd=np.full(3,np.nan,np.float32); axes=np.full((3,3),np.nan,np.float32)
                ids[:len(selection)]=selection; afd[:len(selection)]=mass[selection]; axes[:len(selection)]=np.array([[1,0,0],[0,1,0]],np.float32)[:len(selection)]
                for key,values in [('native_fixel_id',ids),('afd',afd),('fiber_direction',axes)]:
                    save(folder/'fixel_maps'/(key+'.nii.gz'),values.reshape((1,1,1)+values.shape))
                (folder/'provenance').mkdir(); (folder/'provenance/shared_native.json').write_text(json.dumps(record))
            checks=driver.check_outputs(prepared,modules,None)
            self.assertFalse(checks['native_identity']['shared_reported_maps'])
            self.assertEqual(checks['native_identity']['dispersion_only_fixels'],1)
            destination=root/'published'
            layout=publish.build(Namespace(save_fod=False,save_native_fixels=False,save_registration=False),work,destination,modules,reports)
            self.assertFalse((destination/'shared/fixels').exists())
            for name in modules:
                self.assertEqual(layout[name]['fixel_maps/native_fixel_id.nii.gz'],name+'/fixel_maps/native_fixel_id.nii.gz')
                self.assertTrue(publish.equal_image(modules[name]/'fixel_maps/native_fixel_id.nii.gz',destination/name/'fixel_maps/native_fixel_id.nii.gz'))

    def test_changed_shared_fod_or_extraction_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); prepared = root/'native'; (prepared/'fixels').mkdir(parents=True)
            fod = root/'fod'; brain = root/'brain'; fod.write_bytes(b'FOD'); brain.write_bytes(b'brain')
            names = ('fixels/index.nii', 'fixels/directions.nii', 'fixels/mass.nii',
                     'fixels/peak_amplitude.nii', 'brain_mask.nii.gz', 'total_mass.nii.gz', 'summary.json')
            for name in names: (prepared/name).write_bytes(name.encode())
            native.write_json(prepared/'integration_extract.json', dict(status='complete',
                inputs=dict(fod=native.file_record(fod), brain_mask=native.file_record(brain)),
                files=native.native_records(prepared)))
            native.validate_native(prepared, fod, brain)
            fod.write_bytes(b'changed FOD')
            with self.assertRaisesRegex(ValueError, 'fod differs'): native.validate_native(prepared, fod, brain)
            fod.write_bytes(b'FOD'); (prepared/'fixels/mass.nii').write_bytes(b'changed AFD')
            with self.assertRaisesRegex(ValueError, 'files have changed'): native.validate_native(prepared, fod, brain)

    def test_parameter_size_boundary_and_explicit_retention(self):
        for size, explicit, expected in [(9_999_999, False, True), (10_000_000, False, True),
                                         (10_000_001, False, False), (10_000_001, True, True)]:
            with self.subTest(size=size, explicit=explicit), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); out = root/'out'; work = root/'work'; work.mkdir()
                (out/'parameters').mkdir(parents=True); (out/'provenance').mkdir()
                payload = out/'parameters/raw.nii.gz'
                with payload.open('wb') as stream: stream.truncate(size)
                record = native.retain_parameters(out, work, explicit)
                self.assertEqual(record['compressed_bytes'], size)
                self.assertEqual(record['retained'], expected)
                self.assertEqual(payload.exists(), expected)
                self.assertEqual((work/'optional_parameters/raw.nii.gz').exists(), not expected)

    def test_laminarity_does_not_inherit_dispersion_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); overlay = root/'deps'; overlay.mkdir()
            for module in ('laminarity', 'dispersion'):
                work = root/module; work.mkdir()
                with patch.dict(os.environ, {'PYTHONPATH': '/untrusted/inherited',
                                             'GEO_DISPERSION_DEPS': str(overlay)}):
                    env = native.environment(module, work, 2)
                self.assertNotIn('/untrusted/inherited', env['PYTHONPATH'])
                self.assertEqual(str(overlay) in env['PYTHONPATH'], module == 'dispersion')

    def test_existing_destination_is_never_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)/'completed'; destination.mkdir()
            evidence = destination/'result'; evidence.write_text('keep')
            with self.assertRaises(FileExistsError): native.require_fresh(destination)
            self.assertEqual(evidence.read_text(), 'keep')


if __name__ == '__main__': unittest.main()

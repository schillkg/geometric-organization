"""QA refusal paths must leave inputs and existing reports untouched."""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
import ast
from unittest.mock import patch

SPEC=importlib.util.spec_from_file_location('geo_qa',Path(__file__).parents[1]/'qa.py')
qa=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(qa)


class QAAdapterTests(unittest.TestCase):
    def test_display_paths_follow_compact_publication(self):
        publisher=types.SimpleNamespace(IDENTITY={'afd.nii.gz','native_fixel_id.nii.gz','fiber_direction.nii.gz'},
            REGIONS={'JHU_labels_native.nii.gz','JHU_labels_eroded_1voxel.nii.gz','JHU_template_FA_native.nii.gz'})
        cases={'anatomy/FA.nii.gz':'shared/anatomy/FA.nii.gz',
               'fixel_maps/afd.nii.gz':'shared/fixels/afd.nii.gz',
               'regions/JHU_labels_native.nii.gz':'shared/regions/JHU_labels_native.nii.gz',
               'regions/JHU_region_means.csv':'laminarity/regions/JHU_region_means.csv',
               'scales/lamellarity_all_scales.nii.gz':'extras/laminarity/scales/lamellarity_all_scales.nii.gz',
               'manifest.json':'provenance/modules/laminarity/manifest.json',
               'QA/Lamellarity_QA.pdf':'QA/02_lamellarity.pdf'}
        with patch.dict('sys.modules',{'publish':publisher}):
            for original,final in cases.items(): self.assertEqual(qa.guide_path('laminarity',original,True),final)
            self.assertEqual(qa.guide_path('laminarity','fixel_maps/afd.nii.gz',False),'laminarity/fixel_maps/afd.nii.gz')

    def test_dispersion_display_hooks_match_frozen_guide_exactly(self):
        root=qa.engine_source('dispersion')
        if not (root/'qa_report.py').is_file(): root=Path(__file__).parents[2]/'Laminarity/dispersion/pipeline'
        if not (root/'qa_report.py').is_file(): self.skipTest('Frozen Dispersion report source is unavailable')
        tree=ast.parse((root/'qa_report.py').read_text())
        build=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build')
        assignment=next(n for n in ast.walk(build) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='guide' for t in n.targets))
        cells=[cell for row in ast.literal_eval(assignment.value) for cell in row]
        with tempfile.TemporaryDirectory() as directory:
            stage=Path(directory); qa.dispersion_path_wrapper(stage,root,True,True)
            replacements=json.loads((stage/'published_guide.json').read_text())
            self.assertEqual(len(replacements),7)
            for old in replacements: self.assertEqual(cells.count(old),1)
            compile((stage/'render_dispersion.py').read_text(),'render_dispersion.py','exec')

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); out=root/'existing'; out.mkdir(); (out/'report.pdf').write_bytes(b'keep')
            with self.assertRaises(FileExistsError):
                qa.build_reports({'laminarity':root/'module'},root/'prepared',None,out,'sub',{},None)
            self.assertEqual((out/'report.pdf').read_bytes(),b'keep')

    def test_output_cannot_be_created_in_module_input(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'module'; source.mkdir(); out=source/'new-qa'
            with self.assertRaisesRegex(ValueError,'separate'):
                qa.build_reports({'dispersion':source},root/'prepared',None,out,'sub',{},None)
            self.assertFalse(out.exists())

    def test_regeneration_checks_hashes_before_making_work(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); subject=root/'subject'; subject.mkdir(); out=root/'new-qa'
            (subject/'run.json').write_text(json.dumps(dict(status='complete',public_files={'map.nii.gz':{}})))
            def reject(*args): raise ValueError('Changed saved map')
            with patch.dict('sys.modules',{'publish':types.SimpleNamespace(verify=reject)}):
                with self.assertRaisesRegex(ValueError,'Changed saved map'): qa.regenerate(subject,out)
            self.assertFalse(out.exists()); self.assertFalse(list(root.glob('.qa-regeneration-*')))

    def test_regeneration_refuses_subject_subdirectory(self):
        with tempfile.TemporaryDirectory() as d:
            subject=Path(d)/'subject'; subject.mkdir(); out=subject/'new-qa'
            with patch.dict('sys.modules',{'publish':types.SimpleNamespace()}):
                with self.assertRaisesRegex(ValueError,'outside'): qa.regenerate(subject,out)
            self.assertFalse(out.exists())


if __name__=='__main__': unittest.main()

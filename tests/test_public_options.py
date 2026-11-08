"""Public CLI rejects ambiguous or inapplicable settings before creating output."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('public_driver', ROOT/'main.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class PublicOptionsTests(unittest.TestCase):
    def test_module_options_fail_before_input_access_or_output_creation(self):
        cases = [
            (['--modules','laminarity','--sheets-radius-mm','5'], 'Sheets'),
            (['--modules','dispersion','--laminarity-fixed-scale-mm','4'], 'Laminarity'),
            (['--modules','laminarity','--dispersion-fit-angle','25'], 'Dispersion'),
            (['--modules','laminarity','--laminarity-scales-mm','2','2'], 'unique'),
            (['--modules','dispersion','--dispersion-fit-angle','nan'], 'fit'),
            (['--modules','dispersion','--dispersion-directions','99'], 'directions'),
            (['--modules','laminarity','--spacing','1','--native-spacing'], 'not allowed'),
        ]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'untouched'
            base = [sys.executable,str(ROOT/'main.py'),'--fod','absent.nii.gz',
                    '--brain-mask','absent-mask.nii.gz','--fa','absent-fa.nii.gz',
                    '--out',str(out)]
            for options, expected in cases:
                with self.subTest(options=options):
                    result = subprocess.run(base+options,text=True,capture_output=True)
                    self.assertEqual(result.returncode,2,result.stderr)
                    self.assertIn(expected.lower(),result.stderr.lower())
                    self.assertFalse(out.exists())

    def test_explicit_scale_order_is_canonical_for_resume(self):
        def parse(scales):
            args=driver.parser().parse_args(['--fod','f.nii.gz','--out','out',
                '--modules','laminarity','--laminarity-scales-mm',*scales])
            driver.validate_measurement_options(args)
            return args.laminarity_scales_mm
        self.assertEqual(parse(['8','2.5','4']),parse(['2.5','4','8']))


if __name__ == '__main__':
    unittest.main()

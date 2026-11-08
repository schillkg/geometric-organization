import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from portable_qa import Runner


class PortableQACleanupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='geo-qa-owner-')
        root = Path(self.temporary.name)
        self.runner = Runner.__new__(Runner)
        self.runner.out = root/'evidence'; self.runner.out.mkdir()
        self.runner.scratch = root/'scratch'; self.runner.scratch.mkdir()
        self.runner.owner = root/'scratch.owner.json'
        self.runner.run_id = 'test-run'
        self.runner.record = {}
        self.marker = dict(run_id='test-run', scratch=str(self.runner.scratch), evidence=str(self.runner.out))
        self.runner.owner.write_text(json.dumps(self.marker))
        (self.runner.out/'keep.txt').write_text('keep')
        (self.runner.scratch/'remove.txt').write_text('temporary')

    def tearDown(self):
        self.temporary.cleanup()

    def test_success_removes_only_owned_scratch_and_external_marker(self):
        self.runner.cleanup_owned_scratch()
        self.assertFalse(self.runner.scratch.exists())
        self.assertFalse(self.runner.owner.exists())
        self.assertEqual((self.runner.out/'keep.txt').read_text(), 'keep')

    def test_mismatched_marker_refuses_cleanup(self):
        self.runner.owner.write_text(json.dumps(dict(self.marker, run_id='someone-else')))
        with self.assertRaisesRegex(ValueError, 'ownership'):
            self.runner.cleanup_owned_scratch()
        self.assertTrue((self.runner.scratch/'remove.txt').exists())
        self.assertTrue(self.runner.owner.exists())

    def test_failed_cleanup_keeps_external_ownership_marker(self):
        with patch('portable_qa.shutil.rmtree', side_effect=OSError('simulated partial cleanup')):
            with self.assertRaises(OSError):
                self.runner.cleanup_owned_scratch()
        self.assertTrue(self.runner.owner.exists())
        self.assertNotEqual(self.runner.record.get('scratch_cleanup'), 'complete')


if __name__ == '__main__':
    unittest.main()

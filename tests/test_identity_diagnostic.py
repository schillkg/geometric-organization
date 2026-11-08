import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from diagnose_identity import native_crosswalk, paired


class IdentityDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='geo-identity-')
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def save(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(data, np.eye(4)), path)

    def test_complete_sparse_permutation_and_one_ulp_rejection(self):
        mass = np.array([.7, .3, 1], np.float32)
        axes = np.eye(3, dtype=np.float32)
        for name, offsets, order in [('a', [0, 2], [0, 1, 2]), ('b', [1, 0], [2, 0, 1])]:
            index = np.stack(([2, 1], offsets), axis=-1).astype(np.int32).reshape(2, 1, 1, 2)
            self.save(self.root/name/'index.nii', index)
            self.save(self.root/name/'mass.nii', mass[order, None, None])
            self.save(self.root/name/'peak_amplitude.nii', (mass*2)[order, None, None])
            self.save(self.root/name/'directions.nii', axes[order, None, None, :])
        output = self.root/'proof'; output.mkdir()
        result = native_crosswalk(self.root/'a', self.root/'b', output)
        self.assertTrue(result['all_native_lobes_exact_under_row_permutation'])
        self.assertEqual(result['changed_raw_ids'], 3)
        changed = mass[[2, 0, 1]].copy()
        changed[0] = np.nextafter(changed[0], np.float32(2))
        self.save(self.root/'b/mass.nii', changed[:, None, None])
        bad_output = self.root/'bad'; bad_output.mkdir()
        result = native_crosswalk(self.root/'a', self.root/'b', bad_output)
        self.assertFalse(result['all_native_lobes_exact_under_row_permutation'])

    def test_reported_bijection_allows_different_numeric_sets_but_not_direction_change(self):
        for folder, values in [('a', [0, 2]), ('b', [3, 1])]:
            ids = np.full((2, 1, 1, 3), -1, np.int32); ids[:, 0, 0, 0] = values
            afd = np.full(ids.shape, np.nan, np.float32); afd[:, 0, 0, 0] = 1
            axes = np.full(ids.shape+(3,), np.nan, np.float32); axes[:, 0, 0, 0] = [1, 0, 0]
            for name, data in [('native_fixel_id', ids), ('afd', afd), ('fiber_direction', axes)]:
                self.save(self.root/folder/'fixel_maps'/(name+'.nii.gz'), data)
        output = self.root/'proof'; output.mkdir()
        result = paired(self.root, 'a', 'b', 'laminarity', output)
        self.assertTrue(result['permutation_only'])
        self.assertTrue(result['complete_bijective_reported_crosswalk'])
        self.assertFalse(result['left_reported_ids_contiguous_zero_based'])
        axes[0, 0, 0, 0] = [0, 1, 0]
        self.save(self.root/'b/fixel_maps/fiber_direction.nii.gz', axes)
        bad_output = self.root/'bad'; bad_output.mkdir()
        self.assertFalse(paired(self.root, 'a', 'b', 'laminarity', bad_output)['permutation_only'])


if __name__ == '__main__':
    unittest.main()

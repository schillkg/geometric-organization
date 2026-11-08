import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from join_reported_identity import digest, join_reported_ids


class ReportedIdentityJoinTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='geo-id-join-')
        self.root = Path(self.temporary.name)
        self.integrated, self.baseline = self.root/'integrated', self.root/'baseline'
        self.crosswalk = self.root/'all_native_rows.npz'
        self.proof = self.root/'native_identity_diagnostic.json'
        self.mapping = dict(left_native_id=np.arange(4), right_native_id=np.array([2, 3, 0, 1]),
                            flat_C_voxel=np.array([0, 0, 1, 1]), local_native_slot=np.array([0, 1, 0, 1]))
        self.left = np.array([1, 0, -1, 2, -1, -1], np.int32).reshape(2, 1, 1, 3)
        self.right = np.array([3, 2, -1, 0, -1, -1], np.int32).reshape(2, 1, 1, 3)
        hashes = {name: name for name in ('index.nii', 'mass.nii', 'directions.nii', 'peak_amplitude.nii')}
        self.record = dict(all_native_lobes_exact_under_row_permutation=True, native_rows=4,
                           source_hashes=[hashes, hashes])
        path = self.integrated/'provenance/layout.json'; path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'laminarity': {'fixel_maps/native_fixel_id.nii.gz': 'shared/fixels/native_fixel_id.nii.gz'}}))
        path = self.integrated/'provenance/modules/laminarity/provenance/shared_native.json'; path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'status': 'complete', 'files': {'fixels/'+name: {'sha256': value} for name, value in hashes.items()}}))
        self.save(self.integrated/'shared/fixels/native_fixel_id.nii.gz', self.left)
        self.save(self.baseline/'fixel_maps/native_fixel_id.nii.gz', self.right)
        self.save_proof()

    def tearDown(self):
        self.temporary.cleanup()

    def save(self, path, values, affine=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(values, np.eye(4) if affine is None else affine), path)

    def save_proof(self):
        np.savez_compressed(self.crosswalk, **self.mapping)
        self.record['crosswalk'] = dict(sha256=digest(self.crosswalk))
        self.proof.write_text(json.dumps(self.record))

    def join(self):
        return join_reported_ids(self.crosswalk, self.integrated, self.baseline, 'laminarity')

    def test_exact_native_join_preserves_ranked_slots_and_absence(self):
        result = self.join()
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['reported_fixels'], 3)
        self.assertEqual(result['raw_numeric_id_differences'], 3)
        self.assertTrue(result['inputs_unchanged'])

    def test_wrong_reported_slot_fails_even_with_same_id_set(self):
        changed = self.right.copy(); changed[0, 0, 0, :2] = changed[0, 0, 0, :2][::-1]
        self.save(self.baseline/'fixel_maps/native_fixel_id.nii.gz', changed)
        result = self.join()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['mismatched_mapped_ids'], 2)

    def test_duplicate_crosswalk_row_rejected_even_with_matching_hash(self):
        self.mapping['right_native_id'][0] = self.mapping['right_native_id'][1]
        self.save_proof()
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            self.join()

    def test_changed_missingness_rejected(self):
        changed = self.right.copy(); changed[0, 0, 0, 0] = -1
        self.save(self.baseline/'fixel_maps/native_fixel_id.nii.gz', changed)
        with self.assertRaisesRegex(ValueError, 'missingness'):
            self.join()

    def test_affine_and_extraction_provenance_are_exact(self):
        affine = np.eye(4); affine[0, 3] = .00001
        self.save(self.baseline/'fixel_maps/native_fixel_id.nii.gz', self.right, affine)
        with self.assertRaisesRegex(ValueError, 'affine'):
            self.join()
        self.save(self.baseline/'fixel_maps/native_fixel_id.nii.gz', self.right)
        self.record['source_hashes'][0] = dict(self.record['source_hashes'][0], **{'index.nii': 'different'})
        self.save_proof()
        with self.assertRaisesRegex(ValueError, 'not this integrated extraction'):
            self.join()


if __name__ == '__main__':
    unittest.main()

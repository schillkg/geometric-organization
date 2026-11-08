#!/usr/bin/env python3
"""Join compact reported IDs through a separately proved all-native crosswalk.

Read-only audit helper. No IDs or measurement outputs are rewritten. A passed
all-native proof, its exact crosswalk hash, and the integrated extraction hashes
must agree before comparing reported identity at each voxel and slot.
"""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def inside(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    path = (root/relative).resolve()
    if relative.is_absolute() or root not in path.parents:
        raise ValueError('Published layout path escapes its result root')
    return path


def join_reported_ids(crosswalk, integrated, baseline, module, proof=None):
    crosswalk, integrated, baseline = map(Path, (crosswalk, integrated, baseline))
    proof = Path(proof) if proof else crosswalk.with_name('native_identity_diagnostic.json')
    if module not in ('laminarity', 'dispersion'):
        raise ValueError('Choose laminarity or dispersion')
    inputs = {}

    def consume(path):
        path = Path(path)
        inputs[str(path)] = dict(sha256=digest(path), bytes=path.stat().st_size)
        return path

    native_proof = json.loads(consume(proof).read_text())
    if native_proof.get('all_native_lobes_exact_under_row_permutation') is not True:
        raise ValueError('The all-native permutation proof did not pass')
    consume(crosswalk)
    if native_proof.get('crosswalk', {}).get('sha256') != inputs[str(crosswalk)]['sha256']:
        raise ValueError('Crosswalk hash differs from the successful all-native proof')
    with np.load(crosswalk, allow_pickle=False) as saved:
        names = ('left_native_id', 'right_native_id', 'flat_C_voxel', 'local_native_slot')
        rows = {name: saved[name] for name in names}
    n = len(rows['left_native_id'])
    if n == 0 or native_proof.get('native_rows') != n:
        raise ValueError('Crosswalk row count differs from the native proof')
    for name, values in rows.items():
        if values.ndim != 1 or len(values) != n or not np.issubdtype(values.dtype, np.integer) or np.any(values < 0):
            raise ValueError('Invalid crosswalk field: '+name)
    for name in ('left_native_id', 'right_native_id'):
        if not np.array_equal(np.sort(rows[name]), np.arange(n)):
            raise ValueError('Crosswalk must map every native row exactly once: '+name)
    order = np.lexsort((rows['local_native_slot'], rows['flat_C_voxel']))
    voxels, locals_ = rows['flat_C_voxel'][order], rows['local_native_slot'][order]
    starts = np.r_[0, np.flatnonzero(np.diff(voxels)) + 1]
    lengths = np.diff(np.r_[starts, n])
    expected = np.arange(n)-np.repeat(starts, lengths)
    if not np.array_equal(locals_, expected):
        raise ValueError('Duplicate or noncontiguous voxel/local-native-slot crosswalk entries')

    layout_path = integrated/'provenance/layout.json'
    layout = json.loads(consume(layout_path).read_text())
    left_path = inside(integrated, layout[module]['fixel_maps/native_fixel_id.nii.gz'])
    right_path = baseline/'fixel_maps/native_fixel_id.nii.gz'
    native_record_path = integrated/'provenance/modules'/module/'provenance/shared_native.json'
    native_record = json.loads(consume(native_record_path).read_text())
    expected_hashes = native_proof.get('source_hashes', [None])[0]
    if native_record.get('status') != 'complete' or not expected_hashes:
        raise ValueError('Missing completed integrated extraction provenance')
    for name in ('index.nii', 'mass.nii', 'directions.nii', 'peak_amplitude.nii'):
        if native_record.get('files', {}).get('fixels/'+name, {}).get('sha256') != expected_hashes.get(name):
            raise ValueError('Native proof left side is not this integrated extraction: '+name)
    images = [nib.load(consume(path)) for path in (left_path, right_path)]
    arrays = [np.asarray(image.dataobj) for image in images]
    a, b = arrays
    if a.ndim != 4 or a.shape[-1] != 3 or a.shape != b.shape:
        raise ValueError('Reported ID images must have identical X,Y,Z,3 shapes')
    if not np.array_equal(images[0].affine, images[1].affine):
        raise ValueError('Reported ID affine differs')
    if a.dtype != b.dtype or not np.issubdtype(a.dtype, np.integer):
        raise ValueError('Reported ID dtypes differ or are not integer')
    if any(np.any((values < -1) | (values >= n)) for values in arrays):
        raise ValueError('Reported ID is outside -1 or valid native-row bounds')
    present = a >= 0
    if not np.array_equal(present, b >= 0):
        raise ValueError('Reported ID missingness differs')
    for values in arrays:
        selected = values[values >= 0]
        if len(np.unique(selected)) != len(selected):
            raise ValueError('Duplicate reported native ID')
    total_voxels = int(np.prod(a.shape[:3]))
    if np.any(rows['flat_C_voxel'] >= total_voxels):
        raise ValueError('Crosswalk voxel lies outside the reported image grid')
    lookup = np.empty(n, np.int64)
    lookup[rows['left_native_id']] = rows['right_native_id']
    left_voxel = np.empty(n, np.int64)
    left_voxel[rows['left_native_id']] = rows['flat_C_voxel']
    exported_voxel = np.flatnonzero(present.ravel())//3
    if not np.array_equal(left_voxel[a[present]], exported_voxel):
        raise ValueError('Integrated reported ID disagrees with its all-native voxel')
    mapped = lookup[a[present]]
    exact = np.array_equal(mapped, b[present])
    unchanged = all(Path(path).stat().st_size == info['bytes'] and digest(path) == info['sha256']
                    for path, info in inputs.items())
    if not unchanged:
        raise RuntimeError('An audit input changed during the join')
    return dict(status='passed' if exact else 'failed', module=module,
        policy='Exact reported identity after verified extraction-scoped row crosswalk; no tolerance or ID rewriting',
        native_rows=n, reported_fixels=int(present.sum()), shape=list(a.shape), dtype=str(a.dtype),
        affine=images[0].affine.tolist(), missingness_exact=True, shape_affine_dtype_exact=True,
        native_proof_bound_to_integrated_extraction=True, all_reported_ids_in_native_bounds=True,
        reported_ids_unique=True, reported_voxels_match_native=True,
        mapped_reported_ids_exact=bool(exact), mismatched_mapped_ids=int(np.count_nonzero(mapped != b[present])),
        raw_numeric_id_differences=int(np.count_nonzero(a[present] != b[present])),
        input_hashes=inputs, inputs_unchanged=unchanged)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crosswalk', type=Path, required=True)
    parser.add_argument('--proof', type=Path)
    parser.add_argument('--integrated', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--module', choices=('laminarity', 'dispersion'), required=True)
    parser.add_argument('--out', type=Path, required=True, help='Fresh output JSON path outside both input roots')
    args = parser.parse_args()
    destination = args.out.resolve()
    if destination == Path('/nfs2') or Path('/nfs2') in destination.parents:
        parser.error('/nfs2 is read-only')
    if destination.exists() or any(root.resolve() == destination or root.resolve() in destination.parents
                                   for root in (args.integrated, args.baseline)):
        parser.error('Choose a fresh output file outside the read-only input folders')
    try:
        result = join_reported_ids(args.crosswalk, args.integrated, args.baseline, args.module, args.proof)
    except (ValueError, KeyError, IndexError, RuntimeError) as error:
        result = dict(status='failed', error=type(error).__name__+': '+str(error))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Read-only diagnosis of extraction-scoped sparse row permutations.

This never rewrites an acceptance result or relaxes any comparison tolerance.
It saves complete reported-ID crosswalks and independently checks retained
native offsets, mass and directions against each module's exported identity.
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
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    image = nib.load(path)
    return image, np.asarray(image.dataobj)


def published(suite, folder, module, name):
    root = suite / folder
    layout = root / 'provenance/layout.json'
    relative = 'fixel_maps/' + name + '.nii.gz'
    return root / (json.loads(layout.read_text())[module][relative] if layout.exists() else relative)


def identities(suite, folder, module):
    return {name: read(published(suite, folder, module, name))[1]
            for name in ('native_fixel_id', 'afd', 'fiber_direction')}


def paired(suite, left, right, module, out):
    a, b = identities(suite, left, module), identities(suite, right, module)
    x, y = a['native_fixel_id'], b['native_fixel_id']
    present = x >= 0
    shape = x.shape == y.shape
    missing = shape and np.array_equal(present, y >= 0)
    record = dict(left=left, right=right, module=module, shape_equal=shape,
                  present_mask_exact=missing,
                  all_identity_file_hashes={side: {name: digest(published(suite, folder, module, name))
                      for name in a} for side, folder in [('left', left), ('right', right)]})
    if not missing:
        record['permutation_only'] = False
        return record
    u, v = x[present], y[present]
    valid = np.flatnonzero(present.ravel())
    row = np.column_stack((u, v, valid // x.shape[-1], valid % x.shape[-1])).astype(np.int64)
    crosswalk = out / f'{left}-to-{right}-{module}.npz'
    np.savez_compressed(crosswalk, columns=np.array(['left_native_id', 'right_native_id', 'flat_C_voxel', 'reported_slot']), rows=row)
    unique_a, unique_b = len(np.unique(u)) == len(u), len(np.unique(v)) == len(v)
    afd_exact = np.array_equal(a['afd'], b['afd'], equal_nan=True)
    direction_exact = np.array_equal(a['fiber_direction'], b['fiber_direction'], equal_nan=True)
    ties = present[..., :-1] & present[..., 1:] & (a['afd'][..., :-1] == a['afd'][..., 1:])
    record.update(reported_fixels=len(u), left_ids_unique=unique_a, right_ids_unique=unique_b,
                  complete_bijective_reported_crosswalk=unique_a and unique_b,
                  left_reported_ids_contiguous_zero_based=np.array_equal(np.sort(u), np.arange(len(u))),
                  right_reported_ids_contiguous_zero_based=np.array_equal(np.sort(v), np.arange(len(v))),
                  changed_raw_ids=int(np.count_nonzero(u != v)),
                  voxel_slot_afd_exact=afd_exact, voxel_slot_direction_exact=direction_exact,
                  tied_adjacent_afd_pairs=int(ties.sum()),
                  tied_afd_slots_and_directions_unchanged=afd_exact and direction_exact,
                  crosswalk=dict(path=crosswalk.name, sha256=digest(crosswalk), rows=len(row)),
                  permutation_only=unique_a and unique_b and afd_exact and direction_exact)
    manifests = {}
    for side, folder in [('left', left), ('right', right)]:
        path = suite / folder / 'provenance/modules' / module / 'provenance/shared_native.json'
        if path.exists():
            source = json.loads(path.read_text())
            manifests[side] = dict(status=source.get('status'),
                inputs={key: value['sha256'] for key, value in source['inputs'].items()},
                extraction_files=source['files'], provenance_sha256=digest(path))
    record['extraction_provenance'] = manifests
    if len(manifests) == 2:
        record['same_fod_and_brain_hashes'] = manifests['left']['inputs'] == manifests['right']['inputs']
        record['same_sparse_file_hashes'] = manifests['left']['extraction_files'] == manifests['right']['extraction_files']
    return record


def retained_native(suite, module):
    native = suite / 'retained/work/native'
    image, index = read(native / 'fixels/index.nii')
    counts, offsets = index[..., 0].astype(np.int64).ravel(), index[..., 1].astype(np.int64).ravel()
    _, mass = read(native / 'fixels/mass.nii')
    _, axes = read(native / 'fixels/directions.nii')
    mass, axes = mass.ravel(), axes.reshape(-1, 3)
    occupied = np.flatnonzero(counts > 0)
    order = occupied[np.argsort(offsets[occupied])]
    intervals = np.array_equal(offsets[order], np.r_[0, np.cumsum(counts[order])[:-1]])
    d = identities(suite, 'retained', module)
    ids = d['native_fixel_id'].reshape(-1, 3)
    present = ids >= 0
    voxel, slot = np.where(present)
    rows = ids[present]
    valid_rows = bool(np.all((rows >= 0) & (rows < len(mass))))
    # These are the frozen engines' explicit display direction conventions:
    # Laminarity normalizes the stored float32 vector, Dispersion copies it.
    expected_axes = axes / np.linalg.norm(axes, axis=1)[:, None] if module == 'laminarity' else axes
    results = dict(module=module, native_rows=len(mass), reported_rows=len(rows),
        contiguous_native_intervals=bool(intervals), count_sum_matches_rows=int(counts.sum()) == len(mass),
        exported_ids_unique=len(np.unique(rows)) == len(rows), ids_in_native_bounds=valid_rows,
        exported_ids_cover_all_native_rows=np.array_equal(np.sort(rows), np.arange(len(mass))),
        native_interval_matches_exported_voxel=bool(valid_rows and np.all((rows >= offsets[voxel]) & (rows < offsets[voxel] + counts[voxel]))),
        native_mass_matches_exported_afd=bool(valid_rows and np.array_equal(mass[rows], d['afd'].reshape(-1, 3)[present])),
        native_direction_matches_exported_direction=bool(valid_rows and np.array_equal(expected_axes[rows], d['fiber_direction'].reshape(-1, 3, 3)[present])),
        native_files={name: digest(native / 'fixels' / name) for name in ['index.nii', 'mass.nii', 'directions.nii']})
    required = ['contiguous_native_intervals', 'count_sum_matches_rows', 'exported_ids_unique',
                'ids_in_native_bounds', 'native_interval_matches_exported_voxel',
                'native_mass_matches_exported_afd', 'native_direction_matches_exported_direction']
    results['all_checks_pass'] = all(results[key] for key in required)
    return results


def native_crosswalk(left, right, out):
    """Compare every sparse row, preserving local native lobe order exactly."""
    left, right, out = Path(left), Path(right), Path(out)
    refs, indices, sources = [], [], []
    for folder in (left, right):
        ref, index = read(folder / 'index.nii')
        refs.append(ref)
        indices.append(index.astype(np.int64))
        sources.append({name: digest(folder / name) for name in
                        ('index.nii', 'mass.nii', 'directions.nii', 'peak_amplitude.nii')})
    result = dict(left=str(left), right=str(right), source_hashes=sources,
                  shape_equal=refs[0].shape == refs[1].shape,
                  affine_exact=np.array_equal(refs[0].affine, refs[1].affine))
    if not result['shape_equal']:
        result['all_native_lobes_exact_under_row_permutation'] = False
        return result
    counts = [index[..., 0].ravel() for index in indices]
    offsets = [index[..., 1].ravel() for index in indices]
    result['native_count_per_voxel_exact'] = np.array_equal(*counts)
    if not result['native_count_per_voxel_exact'] or np.any(counts[0] < 0):
        result['all_native_lobes_exact_under_row_permutation'] = False
        return result
    occupied = np.flatnonzero(counts[0] > 0)
    voxel = np.repeat(occupied, counts[0][occupied])
    local = np.arange(len(voxel)) - np.repeat(np.r_[0, np.cumsum(counts[0][occupied])[:-1]], counts[0][occupied])
    rows = [offset[voxel] + local for offset in offsets]
    valid = [np.array_equal(np.sort(row), np.arange(len(voxel))) for row in rows]
    result.update(native_rows=len(voxel), left_intervals_bijective=valid[0], right_intervals_bijective=valid[1])
    if not all(valid):
        result['all_native_lobes_exact_under_row_permutation'] = False
        return result
    checks = {}
    for name in ('mass', 'peak_amplitude', 'directions'):
        a, b = [read(folder / (name + '.nii'))[1] for folder in (left, right)]
        a = a.reshape(-1, 3) if name == 'directions' else a.ravel()
        b = b.reshape(-1, 3) if name == 'directions' else b.ravel()
        length_match = len(a) == len(b) == len(voxel)
        entry = dict(dtype_exact=a.dtype == b.dtype, rows_match_index=length_match)
        if length_match:
            a, b = a[rows[0]], b[rows[1]]
            entry.update(values_exact=np.array_equal(a, b, equal_nan=True),
                         missingness_exact=np.array_equal(np.isfinite(a), np.isfinite(b)),
                         all_values_finite=bool(np.isfinite(a).all() and np.isfinite(b).all()))
            if name == 'directions':
                entry['axial_values_exact'] = bool(np.all(np.all(a == b, axis=1) | np.all(a == -b, axis=1)))
            if name == 'mass':
                entry['positive_lobes_left'] = int(np.count_nonzero(a > 0))
                entry['positive_lobes_right'] = int(np.count_nonzero(b > 0))
        checks[name] = entry
    crosswalk = out / 'all_native_rows.npz'
    np.savez_compressed(crosswalk, left_native_id=rows[0], right_native_id=rows[1],
                        flat_C_voxel=voxel, local_native_slot=local)
    result.update(fields=checks, changed_raw_ids=int(np.count_nonzero(rows[0] != rows[1])),
                  crosswalk=dict(path=crosswalk.name, sha256=digest(crosswalk), rows=len(voxel)))
    result['all_native_lobes_exact_under_row_permutation'] = bool(
        result['affine_exact'] and all(v['dtype_exact'] and v['rows_match_index'] and
        v['missingness_exact'] and v['all_values_finite'] and
        v.get('axial_values_exact', v['values_exact']) for v in checks.values()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path)
    parser.add_argument('--native-left', type=Path)
    parser.add_argument('--native-right', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if bool(args.suite) == bool(args.native_left) or bool(args.native_left) != bool(args.native_right):
        parser.error('Supply --suite OR both --native-left and --native-right')
    args.out.mkdir(parents=True, exist_ok=False)
    if args.native_left:
        result = native_crosswalk(args.native_left, args.native_right, args.out)
        (args.out / 'native_identity_diagnostic.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
        return 0 if result['all_native_lobes_exact_under_row_permutation'] else 1
    pairs = []
    for module in ['laminarity', 'dispersion']:
        for folder in ['only-' + module, 'retained', 'baseline-' + module]:
            if (args.suite / folder / 'run.json').exists() or (args.suite / folder / 'fixel_maps/native_fixel_id.nii.gz').exists():
                pairs.append(paired(args.suite, 'all', folder, module, args.out))
    native = [retained_native(args.suite, module) for module in ['laminarity', 'dispersion']]
    result = dict(scope='Read-only diagnosis; original strict acceptance outcomes remain unchanged',
        acceptance_sha256_at_diagnosis=digest(args.suite / 'acceptance.json'),
        pairs=pairs, retained_native_checks=native,
        all_reported_identity_differences_are_bijective_permutations=all(row['permutation_only'] for row in pairs),
        retained_native_linkage_proved=all(row['all_checks_pass'] for row in native),
        limitation='Default runs removed raw sparse arrays; their published identity maps and preserved extraction hashes remain. Retained run independently checks exported IDs against original sparse offsets and vectors.')
    (args.out / 'identity_diagnostic.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['pairs', 'retained_native_checks']}, indent=2))
    return 0 if result['all_reported_identity_differences_are_bijective_permutations'] and result['retained_native_linkage_proved'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

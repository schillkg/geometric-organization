# Validation

Validation checks correspondence with the original measurement engines and the
integrity of the combined workflow. It does not establish anatomical ground
truth or reproducibility across a cohort. Subject images are not distributed.

## Version 0.1.1

The measurement engines, numerical defaults and dependency runtime are unchanged
from 0.1.0. This release exposes the original measurement controls through the
combined command, adds explicit Laminarity scale lists, and preserves each
module's settings in the saved recipe and reports.

- **69 unit checks passed inside the installed container.** These cover
  input/grid checks, module selection, option validation, resume identity,
  publication, missing values, retained geometry and portable reports.
- The installed default command passed an exact comparison against the validated
  0.1.0 synthetic reference. Independently extracted native IDs were compared
  through complete native-lobe and reported-ID crosswalks. Strict raw ID-array
  comparison failures remain in the validation evidence.
- **Custom settings matched the frozen engines exactly.** Sheets used radius
  5.25 mm, 80 draws, seed 42, qref 0.05 and batch size 7: all 17 scientific
  images matched, including 8,064 finite q values. Whole-WM and native/eroded
  regional summaries matched. Laminarity used a fixed 4-mm scale; Dispersion
  changed all eight fitting/matching settings. Across all three measures, all 74 scientific maps and five regional CSVs
  matched at zero tolerance.
- **Laminarity selectors were exercised through the installed command.** A
  minimum of 3 mm produced 3, 4, 6 and 8 mm; an unsorted explicit list produced
  the sorted 3, 4 and 8-mm sweep. Saved scale maps and configuration agreed.
- **Actual DWI reconstruction to 1 mm passed.** A deterministic 70-volume
  single-shell input at 1.25 mm was resampled, reconstructed and processed by
  all three measures. Output grids were exactly 1 mm and shared across modules;
  q, L, T and ODI all had finite measurements. This was a small software control,
  not a second full-subject pilot.
- Already aligned JHU labels were reused without registration. `--no-report`
  omitted the public PDFs; reports were subsequently regenerated from a
  read-only portable result without original inputs, FOD or work directories.
  **All 33 custom report pages passed visual review**, with the selected settings
  and supplied-label captions verified.
- Changed-setting resume attempts were refused, and validation inputs remained
  unchanged. Owned successful scratch was removed.

The custom Dispersion check used fit angle 25°, minimum separation 40°, peak
threshold 0.2, maximum four fitted lobes, 724 sampling directions, match angle
15°, match margin 7° and axis gap 0.2. These are test settings, not suggested
replacements for the defaults.

Two acceptance-harness bookkeeping checks needed correction: one looked for
stage inventories deliberately removed on completion; another expected a
rasterized panel title in extracted PDF text. Original failure records were
retained, prepared inputs were verified against saved hashes, and remaining
checks continued without changing or rerunning successful measurements. The
application had no failure in these runs.

The [release manifest](container/release.json) identifies the exact image,
application source files, dependency runtime and validation scope. Published
images are promoted from tested candidates without rebuilding them.

## Locked 0.1.0 reference

The original image and six integration source files are preserved. The image's
SHA256 is
`d9a8118a5179ff9b49071b48434bfd3d6bf31b5f1d5a3fa35419b6b2ddc30094`.
Its size is 1,453,150,208 bytes. The
[reference record](docs/validated_baseline.json) preserves the source hashes.

### Full diffusion and FreeSurfer pilot

One full subject was run at the default 1.25-mm grid, with a shared reconstruction,
anatomical alignment and JHU registration. Independent runs of the frozen engines
used the same prepared inputs.

| Check | Result |
|---|---|
| Scientific images | 76 of 78 arrays exactly equal, including dtype, affine and missingness |
| Remaining two images | Raw native-fixel row IDs; exact after complete lobe/ID crosswalks |
| Native-lobe proof | All 4,938,608 positive lobes matched in each comparison |
| Reported-fixel proof | All 1,198,381 reported IDs matched in each comparison |
| Regional CSVs | Five byte-identical tables |
| Report inspection | All 34 pages visually reviewed |
| Portable reports | All 34 regenerated pages pixel-identical using only compact results |
| Runtime | About 57 minutes on the tested Linux host |
| Maximum resident memory | About 8.2 GiB |
| Compact results | About 380 MB |

Native IDs are row addresses assigned by an extraction, not biological labels.
Independent extractions can enumerate the same populations differently. The
validation retains the raw comparisons and separately proves a complete mapping
using location, direction and apparent fiber density; it does not relax tolerances
on measurement values.

Within a combined run, Laminarity and Dispersion use the same native extraction.
Their direction exports can differ by at most 5.96e-8 because the frozen
Laminarity exporter normalizes directions while Dispersion copies float32 values.
The publisher therefore keeps separate identity maps whenever their exports are
not exactly equal.

### Small controls and historical comparison

The 0.1.0 acceptance suite included 40 core tests, 28 installed commands and
42 operational checks. Across the small-fixture comparisons, 228 of 234 maps
were array-exact; six raw ID arrays had exact semantic crosswalks.

A historical standalone Sheets run using the same raw inputs had exactly equal
q values wherever both runs were valid. Coverage changed with the WM domain;
the historical and shared masks had Dice 0.894012. FA, mean b0, brain mask,
gradients and sampled field hashes matched. Historical 48-region and shared
50-region tables have different definitions and were not treated as equivalent.

## What changes when a setting changes

The 1-mm option changes the reconstruction grid. Sheets radius changes its spatial
neighborhood, Laminarity scale controls change its integration sigmas, and
Dispersion controls change angular fitting or matching. Those choices can change
the measurements; equality with a default recipe is neither expected nor claimed.
Custom controls are checked against the original engine at the same settings.

Grid, settings, source hashes, validity and coverage are retained with each result.
Changing a setting requires a fresh result directory; `--resume` accepts only an
identical recipe. Inspect alignment and coverage for every new dataset.

## Repeating software checks

See [container checks](container/README.md#local-checks-in-the-released-environment).
The synthetic generators and installed acceptance runners are included in `tests/`.
Their comparisons require exact values, affine matrices, dtypes and missingness,
with complete ID crosswalks only where independent extraction order differs.

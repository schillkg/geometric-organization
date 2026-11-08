# Geometric organization

[Website](https://schillkg.github.io/geometric-organization/) · [Download](https://github.com/schillkg/geometric-organization/releases/latest) · [Validation](VALIDATION.md)

Geometric organization runs Sheets, Laminarity and Dispersion together, using one fiber orientation distribution (FOD), one analysis grid and one optional atlas registration. An FOD describes the fiber directions supported by the diffusion signal within each voxel.

| Measure | Intuitive question | Main output |
|---|---|---|
| **Sheets** | Can two local fiber directions fit together into a smoothly varying sheet? | **q**, a local departure from sheet consistency. Lower q means less departure; it does not prove an anatomical sheet. |
| **Laminarity** | Does the spatial contrast around a fiber population look more sheet-like or tube-like? | **L** (lamellarity) and **T** (tubularity), with the spatial scale and geometry supporting each estimate. |
| **Dispersion** | How wide is each local FOD lobe, and is its spread circular or elongated? | **ODI**, dispersion anisotropy **DA_B**, angular widths and fitted axes. |

These measures describe different properties and have different units and denominators. They are not interchangeable scores or probabilities of tissue type. A voxel can contain several fiber populations; a *fixel* is one such population within one voxel.

## Download

Download the **Linux x86_64** Apptainer/Singularity image from [Releases](https://github.com/schillkg/geometric-organization/releases/latest). It is approximately **1.45 GB** and contains all three engines and their dependencies. You do not need a separate engine image or host Python installation.

[Download geometric-organization-0.1.1.sif](https://github.com/schillkg/geometric-organization/releases/download/v0.1.1/geometric-organization-0.1.1.sif)

Download `SHA256SUMS` from the same release and verify the file:

```bash
sha256sum --check SHA256SUMS --ignore-missing
```

You need [Apptainer](https://apptainer.org/docs/user/latest/quick_start.html) or Singularity on Linux. Use a Linux workstation or cluster for computation; macOS and Windows users can prepare inputs and inspect outputs locally.

The bundled JHU atlas and FMRIB58 reference have FSL noncommercial terms. Other dependencies retain their own licenses. The release includes a companion notices/source archive; see [distribution notes](DISTRIBUTION.md).

## Run your first subject

Start with **preprocessed diffusion data**, its corresponding b-values and rotated b-vectors, and a white-matter mask aligned to the input DWI. Preprocessing is upstream of this pipeline.

Place inputs in `input/`, create `results/`, and run:

```bash
mkdir -p results
apptainer run --cleanenv --containall \
  --bind "$PWD/input:/input:ro,$PWD/results:/output" \
  geometric-organization-0.1.1.sif \
  --dwi /input/dwi.nii.gz \
  --bval /input/bvals --bvec /input/bvecs \
  --wm-mask /input/wm_mask.nii.gz \
  --out /output/sub-01 --threads 4
```

All three modules, regional summaries and PDF reports run by default. The output folder must be new. On installations that use Singularity, replace `apptainer` with `singularity`.

An existing FreeSurfer reconstruction can supply the WM domain instead: bind it read-only and replace `--wm-mask` with `--freesurfer /input/freesurfer`. It must contain `mri/T1.mgz` and `mri/aseg.mgz`. The pipeline aligns the existing anatomy once; it does not run FreeSurfer. The derived domain includes brainstem label 16.

The WM mask selects **Sheets measurement centers only**. It does not clip Laminarity or Dispersion. Optional `--brain-mask` supplies the aligned input-grid observation mask; otherwise DWI mode generates one. A WM source is unnecessary when Sheets is not selected.

## Choose the grid and measurement settings

Grid spacing, the Sheets neighborhood radius, Laminarity integration sigma and Dispersion fitting angle are separate quantities. Changing one does not silently change the others. All defaults reproduce the validated baseline.

Append any of these to the run command above:

```bash
# Shared 1-mm isotropic reconstruction
--spacing 1

# Sheets: change the q neighborhood radius
--sheets-radius-mm 5

# Laminarity: keep only default scales of 3 mm and above
--laminarity-min-scale-mm 3

# Laminarity: calculate one fixed scale
--laminarity-fixed-scale-mm 4

# Laminarity: supply a scale list
--laminarity-scales-mm 2.5 4 6

# Dispersion: change the angular fitting cap
--dispersion-fit-angle 25
```

Use only one Laminarity scale selector per run. Integration scales are Gaussian sigmas in millimeters, not estimates of sheet thickness. Dispersion angles are in degrees; it has no spatial smoothing scale.

### Shared options

| Option | Default | Applies to |
|---|---|---|
| `--modules` | `sheets,laminarity,dispersion` | Any comma-separated subset. |
| `--spacing MM` | `1.25` for DWI | One common isotropic grid before FOD estimation. Use `1` for 1 mm. |
| `--native-spacing` | Off | Use the input DWI's isotropic spacing. Mutually exclusive with `--spacing`. |
| `--csd-mode` | `auto` | One reconstruction: single-shell CSD or multi-shell MSMT. Explicit `sscsd`/`msmt` must agree with the acquired shells. |
| `--threads` | `4` | Requested worker count; Laminarity uses at most two L/T workers. |
| `--subject` | Output folder name | Subject label in reports and provenance. |
| `--brain-mask` | Generated in DWI mode | Brain observations on the input grid. Required for a prepared FOD. |
| `--jhu-labels` | Register the atlas | Reuse existing JHU IDs 1–50 on the exact final grid, shared by all three measures. |
| `--no-jhu` | Off | Skip the shared atlas and regional tables. |
| `--regional-wm-mask` | None | Final-grid support for Laminarity/Dispersion regional summaries only. Does not restrict fitting or change Sheets. |
| `--no-report` | Off | Skip PDFs now; regenerate them later from compact results. |
| `--resume` | Off | Resume or verify the identical recipe; changed inputs/settings are refused. |

A finer grid does not recover detail absent from the acquisition. FreeSurfer anatomy and JHU labels are aligned once for the whole run. The earlier standalone registration selectors are replaced by this shared ANTs route.

### Sheets options

| Option | Default | Meaning |
|---|---|---|
| `--sheets-radius-mm` | `6.875` | Physical neighborhood radius for q fitting. |
| `--sheets-draws` | `100` | Number of structural FOD draws. |
| `--sheets-seed` | `20261015` | Nonnegative random seed. |
| `--sheets-qref` | `0.03` | Display reference in mm⁻¹; changes display/tensor weights, not q. |
| `--sheets-sample-batch-size` | `4` | Draw batching; retain it for exact reproducibility. |

Radius and qref must be positive; draws/batch size must be positive integers. Radius is limited to 16 times the voxel spacing by the frozen estimator's neighborhood indexing. Small neighborhoods or inadequate support can give unavailable estimates. FOD draws describe angular structure, not acquisition bootstrap uncertainty.

### Laminarity options

| Option | Default | Meaning |
|---|---|---|
| `--laminarity-min-scale-mm` | `2.5` | Restrict the default sweep; choices are `2.5`, `3`, `4`, `6`, `8`. |
| `--laminarity-fixed-scale-mm` | None | Calculate one positive integration sigma. |
| `--laminarity-scales-mm` | `2.5 3 4 6 8` | Explicit positive, unique sigmas; sorted before calculation. One value gives fixed-scale output. |

The three selectors are mutually exclusive. Derivative smoothing remains 0.9375 mm and the score normalization reference remains 1.25 mm, independently of `--spacing`. L and T select their maxima separately when more than one integration scale is used.

### Dispersion options

| Option | Default | Meaning |
|---|---|---|
| `--dispersion-fit-angle` | `30` | Fitting cap in degrees; greater than 0 and below 90. |
| `--dispersion-min-separation` | `35` | Minimum detected-peak separation in degrees, in (0, 90]. |
| `--dispersion-peak-threshold` | `0.1` | Peak amplitude relative to the largest detected peak, in (0, 1]. |
| `--dispersion-max-lobes` | `3` | Maximum fitted lobes, 1–10. Reported native slots remain strongest three. |
| `--dispersion-directions` | `2896` | Hemisphere sampling directions; at least 100. |
| `--dispersion-match-angle` | `20` | Maximum axial fit-to-native matching angle in degrees, in (0, 90]. |
| `--dispersion-match-margin` | `5` | Required unique-match margin in degrees, in [0, 90]. |
| `--dispersion-axis-gap` | `0.1` | Relative concentration gap for reporting transverse axes, in [0, 1). |

Changing fitting settings can change lobe detection, matching and measurement availability. Every effective setting is saved with the result. Use the same settings when comparing subjects or runs.

### Reuse a prepared FOD

Replace the DWI/gradient arguments with:

```text
--fod /input/WM_FOD.mif
--brain-mask /input/brain_mask.nii.gz
--wm-mask /input/wm_mask.nii.gz
--fa /input/FA.nii.gz
```

An optional aligned mean b0 can be supplied with `--b0`. Prepared FODs must use the **MRtrix Tournier07 basis, lmax 6 or 8, and an orthogonal isotropic grid**. All supplied images must align to that grid. Its existing spacing is preserved; a conflicting explicit `--spacing` is rejected. Oblique orthogonal grids are supported. No SH interpolation is silently applied to a prepared FOD.

## Read the outputs

Start with `QA/00_overview.pdf`, then the report for the measure of interest.

```text
sub-01/
  README.txt              file guide
  run.json                recipe, status and saved-file hashes
  checks.json             integrity and numerical checks
  QA/
    00_overview.pdf
    01_sheets.pdf
    02_lamellarity.pdf
    03_dispersion.pdf
  shared/
    anatomy/              FA, brain mask, b0; aligned T1 when supplied
    regions/              shared JHU labels, names and registration reference
    fixels/               identity maps, only when identical across modules
  sheets/
    q.nii.gz              pair channels 12, 13, 23; units mm^-1
    wm_mask.nii.gz         effective measurement domain
    summary.csv           whole-domain and regional statistics
    diagnostics/          availability, draw counts, support and reference peaks
    tensors/              reference-plane tensors and availability
    display/              monotone displays of q
    legacy_SPI/           historical SPI summaries with their own denominator
  laminarity/
    fixel_maps/           L, T, paired T, selected scales and sheet geometry
    voxel_maps/           AFD-weighted means, maxima and coverage
    regions/              regional summaries and scale curves
  dispersion/
    fixel_maps/           ODI, DA_B, widths, axes and validity
    voxel_maps/           AFD-weighted means, maxima and coverage
    regions/              regional summaries
  provenance/             settings, software records, responses and logs
  extras/                 only retained optional intermediates, if any
```

Only selected modules produce folders/reports; `shared/regions/` is absent with `--no-jhu`. Each module keeps its own identity maps when its frozen exports differ. Inspect `run.json` and the module report for the actual saved paths.

**Sheets:** q has three fixed pair channels, with pair 12 primary; the pipeline never chooses the lowest-q pair. Missing estimates remain NaN. `Srelative = 1 - q/qref` may be negative; `Sbounded = qref/(qref + q)` is a bounded display, not a probability.

**Laminarity:** L and T have up to three AFD-ranked slots per voxel. Because they maximize over scales independently, their sum need not be ≤1. The separately saved *T at the L-selected scale* supports paired interpretation. An undefined sheet plane remains missing even when a scalar estimate is available.

**Dispersion:** ODI is dimensionless; greater values mean broader fitted angular spread. DA_B describes unequal transverse spread, with zero circular. Width maps give half-maximum opening angles in degrees. Circular spread can have valid widths but no unique transverse axis. These describe FOD shape, not microscopic axon dispersion or NODDI tissue compartments.

## Use the saved tensors

Both tensor formats store six symmetric components in order **xx, yy, zz, xy, xz, yz**, in physical/scanner RAS coordinates. They are weighted plane tensors, not diffusion tensors.

| File | Shape | Definition |
|---|---|---|
| `sheets/tensors/pair12.nii.gz` (also `pair13`, `pair23`) | `(X,Y,Z,6)` | `w × (I − n nᵀ)`, where `w = qref/(qref+q)` and `n` is the normal from the pair's reference directions. |
| `sheets/tensors/available.nii.gz` | `(X,Y,Z,3)` | Availability for pairs 12, 13, 23. Unavailable tensors contain six NaNs. |
| `laminarity/fixel_maps/sheet_tensor.nii.gz` | `(X,Y,Z,3,6)` | `L × (I − n nᵀ)` at the L-selected scale; the fourth axis is the local fixel slot. |

Reconstruct a 3×3 matrix from either six-component array:

```python
import nibabel as nib
import numpy as np

six = nib.load("sheets/tensors/pair12.nii.gz").get_fdata(dtype=np.float32)
tensor = np.empty(six.shape[:-1] + (3, 3), dtype=six.dtype)
for k, (i, j) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
    tensor[..., i, j] = tensor[..., j, i] = six[..., k]
```

Native fixel IDs are extraction-local addresses, not tract labels or cross-subject identities. Sheets pairs and Laminarity/Dispersion fixel ranks are different identity systems.

Across modules, distinguish genuine zero from missingness, and inspect coverage before comparing means. Laminarity and Dispersion voxel means are weighted by native apparent fiber density (AFD), renormalized over valid fixels; maxima answer a different question. Regional tables use a shared atlas but retain each measure's own denominator.

## Keep extra files only when needed

Default output retains maps, geometry, coverage, summaries, anatomy, labels and reports. Successful-run scratch and the generated FOD are removed. Failed-run work is retained for diagnosis/resume.

| Flag | Extra retained output |
|---|---|
| `--save-fod` | Shared reconstructed FOD |
| `--save-native-fixels` | Complete original native-fixel extraction |
| `--save-registration` | Registration transforms and intermediates |
| `--sheets-save-draws` | Full Sheets draw arrays |
| `--laminarity-save-all-scales` | All L/T scale volumes |
| `--dispersion-save-parameters` | Full Bingham parameter bundle regardless of size |
| `--keep-work` | Successful-run work directory for debugging |

A Bingham parameter bundle is retained automatically only when its compressed total is ≤10,000,000 bytes. Module-specific save flags require that module to be selected.

## Regenerate reports after moving a result

Copy the **whole subject folder**. Original diffusion data, FOD and scratch are not needed for QA:

```bash
mkdir -p regenerated-qa
apptainer run --cleanenv --containall \
  --bind "$PWD/results/sub-01:/subject:ro,$PWD/regenerated-qa:/qa" \
  geometric-organization-0.1.1.sif \
  qa /subject --out /qa/reports
```

Use a fresh report destination. This regenerates reports from saved results; it does not rerun measurements.

## Validation and versions

Version 0.1.0 passed small-fixture controls and one full diffusion/FreeSurfer pilot. Independent frozen-engine comparisons found exact scientific outputs, with two raw native-ID arrays reconciled through complete exact lobe crosswalks. Five regional CSVs matched byte-for-byte. All 34 pilot report pages were reviewed, and all 34 regenerated pages matched pixel-for-pixel using only the portable compact result.

This establishes software correspondence for the tested inputs. It does not establish cohort reproducibility, diagnostic accuracy or anatomical ground truth. Inspect alignment, coverage and the module-specific limitations for every new dataset.


Version 0.1.1 exposes the standalone measurement controls while preserving the 0.1.0 defaults and frozen engines. Its additional tests and exact image checksum are recorded in [VALIDATION.md](VALIDATION.md) and the release assets. Version 0.1.0 remains the locked reference for the full pilot.

## GitHub Pages

The website is static content in `docs/`. To turn it off, open **Settings → Pages**, set the publishing branch to **None**, and save. The repository and container releases remain available. [GitHub instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/deleting-a-github-pages-site).

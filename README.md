# Geometric organization

[Website](https://schillkg.github.io/geometric-organization/) · [Download](https://github.com/schillkg/geometric-organization/releases/latest) · [Validation](VALIDATION.md)

This pipeline investigates three levels of the geometric organization of white matter. All are derived from diffusion MRI, using the fiber orientation distribution (FOD) within each voxel and the field of FODs across neighboring voxels.

- **Dispersion** describes how wide each FOD lobe is and whether its spread is circular or elongated. Outputs include the orientation dispersion index **ODI**, dispersion anisotropy **DA_B**, angular widths, fitted axes and fiber spread **FS**.
- **Laminarity** describes whether the spatial contrast around a fiber population looks more sheet-like or tube-like. Outputs include **L** (lamellarity), **T** (tubularity), their selected spatial scales and sheet geometry.
- **Sheets** describes whether two local fiber directions fit together into a smoothly varying sheet. The main output is **q**, a local departure from sheet consistency. See [Wedeen et al. (2012)](https://doi.org/10.1126/science.1215280), [Tax et al. (2016)](https://doi.org/10.1016/j.neuroimage.2016.07.042) and [Tax et al. (2017)](https://doi.org/10.1016/j.media.2017.03.007).

All three use the same FOD, analysis grid and atlas registration.

## Download

Download the **Linux x86_64** Apptainer/Singularity image from [Releases](https://github.com/schillkg/geometric-organization/releases/latest). It is approximately **1.45 GB** and contains all three engines and their dependencies. You do not need a separate engine image or host Python installation.

[Download geometric-organization-0.1.1.sif](https://github.com/schillkg/geometric-organization/releases/download/v0.1.1/geometric-organization-0.1.1.sif)

Download `SHA256SUMS` from the same release and verify the file:

```bash
sha256sum --check SHA256SUMS --ignore-missing
```

You need [Apptainer](https://apptainer.org/docs/user/latest/quick_start.html) or Singularity on Linux. Use a Linux workstation or cluster for computation; macOS and Windows users can prepare inputs and inspect outputs locally.

The bundled JHU atlas and FMRIB58 reference have FSL noncommercial terms. Other dependencies retain their own licenses. The release includes a companion notices/source archive; see [distribution notes](DISTRIBUTION.md).

## Run on an individual subject

Use **preprocessed diffusion data**, matching b-values and rotated b-vectors, and a white-matter mask aligned to the DWI. The WM mask is used for Sheets measurements only.

Place the files in `input/`, create `results/`, and run:

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

The `--bind` line makes your local `input/` folder available as `/input` inside the container and writes outputs to your local `results/` folder. The `:ro` setting makes the inputs read-only.

All three measures, regional summaries and PDF reports run by default. Use a new subject output folder. On systems that provide Singularity, replace `apptainer` with `singularity`.

## Optional inputs and parameters

### FreeSurfer segmentation

If you have already run FreeSurfer, the pipeline can use its segmentation to create the WM mask. For example, if the subject's FreeSurfer folder is `input/freesurfer/`, replace `--wm-mask /input/wm_mask.nii.gz` in the command above with:

```text
--freesurfer /input/freesurfer
```

That folder should contain `mri/T1.mgz` and `mri/aseg.mgz`. The pipeline aligns the T1 and segmentation to the diffusion data.

### Prepared FOD

To use an existing FOD, provide its aligned brain mask, WM mask and FA image:

```bash
mkdir -p results
apptainer run --cleanenv --containall \
  --bind "$PWD/input:/input:ro,$PWD/results:/output" \
  geometric-organization-0.1.1.sif \
  --fod /input/WM_FOD.mif \
  --brain-mask /input/brain_mask.nii.gz \
  --wm-mask /input/wm_mask.nii.gz \
  --fa /input/FA.nii.gz \
  --out /output/sub-01 --threads 4
```

The WM mask is needed when running Sheets. An aligned mean b0 is optional via `--b0`.

The FOD must use the **MRtrix Tournier07 basis, lmax 6 or 8, on an orthogonal isotropic grid**. All supplied images must align to that grid. A prepared FOD keeps its existing spacing.

### Shared settings

These settings apply to the whole run. DWI is resampled to **1.25 mm isotropic** before FOD estimation by default. To use a 1-mm grid for all three measures, add:

```text
--spacing 1
```

| Option | Default | Meaning |
|---|---|---|
| `--modules` | `sheets,laminarity,dispersion` | Measures to run, separated by commas. |
| `--spacing MM` | `1.25` for DWI | Isotropic reconstruction grid in millimeters. |
| `--native-spacing` | Off | Keep an already isotropic DWI grid; use instead of `--spacing`. |
| `--csd-mode` | `auto` | Choose CSD from the acquired shells; explicit choices are `sscsd` or `msmt`. |
| `--threads` | `4` | Requested worker count. |
| `--subject` | Output folder name | Subject label in reports. |
| `--brain-mask` | Generated for DWI | Aligned brain mask; required with a prepared FOD. |
| `--jhu-labels` | Register the atlas | Use JHU labels 1–50 already aligned to the final analysis grid. |
| `--no-jhu` | Off | Skip atlas registration and regional summaries. |
| `--regional-wm-mask` | None | Restrict Laminarity/Dispersion regional summaries to this final-grid mask. |
| `--no-report` | Off | Omit PDFs; they can be generated later from the saved results. |
| `--resume` | Off | Continue an interrupted run with the same inputs and settings. |

Anatomical alignment and JHU registration are shared across the three measures.

### Sheets parameters

These control the q calculation and its display maps. For a 5-mm neighborhood radius, add:

```text
--sheets-radius-mm 5
```

| Option | Default | Meaning |
|---|---|---|
| `--sheets-radius-mm` | `6.875` | Spatial neighborhood radius in millimeters. |
| `--sheets-draws` | `100` | Number of FOD sampling draws. |
| `--sheets-seed` | `20261015` | Random seed. |
| `--sheets-qref` | `0.03` | Reference q in mm⁻¹ for the [display maps and tensor weights](#sheets-outputs). |
| `--sheets-sample-batch-size` | `4` | Number of draws sampled together. |

The radius is independent of voxel spacing and can be up to 16 times the voxel size.

### Laminarity parameters

Laminarity evaluates spatial scales of **2.5, 3, 4, 6 and 8 mm** by default. Scales are Gaussian integration sigmas. Choose one of the following controls to change them:

| Option and example | Effect |
|---|---|
| `--laminarity-min-scale-mm 3` | Use 3, 4, 6 and 8 mm from the default scale list. Allowed minima: 2.5, 3, 4, 6 or 8 mm. |
| `--laminarity-fixed-scale-mm 4` | Calculate L and T at a single 4-mm scale. |
| `--laminarity-scales-mm 2.5 4 6` | Use an explicit list of positive, distinct scales. |

### Dispersion parameters

These control the angular Bingham fit to each FOD lobe and its matching to the local fiber populations. For a 25° fitting cap, add:

```text
--dispersion-fit-angle 25
```

| Option | Default | Meaning |
|---|---|---|
| `--dispersion-fit-angle` | `30` | Angular fitting cap in degrees; greater than 0 and below 90. |
| `--dispersion-min-separation` | `35` | Minimum detected-peak separation in degrees, in (0, 90]. |
| `--dispersion-peak-threshold` | `0.1` | Peak amplitude relative to the largest detected peak, in (0, 1]. |
| `--dispersion-max-lobes` | `3` | Maximum number of fitted lobes, from 1 to 10. |
| `--dispersion-directions` | `2896` | Hemisphere sampling directions; at least 100. |
| `--dispersion-match-angle` | `20` | Maximum angle between a fitted peak and its matched native fixel, in (0, 90] degrees. |
| `--dispersion-match-margin` | `5` | Angular margin required for a unique match, in [0, 90] degrees. |
| `--dispersion-axis-gap` | `0.1` | Minimum relative difference between concentrations for reporting transverse axes, in [0, 1). |

## Output files and folder structure

`QA/00_overview.pdf` summarizes the run. The other PDFs show the maps, distributions and regional results for each measure.

```text
sub-01/
  README.txt              file guide
  run.json                inputs, settings, status and saved-file hashes
  checks.json             numerical and file checks
  QA/
    00_overview.pdf
    01_sheets.pdf
    02_lamellarity.pdf
    03_dispersion.pdf
  shared/
    anatomy/              FA, brain mask, b0 and aligned T1 when supplied
    regions/              JHU labels, region names and atlas reference
    fixels/               shared fiber directions, AFD and IDs when identical
  sheets/
    q.nii.gz              q for direction pairs 12, 13 and 23
    wm_mask.nii.gz         WM mask used for the measurements
    summary.csv           whole-WM and regional statistics
    diagnostics/          availability, draw counts, support and reference peaks
    tensors/              plane tensors for each direction pair
    display/              Srelative and Sbounded maps derived from q
    legacy_SPI/           earlier SPI-based outputs
  laminarity/
    fixel_maps/           L, T, selected scales and sheet geometry
    voxel_maps/           AFD-weighted means, maxima and coverage
    regions/              regional summaries and scale curves
  dispersion/
    fixel_maps/           ODI, DA_B, widths, axes, FS and validity maps
    voxel_maps/           AFD-weighted means, maxima and coverage
    regions/              regional summaries
  provenance/             settings, software records, responses and logs
  extras/                 optional retained files
```

Folders and reports are created for the selected measures. Laminarity and Dispersion provide both fixel maps (one fiber population within a voxel) and voxel summaries. Voxel means are weighted by apparent fiber density (AFD); regional CSVs summarize measurements within the JHU parcels. Unavailable values are saved as NaN.

### Sheets outputs

Paths below are relative to `sheets/`.

- **q** (`q.nii.gz`): local departure from sheet consistency, in mm⁻¹. Its three channels correspond to direction pairs 12, 13 and 23.
- **Display maps** (`display/`): rescaled versions of q in which higher values indicate greater sheet consistency. `Srelative.nii.gz` uses the linear scaling `1 − q/qref`; `Sbounded.nii.gz` uses `qref/(qref + q)`, scaled between 0 and 1. The reference `qref` is set with `--sheets-qref`.
- **Plane tensors** (`tensors/pair12.nii.gz`, `pair13.nii.gz`, `pair23.nii.gz`): plane orientation from each direction pair, weighted by Sbounded. `tensors/available.nii.gz` records where each tensor is available.

### Laminarity outputs

The following files are in `laminarity/fixel_maps/`:

- **L and T** (`lamellarity.nii.gz`, `tubularity.nii.gz`): strength of sheet-like and tube-like spatial contrast around each fiber population.
- **Selected scales** (`lamellarity_scale_mm.nii.gz`, `tubularity_scale_mm.nii.gz`): the scale giving the strongest response for each measure, in millimeters. `tubularity_at_lamellarity_scale.nii.gz` also saves T evaluated at the L-selected scale.
- **Sheet geometry** (`sheet_normal.nii.gz`, `sheet_transverse.nii.gz`, `sheet_tensor.nii.gz`): the plane normal, in-plane transverse direction and L-weighted plane tensor.

### Dispersion outputs

The following files are in `dispersion/fixel_maps/`:

- **ODI** (`odi_total.nii.gz`): overall angular spread on a 0–1 scale; higher values indicate broader spread.
- **DA_B** (`dispersion_anisotropy.nii.gz`): elongation of the angular spread on a 0–1 scale; zero corresponds to circular spread.
- **Angular widths** (`hwhm_broad_deg.nii.gz`, `hwhm_narrow_deg.nii.gz`): angles from the fitted axis to half maximum along the broad and narrow directions, in degrees.
- **Fitted axes** (`fitted_peak.nii.gz`, `broad_axis.nii.gz`, `narrow_axis.nii.gz`): the central direction and broad/narrow transverse axes of the fitted lobe.
- **FS** (`fiber_spread_FS.nii.gz`): effective angular area of the fitted lobe, including both antipodal directions, in steradians.

## Plane tensors

Both tensor formats store six symmetric components in order **xx, yy, zz, xy, xz, yz**, in physical/scanner RAS coordinates. The plane normal is `n`.

| File | Shape | Definition |
|---|---|---|
| `sheets/tensors/pair12.nii.gz` (also `pair13`, `pair23`) | `(X,Y,Z,6)` | `Sbounded × (I − n nᵀ)` |
| `laminarity/fixel_maps/sheet_tensor.nii.gz` | `(X,Y,Z,3,6)` | `L × (I − n nᵀ)` at the L-selected scale; the fourth dimension is the fixel slot. |

Reconstruct a 3×3 matrix from either six-component array:

```python
import nibabel as nib
import numpy as np

six = nib.load("sheets/tensors/pair12.nii.gz").get_fdata(dtype=np.float32)
tensor = np.empty(six.shape[:-1] + (3, 3), dtype=six.dtype)
for k, (i, j) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
    tensor[..., i, j] = tensor[..., j, i] = six[..., k]
```

## Optional saved files

The default output includes maps, tensors, summaries, anatomy, labels and reports. Use these flags to keep additional files:

| Flag | Extra retained output |
|---|---|
| `--save-fod` | Shared reconstructed FOD |
| `--save-native-fixels` | Complete native-fixel extraction |
| `--save-registration` | Registration transforms and intermediate images |
| `--sheets-save-draws` | Full Sheets draw arrays |
| `--laminarity-save-all-scales` | All L/T scale volumes |
| `--dispersion-save-parameters` | Full Bingham parameter bundle, including directional ODI maps |
| `--keep-work` | Working files from a successful run |

The Bingham parameter bundle is retained automatically when its compressed size is ≤10,000,000 bytes. `--dispersion-save-parameters` keeps it regardless of size. Working files from failed runs are retained for `--resume`.

## Regenerate reports

Reports can be regenerated from the saved subject folder. For a result in `results/sub-01/`, run:

```bash
mkdir -p regenerated-qa
apptainer run --cleanenv --containall \
  --bind "$PWD/results/sub-01:/subject:ro,$PWD/regenerated-qa:/qa" \
  geometric-organization-0.1.1.sif \
  qa /subject --out /qa/reports
```

Use a new report destination. This command reads the saved measurements and rebuilds the PDFs.

## Validation

Version 0.1.1 was checked against the original engines at default and custom settings, including a DWI reconstruction to 1 mm. The earlier full-subject comparison remains the reference for the default pipeline. Comparison results, report checks and image hashes are recorded in [VALIDATION.md](VALIDATION.md) and the [release assets](https://github.com/schillkg/geometric-organization/releases/tag/v0.1.1).

## References

- Wedeen VJ et al. (2012). [The Geometric Structure of the Brain Fiber Pathways](https://doi.org/10.1126/science.1215280). *Science*, 335(6076), 1628–1634.
- Tax CMW et al. (2016). [Sheet Probability Index (SPI): Characterizing the geometrical organization of the white matter with diffusion MRI](https://doi.org/10.1016/j.neuroimage.2016.07.042). *NeuroImage*, 142, 260–279.
- Tax CMW et al. (2017). [Quantifying the brain’s sheet structure with normalized convolution](https://doi.org/10.1016/j.media.2017.03.007). *Medical Image Analysis*, 39, 162–177.

## GitHub Pages

The website is static content in `docs/`. To turn it off, open **Settings → Pages**, set the publishing branch to **None**, and save. The repository and container releases remain available. [GitHub instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/deleting-a-github-pages-site).

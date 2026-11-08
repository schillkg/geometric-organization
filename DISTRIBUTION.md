# Distribution and third-party materials

The container bundles several independently licensed projects. Their original
terms remain applicable; this page does not replace them or assign a blanket
license to the integration or measurement-engine code.

## Companion release asset

Download the [runtime notices and source materials](https://github.com/schillkg/geometric-organization/releases/download/v0.1.1/geometric-organization-0.1.0-third-party-materials.tar.gz)
alongside the container. The asset keeps `0.1.0` in its name because it records the
unchanged runtime reused by 0.1.1. New application controls do not replace these
third-party binaries.

- Filename: `geometric-organization-0.1.0-third-party-materials.tar.gz`
- Size: **11,581,376 bytes**
- SHA256: `f0b9f451a95460fe2637cd3b4318d656c8357502cffb7a46ba0a56579750b471`

The archive contains `THIRD_PARTY_NOTICES.md`, `SYSTEM_SOURCE_ACCESS.md`, component
license texts, source archives, provenance/hash records and `SHA256SUMS`. Preserve
these materials with redistributed copies. See the [release page](https://github.com/schillkg/geometric-organization/releases/tag/v0.1.1)
for the separate container checksum and version-specific checks.

## FSL atlas restriction

The bundled JHU ICBM-DTI-81 atlas and FMRIB58 FA reference carry the **FSL
noncommercial license**. A public download does not grant commercial-use
permission. Keep `LICENCE.FSL` and the atlas attribution with redistribution;
follow the license's conditions on passing terms and original materials to
recipients. The JHU atlas has its own owner, identified on the official
[FSL license page](https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html).

The image does not contain a FreeSurfer installation or FSL executable suite.
Some third-party Python and graphics libraries retain a historical `/opt/fsl/`
installation prefix; each keeps its own license.

## Other retained references

The unchanged Sheets engine also contains the ENIGMA FA reference and its
historical JHU atlas/lookup table. The [ENIGMA-DTI project record](https://www.nitrc.org/projects/enigma_dti/)
lists ENIGMA-owned resources as Public Domain; this does not replace the separate
FSL/JHU terms. All three files exactly match the primary ENIGMA protocol
repository. The integrated workflow uses the shared FMRIB58/JHU50 registration.

`TEMPLATE_NOTICES.md`, `template-inventory.json` and
`enigma-template-provenance.json` in the companion archive identify every retained
reference, exact hashes, upstream sources and attribution.

## Source and notice coverage

| Component | Included materials |
|---|---|
| MRtrix 3.0.4 | MPL-2.0 license and original source tar; all 1,293 source files verified against the installed source tree. |
| MRtrix 3.0.8 | MPL-2.0 license and exact clean source at commit `a4077b82190706373660d2a43e9329f2590d3422`. |
| ANTs 2.6.5 | Apache-2.0 license and source at installed-version commit `fdce4d2f84b60ac75d7e45aafe18488b1f9f5303`. |
| ITK | Notices for the exact dependency commit pinned by that ANTs source, including third-party notices. The inherited compiled build cache is unavailable, so its compiled revision is not independently verified. |
| Python, DIPY and scientific packages | Original installed license texts, bundled-library notices and font licenses. |
| Copied system libraries | Package copyright records for all 26 copied files, with exact binary/source package names and versions. |
| Conda runtime libraries | Notices/metadata for all 11 involved package names; FreeType's parent-package license texts recovered separately. |
| uv | MIT and Apache-2.0 license texts matching the installed version. |

MRtrix uses **MPL-2.0**, not GPL. Its exact source archives are provided in the
companion asset so recipients can obtain the source form of the bundled
executables. Original build/configuration scripts are included.

For inherited Ubuntu packages, `SYSTEM_SOURCE_ACCESS.md` supplies versioned
Launchpad source-package links and retrieval instructions. Package tables include
the full inherited OS inventory. Retrieve every file referenced by a package's
`.dsc`, including Ubuntu patches, and verify the published checksums. This is
explicit source-location information, **not an all-package source mirror**.

## Project-authored code

No standalone license file was found for the integration, Sheets, Laminarity or
Dispersion project code in the inspected snapshots. None is invented or inferred
here. Third-party licenses do not automatically grant rights to unrelated
project-authored files; the owner's chosen permission/license statement must be
provided separately.

The companion capture is a documented notice/source inventory. It does not claim
a complete legal audit of every inherited OS binary or statically linked
third-party dependency. Concrete coverage limits are retained in its manifest.

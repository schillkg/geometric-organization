# Container build

Use the [release container](https://github.com/schillkg/geometric-organization/releases/latest)
for ordinary analyses. The source in this repository is the integration layer;
the downloadable image also includes the frozen measurement engines and their
dependencies. See the [run guide](../README.md), [release manifest](release.json)
and [validation results](../VALIDATION.md).

The recipes below document the build from the two original, pinned engine
images. They require access to those images and are not a replacement for the
standalone download.

The final image is standalone. Frozen Sheets q1.0, Laminarity1.2.0 and
Dispersion1.0.0 sources remain unchanged inside it. One shared MRtrix3.0.4
reconstruction and ANTs2.6.5 registration serve the integrated pipeline;
the original Sheets Python3.12 environment is retained separately from the
Laminarity/Dispersion Python3.11 environment. No engine image or host source
mount is needed at run time.

Build the reusable dependency runtime once on an authorized Linux x86_64 host:

```sh
python3 container/build_runtime.py \
  --base /readonly/dispersion-1.0.0.sif \
  --sheets /readonly/sheets-q1.0.sif \
  --work /project/build/runtime-unique \
  --scratch /tmp/geometric-runtime-unique \
  --output /project/geometric-runtime-0.1.0.sif
```

Both source images must match hardcoded release SHA256 pins. The builder copies
the Sheets interpreter, standard library, virtual environment, needed libraries,
and unchanged source from the read-only source image. Four original MRtrix3.0.8
tools (sh2peaks, sh2amp, mrconvert, mrinfo) and their library preserve Sheets'
reference extraction from the shared FOD; they perform no second reconstruction.
It does not copy the FSL
command suite or FreeSurfer. The Dispersion base already contains Laminarity,
MRtrix and ANTs. Isolated imports, original CLI help, source hashes and package
versions are checked before completion.

Install the current application snapshot on the verified runtime:

```sh
python3 container/build.py \
  --runtime /project/geometric-runtime-0.1.0.sif \
  --runtime-record /project/build/runtime-unique/build.json \
  --work /project/build/application-unique \
  --scratch /tmp/geometric-application-unique \
  --output /project/geometric-organization-0.1.1.sif
```

All output/work/scratch paths must be fresh. The builders reject NFS scratch,
use low process priority and four compression workers, retain failed
build evidence, and remove only their own successful local scratch. Build logs,
source snapshots, manifests and image hashes stay outside source control.
Rebuilds can have different SIF hashes because container metadata has timestamps;
installed source/dependency hashes identify their contents.

Internal execution API (not ordinary investigator options):

| Launcher | Interpreter / import environment |
|---|---|
| `geo-python`, `laminarity-python` | `/opt/laminarity/venv/bin/python`; `/opt/laminarity/transverse` |
| `dispersion-python` | Same Python3.11; `/opt/dispersion/deps:/opt/dispersion/pipeline` |
| `sheets-python` | `/opt/venvs/sheets/bin/python`; `/opt/sheets-q/Measurement:/opt/sheets-q/Code/spi_python`, private copied libraries |

Use these launchers rather than mixing `PYTHONPATH` values. Default/L/D select
shared MRtrix3.0.4 and ANTs2.6.5. Sheets prepends `/opt/sheets-mrtrix3/bin` for
the four frozen reference-extraction tools. Original entry scripts remain available at
`/opt/sheets-q/Measurement/sheets_single.py`,
`/opt/laminarity/transverse/run_subject.py`, and
`/opt/dispersion/pipeline/dispersion.py`. Historical Sheets DWI/FLIRT baselines
use the original read-only release SIF; the integrated image intentionally has
one reconstruction/registration implementation.

The runtime image alone contains dependencies and engine CLIs. The second build
adds `/opt/geometric-organization/main.py` and makes the ordinary application
entrypoint runnable. Final acceptance must test this installed application image,
without a host source-code mount.

## Local checks in the released environment

Mount this checkout read-only to run its unit checks with the container's Python:

```sh
apptainer exec --cleanenv --containall \
  --bind "$PWD:/project:ro" geometric-organization-0.1.1.sif \
  geo-python -B -m unittest discover -s /project/tests -p 'test_*.py'
```

The installed acceptance runners in `tests/` additionally exercise the actual
container entrypoint and compare scientific outputs with the frozen engines.
They require fresh output/scratch paths and the documented synthetic/reference
fixtures; run each script with `--help` for its inputs. Acceptance records keep
strict failed ID-array comparisons alongside the complete native-lobe crosswalk
that explains extraction-local row permutations.

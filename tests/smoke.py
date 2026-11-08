"""Run the installed application once on the existing small prepared-FOD fixture."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--image', type=Path, required=True)
p.add_argument('--fixture', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--scratch', type=Path, required=True)
a = p.parse_args()
for key, value in vars(a).items(): setattr(a, key, value.resolve())
for path in (a.out, a.scratch):
    if path == Path('/nfs2') or Path('/nfs2') in path.parents: p.error('/nfs2 is read-only')
    path.mkdir(parents=True, exist_ok=False)
cmd = ['nice', '-n', '10', '/usr/bin/singularity', 'run', '--cleanenv', '--containall',
       '--no-home', '--no-mount', 'cwd,hostfs,bind-paths', '--pwd', '/',
       '--bind', f'{a.fixture}:/input:ro', '--bind', f'{a.out}:/output',
       '--bind', f'{a.scratch}:/tmp', a.image,
       '--fod', '/input/fod.nii.gz', '--brain-mask', '/input/brain_mask.nii.gz',
       '--wm-mask', '/input/brain_mask.nii.gz', '--fa', '/input/FA.nii.gz',
       '--out', '/output/subject', '--subject', 'tiny', '--threads', '2', '--no-jhu']
env = {k: v for k, v in os.environ.items() if not k.startswith(('APPTAINER_', 'SINGULARITY_'))}
env.update(TMPDIR=str(a.scratch), APPTAINER_TMPDIR=str(a.scratch), APPTAINER_CACHEDIR=str(a.scratch/'cache'))
record = dict(status='running', command=list(map(str, cmd)), image=str(a.image))
(a.out/'launch.json').write_text(json.dumps(record, indent=2)+'\n')
start = time.monotonic()
with (a.out/'run.log').open('w') as log:
    result = subprocess.run(list(map(str, cmd)), env=env, stdout=log, stderr=subprocess.STDOUT)
record.update(status='passed' if result.returncode == 0 else 'failed',
              returncode=result.returncode, elapsed_seconds=time.monotonic()-start)
(a.out/'exit.json').write_text(json.dumps(record, indent=2)+'\n')
raise SystemExit(result.returncode)

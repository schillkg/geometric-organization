#!/usr/bin/env python3
"""Install a frozen application snapshot on the verified integration runtime."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

SOURCE = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--runtime-record', type=Path, required=True, help='Successful build_runtime.py build.json')
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True, help='Fresh node-local directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for name in vars(args):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    for path in (args.work, args.scratch, args.output):
        if path == Path('/nfs2') or Path('/nfs2') in path.parents:
            parser.error('/nfs2 is read-only')
        if path.exists():
            parser.error(f'Choose a fresh path: {path}')
    if not (SOURCE/'main.py').is_file():
        parser.error('Application main.py is not present')
    version=re.search(r"^VERSION = '([^']+)'",(SOURCE/'main.py').read_text(),re.M).group(1)
    runtime = json.loads(args.runtime_record.read_text())
    if runtime.get('status') != 'complete' or sha(args.runtime) != runtime.get('output_sha256'):
        parser.error('Runtime image does not match a successful pinned build record')
    args.scratch.parent.mkdir(parents=True, exist_ok=True)
    filesystem = subprocess.check_output(['stat','-f','-c','%T',args.scratch.parent],text=True).strip()
    if filesystem.startswith('nfs'):
        parser.error('Build scratch must be node-local')
    args.work.mkdir(parents=True)
    args.scratch.mkdir()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stage = args.work/'source'
    stage.mkdir()
    ignored = {'container','audit','tests','data','results','validation','build','docs','__pycache__'}
    sources = {}
    for path in sorted(SOURCE.rglob('*')):
        relative = path.relative_to(SOURCE)
        if not path.is_file() or any(part.startswith('.') or part in ignored for part in relative.parts):
            continue
        if relative == Path('WORK_LOG.md'):
            continue
        if path.suffix not in ('.py','.json','.md','.txt','.sh'):
            continue
        target = stage/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path,target)
        sources[str(relative)] = {'sha256':sha(target),'bytes':target.stat().st_size}
    (stage/'application_manifest.json').write_text(json.dumps({'files':sources,
        'runtime_sha256':runtime['output_sha256'],'upstream_images':runtime['pins']},indent=2)+'\n')
    archive = args.work/'application.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for path in sorted(stage.iterdir()):
            tar.add(path,arcname=path.name)
    definition = args.work/'application.def'
    definition.write_text(f'''Bootstrap: localimage
From: {args.runtime}

%files
    application.tar.gz /opt/geometric-application.tar.gz

%post
    set -eu
    tar --no-same-owner -xzf /opt/geometric-application.tar.gz -C /opt/geometric-organization
    rm /opt/geometric-application.tar.gz
    chmod -R a+rX /opt/geometric-organization

%runscript
    exec /usr/local/bin/geometric-organization "$@"

%test
    /usr/local/bin/geometric-organization --help
    /usr/local/bin/geo-python /opt/geometric-organization/container/check_runtime.py

%labels
    org.opencontainers.image.title Geometric-Organization
    org.opencontainers.image.version {version}
    org.geometric-organization.runtime-sha256 {runtime['output_sha256']}
    org.geometric-organization.source-manifest-sha256 {sha(stage/'application_manifest.json')}
''')
    state = {'status':'building','started_utc':datetime.now(timezone.utc).isoformat(),
             'runtime_sha256':runtime['output_sha256'],'source_manifest_sha256':sha(stage/'application_manifest.json'),
             'application_archive_sha256':sha(archive),'definition_sha256':sha(definition),'sources':sources,
             'paths':{k:str(v) for k,v in vars(args).items()}}
    record = args.work/'build.json'
    record.write_text(json.dumps(state,indent=2)+'\n')
    env = os.environ.copy()
    env.update(APPTAINER_TMPDIR=str(args.scratch/'apptainer'),APPTAINER_CACHEDIR=str(args.scratch/'cache'),
               SINGULARITY_TMPDIR=str(args.scratch/'apptainer'),SINGULARITY_CACHEDIR=str(args.scratch/'cache'),
               TMPDIR=str(args.scratch/'tmp'))
    for key in ('APPTAINER_TMPDIR','APPTAINER_CACHEDIR','TMPDIR'):
        Path(env[key]).mkdir(parents=True,exist_ok=True)
    started = time.monotonic()
    try:
        if args.output.exists():
            raise FileExistsError(f'Refusing to replace an image created during preparation: {args.output}')
        command = ['nice','-n','10','/usr/bin/singularity','build','--fakeroot','--force','--disable-cache',
                   '--mksquashfs-args','-processors 4',str(args.output),str(definition)]
        with (args.work/'build.log').open('w') as stream:
            subprocess.run(command,check=True,env=env,cwd=args.work,stdout=stream,stderr=subprocess.STDOUT)
        verify = '''import hashlib,json,pathlib
r=pathlib.Path('/opt/geometric-organization')
m=json.loads((r/'application_manifest.json').read_text())
for p,v in m['files'].items():
 assert hashlib.sha256((r/p).read_bytes()).hexdigest()==v['sha256'],p
print(json.dumps({'status':'passed','application_files':len(m['files']),'runtime_sha256':m['runtime_sha256']}))
'''
        check = ['/usr/bin/singularity','exec','--cleanenv','--containall','--no-home',
                 '--no-mount','cwd,hostfs,bind-paths','--pwd','/',str(args.output),'/usr/local/bin/geo-python','-c',verify]
        result = subprocess.run(check,check=True,env=env,text=True,capture_output=True)
        (args.work/'installed_check.json').write_text(result.stdout)
        state.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat(),
                     elapsed_seconds=time.monotonic()-started,output_sha256=sha(args.output),
                     output_bytes=args.output.stat().st_size)
        shutil.rmtree(args.scratch)
    except BaseException as error:
        state.update(status='failed',error=str(error))
        raise
    finally:
        record.write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state,indent=2))


if __name__ == '__main__':
    main()

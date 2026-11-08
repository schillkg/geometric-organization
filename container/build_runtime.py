#!/usr/bin/env python3
"""Build one self-contained runtime from the two pinned, read-only images."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parent
PINS = {
    'base': '6339823e3b006ca6463ca84ae836210a90552c6d9d60f6dffcbe079b34de7320',
    'sheets': 'ff9d1f50119a1080d21044fb9131059e6df141c6d54ef0c85590846c680095e7',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, record):
    path.write_text(json.dumps(record, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True, help='Pinned Dispersion1.0.0 SIF')
    parser.add_argument('--sheets', type=Path, required=True, help='Pinned SHEETS q1.0 SIF')
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True, help='Fresh node-local directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verified-input-record', type=Path, help='Reuse completed image hashes only when exact path/size/mtime/pin still match')
    args = parser.parse_args()
    for name in ('base', 'sheets', 'work', 'scratch', 'output'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    for path in (args.work, args.scratch, args.output):
        if path == Path('/nfs2') or Path('/nfs2') in path.parents:
            parser.error('/nfs2 is read-only; no writable path may be there')
        if path.exists():
            parser.error(f'Choose a fresh path: {path}')
    args.scratch.parent.mkdir(parents=True, exist_ok=True)
    filesystem = subprocess.check_output(['stat', '-f', '-c', '%T', args.scratch.parent], text=True).strip()
    if filesystem.startswith('nfs'):
        parser.error('Build scratch must be node-local, not NFS')
    args.work.mkdir(parents=True)
    args.scratch.mkdir()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state = {'status':'checking', 'started_utc':now(), 'pins':PINS, 'paths':{k:str(v) for k,v in vars(args).items()}}
    write(args.work / 'build.json', state)
    started = time.monotonic()
    log = (args.work / 'build.log').open('w')
    env = os.environ.copy()
    env.update(APPTAINER_TMPDIR=str(args.scratch/'apptainer'), APPTAINER_CACHEDIR=str(args.scratch/'cache'),
               SINGULARITY_TMPDIR=str(args.scratch/'apptainer'), SINGULARITY_CACHEDIR=str(args.scratch/'cache'),
               TMPDIR=str(args.scratch/'tmp'))
    for key in ('APPTAINER_TMPDIR', 'APPTAINER_CACHEDIR', 'TMPDIR'):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    def run(command):
        print(' '.join(map(str, command)), flush=True)
        log.write(json.dumps(list(map(str, command))) + '\n'); log.flush()
        subprocess.run(list(map(str, command)), check=True, env=env, cwd=args.work,
                       stdout=log, stderr=subprocess.STDOUT)
    try:
        state['input_hashes'] = {}
        previous = json.loads(args.verified_input_record.read_text()) if args.verified_input_record else {}
        for name in ('base', 'sheets'):
            path = getattr(args,name)
            old = previous.get('input_hashes',{}).get(name,{})
            stat = path.stat()
            reusable = (previous.get('paths',{}).get(name) == str(path)
                        and old.get('sha256') == PINS[name]
                        and old.get('bytes') == stat.st_size and old.get('mtime_ns') == stat.st_mtime_ns)
            actual = old['sha256'] if reusable else sha(path)
            if actual != PINS[name]:
                raise ValueError(f'Pinned {name} image hash differs: {actual}')
            state['input_hashes'][name] = {'sha256':actual,'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,
                                          'verified_record_reused':str(args.verified_input_record) if reusable else None}
        write(args.work/'build.json',state)
        extraction = args.scratch/'sheets-runtime'
        extraction.mkdir()
        run(['nice','-n','10','/usr/bin/singularity','exec','--cleanenv','--containall','--no-home',
             '--no-mount','cwd,hostfs,bind-paths','--pwd','/',
             '--bind',f'{extraction}:/stage', '--bind',f'{ROOT}:/build-code:ro',
             args.sheets,'/opt/venvs/sheets/bin/python','/build-code/extract_sheets_runtime.py','/stage'])
        archive = args.work/'sheets-runtime.tar.gz'
        with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
            tar.add(extraction/'opt',arcname='opt')
        shutil.copy2(extraction/'runtime-files.json',args.work/'sheets-runtime-files.json')
        state['extracted_archive_sha256'] = sha(archive)
        for name in ('python-env','geometric-organization','check_runtime.py'):
            shutil.copy2(ROOT/name,args.work/name)
        definition = f'''Bootstrap: localimage
From: {args.base}

%files
    sheets-runtime.tar.gz /opt/geometric-runtime.tar.gz
    sheets-runtime-files.json /opt/geometric-organization/container/sheets-runtime-files.json
    python-env /usr/local/bin/geometric-python-env
    geometric-organization /usr/local/bin/geometric-organization
    check_runtime.py /opt/geometric-organization/container/check_runtime.py

%post
    set -eu
    tar --no-same-owner -xzf /opt/geometric-runtime.tar.gz -C /
    rm /opt/geometric-runtime.tar.gz
    chmod 755 /usr/local/bin/geometric-python-env /usr/local/bin/geometric-organization
    for name in geo-python laminarity-python dispersion-python sheets-python; do
        ln -s geometric-python-env /usr/local/bin/$name
    done

%environment
    export PATH=/opt/laminarity/venv/bin:/opt/laminarity/mrtrix3/bin:/opt/ants/bin:/usr/local/bin:/usr/bin:/bin
    export PYTHONPATH=/opt/laminarity/transverse
    export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
    export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
    export ANTSPATH=/opt/ants/bin

%runscript
    exec /usr/local/bin/geometric-organization "$@"

%test
    /usr/local/bin/geo-python /opt/geometric-organization/container/check_runtime.py

%labels
    org.opencontainers.image.title Geometric-Organization-Runtime
    org.opencontainers.image.version 0.1.0
    org.geometric-organization.dispersion-sha256 {PINS['base']}
    org.geometric-organization.sheets-sha256 {PINS['sheets']}
'''
        path = args.work/'runtime.def'
        path.write_text(definition)
        state.update(status='building', definition_sha256=sha(path))
        write(args.work/'build.json',state)
        if args.output.exists():
            raise FileExistsError(f'Refusing to replace an image created during preparation: {args.output}')
        run(['nice','-n','10','/usr/bin/singularity','build','--fakeroot','--force','--disable-cache',
             '--mksquashfs-args','-processors 4',args.output,path])
        check = args.work/'installed_check.json'
        cmd = ['/usr/bin/singularity','exec','--cleanenv','--containall','--no-home',
               '--no-mount','cwd,hostfs,bind-paths','--pwd','/',args.output,
               '/usr/local/bin/geo-python','/opt/geometric-organization/container/check_runtime.py']
        result = subprocess.run(list(map(str,cmd)),check=True,env=env,text=True,capture_output=True)
        check.write_text(result.stdout)
        json.loads(result.stdout)
        state.update(status='complete',finished_utc=now(),elapsed_seconds=time.monotonic()-started,
                     output_sha256=sha(args.output),output_bytes=args.output.stat().st_size,
                     installed_check_sha256=sha(check))
        for name in ('base','sheets'):
            stat = getattr(args,name).stat()
            record = state['input_hashes'][name]
            if (stat.st_size,stat.st_mtime_ns) != (record['bytes'],record['mtime_ns']):
                raise ValueError('Read-only source image changed during build')
        write(args.work/'build.json',state)
        shutil.rmtree(args.scratch)
    except BaseException as error:
        state.update(status='failed',error=str(error),failed_utc=now())
        write(args.work/'build.json',state)
        raise
    finally:
        log.close()
    print(json.dumps(state,indent=2))


if __name__ == '__main__':
    main()

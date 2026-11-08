#!/usr/bin/env python3
"""Bounded installed-image acceptance on the 2-mm 16x18x20 prepared-FOD fixture.

This does not launch a cohort or registration. It uses at most two threads and
executes the original frozen CLIs as independent same-input baselines. All
NIfTI comparison tolerances are fixed at zero before execution: a floating
difference fails for scientific review, rather than silently widening tolerance.
PDFs are generated and counted, but every page still needs human visual review.

Run only after the first installed-image smoke test has passed:
  python3 acceptance.py --image IMAGE.sif --fixture FIXTURE_DIR --out NEW_DIR
Optional --sheets-image uses the original q-1.0 SIF for its standalone baseline;
otherwise the image's pinned Sheets runtime is used. No host source is mounted.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


THREADS = 2
MODULES = ('sheets', 'laminarity', 'dispersion')
SUBJECT = 'acceptance'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def files(root):
    return {str(p.relative_to(root)): dict(bytes=p.stat().st_size, sha256=sha(p))
            for p in sorted(Path(root).rglob('*')) if p.is_file() and not p.is_symlink()}


def fixture_file(root, candidates):
    matches = [root / name for name in candidates if (root / name).is_file()]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one of {candidates} in {root}; found {matches}')
    return matches[0]


# Executed inside the image; host Python needs only its standard library.
CHECK_IMAGES = r'''
import json, sys
from pathlib import Path
import nibabel as nib
import numpy as np
job=json.loads(Path(sys.argv[1]).read_text()); rows=[]
for pair in job['pairs']:
    a,b=nib.load(pair['left']),nib.load(pair['right'])
    x,y=np.asanyarray(a.dataobj),np.asanyarray(b.dataobj)
    row=dict(module=pair.get('module'),relative=pair['relative'],shape_equal=a.shape==b.shape,
             affine_equal=np.array_equal(a.affine,b.affine),dtype_equal=x.dtype==y.dtype,
             left_dtype=str(x.dtype),right_dtype=str(y.dtype),left_shape=list(a.shape),right_shape=list(b.shape))
    row['exact']=False
    if row['shape_equal']:
        row['nan_mask_equal']=np.array_equal(np.isnan(x),np.isnan(y))
        row['finite_mask_equal']=np.array_equal(np.isfinite(x),np.isfinite(y))
        row['values_exact']=np.array_equal(x,y,equal_nan=True)
        finite=np.isfinite(x)&np.isfinite(y)
        if np.any(finite):
            u,v=x[finite].astype(float),y[finite].astype(float); delta=np.abs(u-v)
            row['absolute_error_p50_p95_p99_max']=np.percentile(delta,[50,95,99,100]).tolist()
            nz=np.abs(u)>0
            row['relative_error_p50_p95_p99_max']=np.percentile(delta[nz]/np.abs(u[nz]),[50,95,99,100]).tolist() if np.any(nz) else None
            row['unequal_finite_values']=int(np.count_nonzero(u!=v))
            row['finite_values']=int(len(u))
        if any(key in pair['relative'] for key in ('reference_peaks','fiber_direction','_axis','_normal')) and x.shape[-1] in (3,9):
            u,v=x.reshape(-1,3).astype(float),y.reshape(-1,3).astype(float)
            good=np.isfinite(u).all(axis=1)&np.isfinite(v).all(axis=1)&(np.linalg.norm(u,axis=1)>1e-12)&(np.linalg.norm(v,axis=1)>1e-12)
            if np.any(good):
                angle=np.degrees(np.arctan2(np.linalg.norm(np.cross(u[good],v[good]),axis=1),np.abs(np.sum(u[good]*v[good],axis=1))))
                row['axial_angle_error_deg_p50_p95_p99_max']=np.percentile(angle,[50,95,99,100]).tolist()
        row['exact']=all(row[k] for k in ('shape_equal','affine_equal','dtype_equal','nan_mask_equal','finite_mask_equal','values_exact'))
    rows.append(row)
result=dict(status='passed' if rows and all(r['exact'] for r in rows) else 'failed',
            tolerances=job['tolerances'],images=len(rows),comparisons=rows)
Path(sys.argv[2]).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:result[k] for k in ('status','images','tolerances')}))
'''

CHECK_FIXTURE = r'''
import json,sys
from pathlib import Path
import nibabel as nib
import numpy as np
fod,brain,fa=[nib.load(p) for p in sys.argv[1:4]]
assert fod.shape[:3]==(16,18,20) and fod.ndim==4 and fod.shape[3] in (28,45),'Use only the agreed small fixture'
spacing=np.linalg.norm(fod.affine[:3,:3],axis=0)
assert np.allclose(spacing,2,atol=1e-6,rtol=0),'Fixture must be 2 mm'
frame=fod.affine[:3,:3]/spacing
assert np.allclose(frame.T@frame,np.eye(3),atol=1e-6,rtol=0),'Fixture must be orthogonal'
mask=np.asarray(brain.dataobj)
assert brain.shape==fa.shape==fod.shape[:3]
assert np.array_equal(brain.affine,fod.affine) and np.array_equal(fa.affine,fod.affine)
assert np.isfinite(mask).all() and np.isin(mask,[0,1]).all() and mask.any(),'Brain must be nonempty and binary'
assert np.isfinite(np.asarray(fod.dataobj)[mask>0]).all()
assert np.isfinite(np.asarray(fa.dataobj)[mask>0]).all()
record=dict(shape=list(fod.shape),spacing_mm=spacing.tolist(),brain_voxels=int(mask.sum()),wm_source='same brain mask for this fixture only')
Path(sys.argv[4]).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
'''

CHECK_PDFS = r'''
import json,sys
from pathlib import Path
from pypdf import PdfReader
folder=Path(sys.argv[1]); rows={}
for path in sorted(folder.glob('*.pdf')):
    reader=PdfReader(path); assert len(reader.pages)>0,path
    rows[path.name]=dict(pages=len(reader.pages),bytes=path.stat().st_size)
    text=''.join(''.join((page.extract_text() or '').split()) for page in reader.pages)
    expected={'01_sheets.pdf':['sheets/summary.csv','shared/anatomy/'],
              '02_lamellarity.pdf':['laminarity/fixel_maps/','shared/anatomy/'],
              '03_dispersion.pdf':['dispersion/fixel_maps/','shared/anatomy/']}.get(path.name,[])
    assert all(item in text for item in expected),'Published guide paths missing: '+path.name
    rows[path.name]['published_guide_paths_verified']=bool(expected)
assert rows,'No PDFs'
Path(sys.argv[2]).write_text(json.dumps(dict(files=rows,visual_review='required for every page'),indent=2)+'\n')
print(json.dumps(rows))
'''

CHECK_COVERAGE = r'''
import json,sys
from pathlib import Path
import nibabel as nib
import numpy as np
root=Path(sys.argv[1]);records={}
for name,relative in [('q','sheets/q.nii.gz'),('lamellarity','laminarity/fixel_maps/lamellarity.nii.gz'),
                      ('tubularity','laminarity/fixel_maps/tubularity.nii.gz'),('ODI','dispersion/fixel_maps/odi_total.nii.gz')]:
    values=np.asarray(nib.load(root/relative).dataobj)
    finite=values[np.isfinite(values)]
    records[name]=dict(finite_values=int(len(finite)),range=[float(finite.min()),float(finite.max())] if len(finite) else None)
    if name=='q':records[name]['finite_by_pair']=[int(np.isfinite(values[...,i]).sum()) for i in range(3)]
records['status']='passed' if all(records[name]['finite_values']>0 for name in ('q','lamellarity','tubularity','ODI')) and records['q']['finite_by_pair'][0]>0 else 'failed'
Path(sys.argv[2]).write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records))
'''


class Suite:
    def __init__(self, args):
        self.image, self.fixture, self.out = (Path(p).expanduser().resolve() for p in (args.image,args.fixture,args.out))
        self.sheets_image = Path(args.sheets_image).expanduser().resolve() if args.sheets_image else None
        if self.out.exists():
            raise FileExistsError('Choose a fresh acceptance output directory: ' + str(self.out))
        if self.out == Path('/nfs2') or Path('/nfs2') in self.out.parents:
            raise ValueError('/nfs2 is read-only')
        if self.out == self.fixture or self.fixture in self.out.parents or self.out in self.fixture.parents:
            raise ValueError('Acceptance output must be separate from fixture inputs')
        for image in (self.image,self.sheets_image):
            if image and not image.is_file(): raise FileNotFoundError(image)
        self.runtime = shutil.which('apptainer') or shutil.which('singularity')
        if not self.runtime: raise FileNotFoundError('Apptainer/Singularity is required on the execution host')
        self.inputs = dict(fod=fixture_file(self.fixture,('fod.nii.gz','WM_FOD.nii.gz')),
                           brain=fixture_file(self.fixture,('brain_mask.nii.gz','mask.nii.gz')),
                           fa=fixture_file(self.fixture,('FA.nii.gz',)))
        self.initial_inputs = {k: dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for k,p in self.inputs.items()}
        self.out.mkdir(parents=True)
        for name in ('logs','scratch','container-cache','comparisons'):
            (self.out/name).mkdir()
        self.record = dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),
            threads=THREADS,scope='Small prepared-FOD software acceptance; no registration or full scan',
            image=dict(path=str(self.image),sha256=sha(self.image)),
            sheets_image=dict(path=str(self.sheets_image),sha256=None,hash_status='pending before standalone baseline') if self.sheets_image else None,
            inputs=self.initial_inputs,tolerances=dict(absolute=0,relative=0,affine=0,
                discrete='exact',missingness='exact',rule='Any difference fails pending explicit scientific review'),
            commands=[],checks=[],visual_review='Every generated and regenerated QA page remains to be visually inspected')
        self.save()

    def save(self):
        write_json(self.out/'acceptance.json',self.record)

    def check(self,name,condition,details=None):
        self.record['checks'].append(dict(name=name,status='passed' if condition else 'failed',details=details))
        self.save()
        return condition

    def container(self,label,arguments,*,image=None,bindings=None,expected_failure=None,fault_folder=None):
        number=len(self.record['commands'])+1
        scratch=self.out/'scratch'/f'{number:02d}-{label}';scratch.mkdir()
        bindings = bindings if bindings is not None else [
            (self.fixture,'/fixture','ro'),(self.out,'/test-output','rw')]
        mounts=[*bindings,(scratch,'/scratch','rw')]
        argv=['nice','-n','10',self.runtime,'exec','--cleanenv','--containall','--no-home',
              '--no-mount','cwd,hostfs,bind-paths','--pwd','/']
        for source,dest,mode in mounts:
            if mode!='ro' and (Path(source).resolve()==Path('/nfs2') or Path('/nfs2') in Path(source).resolve().parents):
                raise ValueError('Writable /nfs2 mount rejected')
            argv+=['--bind',f'{source}:{dest}:{mode}']
        for key,value in dict(TMPDIR='/scratch',MPLCONFIGDIR='/scratch/mpl',NUMBA_CACHE_DIR='/scratch/numba',
                              XDG_CACHE_HOME='/scratch/cache',PYTHONDONTWRITEBYTECODE='1',
                              NUMBA_NUM_THREADS=str(THREADS),ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=str(THREADS),
                              ANTS_RANDOM_SEED='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1').items():
            argv+=['--env',f'{key}={value}']
        argv += [str(image or self.image),*map(str,arguments)]
        env={key:value for key,value in os.environ.items()
             if not key.startswith(('APPTAINER_','SINGULARITY_','APPTAINERENV_','SINGULARITYENV_'))}
        env.update(APPTAINER_CACHEDIR=str(self.out/'container-cache'),
                   SINGULARITY_CACHEDIR=str(self.out/'container-cache'),
                   APPTAINER_TMPDIR=str(scratch),SINGULARITY_TMPDIR=str(scratch),TMPDIR=str(scratch))
        log=self.out/'logs'/f'{number:02d}-{label}.log'
        command=dict(label=label,argv=argv,log=str(log.relative_to(self.out)),expected_failure=expected_failure)
        self.record['commands'].append(command);self.save()
        started=time.monotonic()
        with log.open('w') as stream:
            if fault_folder is None:
                result=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,env=env,timeout=3600)
            else:
                # A new, owned file blocks module-directory creation. No image,
                # input, permission or estimator is altered. The driver records
                # the real stage failure; the identical run can then be resumed.
                process=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=env)
                blocker=self.out/fault_folder/'work/modules'
                payload=b'Owned acceptance fault: temporarily blocks the modules directory.\n'
                while process.poll() is None and time.monotonic()-started<120:
                    owner=blocker.parent/'.owner.json'
                    state_path=blocker.parent.parent/'run.json'
                    if owner.is_file() and state_path.is_file() and not blocker.exists():
                        state=json.loads(state_path.read_text())
                        if json.loads(owner.read_text()).get('run_id')==state.get('run_id'):
                            try:
                                with blocker.open('xb') as fault: fault.write(payload)
                            except FileExistsError:
                                pass
                            else:
                                command['injected_blocker']=str(blocker.relative_to(self.out))
                                command['injected_blocker_sha256']=sha(blocker)
                                self.save()
                                break
                    time.sleep(.02)
                try:
                    code=process.wait(timeout=max(1,3600-(time.monotonic()-started)))
                except subprocess.TimeoutExpired:
                    process.kill();process.wait()
                    raise
                result=subprocess.CompletedProcess(argv,code)
        command.update(returncode=result.returncode,seconds=time.monotonic()-started)
        text=log.read_text(errors='replace')
        passed=(result.returncode!=0 and expected_failure in text) if expected_failure else result.returncode==0
        command['status']='passed' if passed else 'failed';self.save()
        if not passed:
            raise RuntimeError(f'{label} failed expectation; see {log}\n{text[-4000:]}')
        if fault_folder is not None and 'injected_blocker' not in command:
            raise RuntimeError('Controlled owned-work fault was not injected; failure-resume was not tested')
        return result

    def common(self,folder,module=None):
        args=['geometric-organization','--fod','/fixture/'+self.inputs['fod'].name,
              '--brain-mask','/fixture/'+self.inputs['brain'].name,'--fa','/fixture/'+self.inputs['fa'].name,
              '--threads',str(THREADS),'--subject',SUBJECT,'--no-jhu','--out','/test-output/'+folder]
        if module: args+=['--modules',module]
        if module in (None,'sheets'): args+=['--wm-mask','/fixture/'+self.inputs['brain'].name]
        return args

    def public(self,folder):
        root=self.out/folder
        state=json.loads((root/'run.json').read_text())
        self.check(folder+': complete',state.get('status')=='complete')
        for relative,entry in state['public_files'].items():
            path=root/relative
            if not path.is_file() or path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:
                raise ValueError('Public inventory mismatch: '+str(path))
        return state

    def maps(self,folder,module):
        root=self.out/folder
        layout=json.loads((root/'provenance/layout.json').read_text())[module]
        allowed=('fixel_maps/','voxel_maps/') if module!='sheets' else ('diagnostics/','tensors/','display/','legacy_SPI/')
        return {old:'/test-output/'+folder+'/'+new for old,new in layout.items()
                if old.endswith(('.nii','.nii.gz')) and (old.startswith(allowed) or old in ('q.nii.gz','wm_mask.nii.gz'))}

    def compare(self,label,left,right,module,standalone=False):
        left_maps=self.maps(left,module)
        if standalone:
            allowed=('fixel_maps/','voxel_maps/') if module!='sheets' else ('diagnostics/','tensors/','display/','legacy_SPI/')
            right_maps={name:'/test-output/'+right+'/'+name for name in files(self.out/right)
                        if name.endswith(('.nii','.nii.gz')) and
                        (name.startswith(allowed) or name in ('q.nii.gz','wm_mask.nii.gz'))}
        else:
            right_maps=self.maps(right,module)
        self.check(label+': same map set',set(left_maps)==set(right_maps),
                   dict(left_only=sorted(set(left_maps)-set(right_maps)),right_only=sorted(set(right_maps)-set(left_maps))))
        pairs=[dict(module=module,relative=name,left=left_maps[name],right=right_maps[name])
               for name in sorted(set(left_maps)&set(right_maps))]
        if not pairs: raise ValueError('No scientific maps to compare: '+label)
        job=self.out/'comparisons'/f'{label}-input.json';report=self.out/'comparisons'/f'{label}.json'
        write_json(job,dict(tolerances=self.record['tolerances'],pairs=pairs))
        self.container(label,['geo-python','-c',CHECK_IMAGES,'/test-output/'+str(job.relative_to(self.out)),
                              '/test-output/'+str(report.relative_to(self.out))])
        result=json.loads(report.read_text())
        self.check(label,result['status']=='passed',dict(images=result['images'],report=str(report.relative_to(self.out))))
        if module=='sheets':
            a=self.out/left/'sheets/summary.csv'
            b=self.out/right/('summary.csv' if standalone else 'sheets/summary.csv')
            self.check(label+': summary exact',a.read_bytes()==b.read_bytes())

    def storage(self,folder,retained=False):
        root=self.out/folder;names=set(files(root))
        if retained:
            requirements={'FOD':root/'extras/fod/WM_FOD.nii.gz','native fixels':root/'extras/native_fixels',
                          'draws':root/'extras/sheets/draws','all scales':root/'extras/laminarity/scales',
                          'parameters':root/'extras/dispersion/parameters','work':root/'work'}
            for kind,path in requirements.items():
                self.check(folder+': retained '+kind,path.is_file() or (path.is_dir() and any(path.iterdir())))
        else:
            prohibited=[name for name in names if name.startswith(('work/','extras/fod/','extras/native_fixels/',
                'extras/sheets/draws/','extras/laminarity/scales/')) or '/mrtrix_fixels/' in name
                or name.endswith(('signed_plane_L.npy','signed_legacy_L.npy','support_fraction.npy'))
                or name.endswith(('.nii','.nii.gz','.mif')) and ('WM_FOD' in name or 'all_scales' in name)]
            self.check(folder+': compact defaults',not prohibited,prohibited)
            policy_path=root/'provenance/modules/dispersion/provenance/parameter_retention.json'
            policy=json.loads(policy_path.read_text())
            actual=sum(p.stat().st_size for p in (root/'extras/dispersion/parameters').rglob('*') if p.is_file())
            okay=(policy['mode']=='automatic' and policy['automatic_limit_bytes']==10_000_000
                  and policy['retained']==(policy['compressed_bytes']<=10_000_000)
                  and actual==(policy['compressed_bytes'] if policy['retained'] else 0))
            self.check(folder+': bounded automatic Bingham retention',okay,dict(policy=policy,actual_bytes=actual))
        self.check(folder+': no published symlinks',not any(p.is_symlink() for p in root.rglob('*')
                                                          if p.relative_to(root).parts[0]!='work'))
        self.container(folder+'-pdf-check',['geo-python','-c',CHECK_PDFS,'/test-output/'+folder+'/QA',
                       '/test-output/'+folder+'-pdfs.json'])
        pdfs=json.loads((self.out/(folder+'-pdfs.json')).read_text())['files']
        self.check(folder+': four reports',set(pdfs)=={'00_overview.pdf','01_sheets.pdf','02_lamellarity.pdf','03_dispersion.pdf'},pdfs)

    def run(self):
        try:
            self.container('installed-runtime',['geo-python','/opt/geometric-organization/container/check_runtime.py'])
            self.container('fixture-preflight',['geo-python','-c',CHECK_FIXTURE,
                '/fixture/'+self.inputs['fod'].name,'/fixture/'+self.inputs['brain'].name,
                '/fixture/'+self.inputs['fa'].name,'/test-output/fixture.json'])
            self.container('all-default',self.common('all'))
            self.public('all');self.storage('all')
            self.container('fixture-measurement-coverage',['geo-python','-c',CHECK_COVERAGE,
                           '/test-output/all','/test-output/measurement-coverage.json'])
            coverage=json.loads((self.out/'measurement-coverage.json').read_text())
            if not self.check('fixture exercises finite primary q, L, T and ODI',coverage['status']=='passed',coverage):
                raise ValueError('Fixture is degenerate for one or more primary measures; retain smoke evidence and use a suitable fixture')
            self.record['first_all_modules_ready_utc']=datetime.now(timezone.utc).isoformat();self.save()
            print('All modules, published QA guides and finite primary measures passed; subset/baseline checks continue.',flush=True)
            for module in MODULES:
                folder='only-'+module
                if module=='sheets':
                    self.container(folder+'-controlled-failure',self.common(folder,module),
                                   expected_failure='Command failed',fault_folder=folder)
                    failure=json.loads((self.out/folder/'run.json').read_text())
                    self.check('installed failed run preserves owned work',failure['status']=='failed'
                        and (self.out/folder/'work/.owner.json').is_file())
                    evidence=self.out/'failure-evidence';evidence.mkdir()
                    write_json(evidence/'failed_run.json',failure)
                    fault=self.record['commands'][-1]
                    blocker=self.out/fault['injected_blocker']
                    if not blocker.is_file() or sha(blocker)!=fault['injected_blocker_sha256']:
                        raise RuntimeError('Owned failure sentinel changed; refusing to remove anything')
                    blocker.replace(evidence/'injected_modules_blocker.txt')
                    self.container(folder+'-failed-resume',self.common(folder,module)+['--resume'])
                    log=self.out/self.record['commands'][-1]['log']
                    self.check('installed failed resume verifies completed preparation','prepare: verified' in log.read_text())
                else:
                    self.container(folder,self.common(folder,module))
                self.public(folder)
                self.compare('all-vs-only-'+module,'all',folder,module)
            retained=self.common('retained')+['--save-fod','--save-native-fixels','--sheets-save-draws',
                '--laminarity-save-all-scales','--dispersion-save-parameters','--keep-work']
            self.container('explicit-retention',retained);self.public('retained');self.storage('retained',True)
            for module in MODULES:
                self.compare('default-vs-retained-'+module,'all','retained',module)
            # Each original CLI independently prepares/extracts the identical supplied FOD.
            if self.sheets_image:
                self.record['sheets_image']['hash_status']='hashing before standalone baseline';self.save()
                self.record['sheets_image']['sha256']=sha(self.sheets_image)
                self.record['sheets_image']['hash_status']='fresh hash complete';self.save()
            for module in MODULES:
                folder='baseline-'+module
                image=self.sheets_image if module=='sheets' and self.sheets_image else self.image
                python='/opt/venvs/sheets/bin/python' if module=='sheets' and self.sheets_image else module+'-python'
                entry={'sheets':'/opt/sheets-q/Measurement/sheets_single.py',
                       'laminarity':'/opt/laminarity/transverse/run_subject.py',
                       'dispersion':'/opt/dispersion/pipeline/dispersion.py'}[module]
                args=[python,entry,'--fod','/fixture/'+self.inputs['fod'].name,
                    '--mask','/fixture/'+self.inputs['brain'].name,'--fa','/fixture/'+self.inputs['fa'].name,
                    '--out','/test-output/'+folder,'--subject',SUBJECT,'--threads',str(THREADS),'--no-jhu']
                if module=='sheets': args+=['--wm-mask','/fixture/'+self.inputs['brain'].name,
                    '--native-spacing','--draws','100','--radius-mm','6.875','--seed','20261015','--qref','0.03']
                else: args+=['--spacing','2','--no-report']
                self.container(folder,args,image=image)
                self.compare('original-vs-integrated-'+module,'all',folder,module,True)
            before=self.public('all')['public_files']
            self.container('completed-resume',self.common('all')+['--resume'])
            self.check('completed resume preserves public inventory',before==self.public('all')['public_files'])
            changed=self.common('all')+['--resume'];changed[changed.index('--threads')+1]='1'
            self.container('changed-recipe-refused',changed,expected_failure='Inputs, settings, or source changed')
            shutil.copytree(self.out/'all',self.out/'missing-public')
            missing=self.out/'missing-public/sheets/q.nii.gz';missing.unlink()
            self.container('missing-public-refused',self.common('missing-public')+['--resume'],
                           expected_failure='Missing or changed saved file')
            self.check('missing public file is not silently recomputed',not missing.exists())
            moved=self.out/'relocated';shutil.copytree(self.out/'all',moved)
            snapshot=files(moved)
            fresh=self.out/'qa-regenerated';fresh.mkdir()
            # Crucially no /fixture or /test-output mount: neither original inputs
            # nor original result/work paths can be reached by this regeneration.
            self.container('portable-qa',['geometric-organization','qa','/relocated','--out','/qa-output/reports'],
                           bindings=[(moved,'/relocated','ro'),(fresh,'/qa-output','rw')])
            self.check('portable QA leaves compact source unchanged',snapshot==files(moved))
            self.container('portable-qa-pdf-check',['geo-python','-c',CHECK_PDFS,'/qa-output/reports','/qa-output/pdf-check.json'],
                           bindings=[(fresh,'/qa-output','rw')])
            portable=json.loads((fresh/'pdf-check.json').read_text())['files']
            original=json.loads((self.out/'all-pdfs.json').read_text())['files']
            self.check('portable QA preserves report page counts',
                       {k:v['pages'] for k,v in portable.items()}=={k:v['pages'] for k,v in original.items()},portable)
            self.check('input hashes unchanged',all(sha(p)==self.initial_inputs[k]['sha256'] for k,p in self.inputs.items()))
            self.record['status']='passed' if all(c['status']=='passed' for c in self.record['checks']) else 'failed'
        except BaseException as error:
            self.record.update(status='failed',error=type(error).__name__+': '+str(error))
            raise
        finally:
            self.record['completed_utc']=datetime.now(timezone.utc).isoformat()
            self.record['inputs_unchanged']=all(sha(p)==self.initial_inputs[k]['sha256'] for k,p in self.inputs.items())
            self.save()
        return 0 if self.record['status']=='passed' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--fixture',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--sheets-image',type=Path)
    return Suite(parser.parse_args()).run()


if __name__=='__main__':
    raise SystemExit(main())

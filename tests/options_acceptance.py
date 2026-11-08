#!/usr/bin/env python3
"""Bounded installed 0.1.1 option acceptance; no application source is mounted.

Uses two threads, fresh owned outputs/scratch, exact scientific comparisons,
and the already reviewed native-row crosswalk for independent extractions.
The frozen lower-level engines receive the same prepared/native inputs for
custom-setting baselines. Synthetic data exercise software, not histology.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from acceptance import CHECK_IMAGES, CHECK_COVERAGE, CHECK_FIXTURE, sha, write_json, files

THREADS = 2
CUSTOM_D = dict(fit_angle=25., min_separation=40., peak_threshold=.2, max_lobes=4,
                directions=724, match_angle=15., match_margin=7., axis_gap=.2)
CUSTOM_Q = dict(radius_mm=5.25, draws=80, seed=42, qref=.05, sample_batch_size=7)


def writable_path(path):
    if path==Path('/nfs2') or Path('/nfs2') in path.parents:raise ValueError('/nfs2 is read-only')
    if any(c in str(path) for c in (':',',','\n')):raise ValueError('Unsafe container bind path')


def capacity(scratch,destination,threads):
    memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    cpu=os.cpu_count() or 1;load=os.getloadavg()
    record=dict(hostname=os.uname().nodename,logical_cpus=cpu,load_1_5_15=list(load),
                available_memory_bytes=memory['MemAvailable'],scratch_free_bytes=shutil.disk_usage(scratch).free,
                persistent_free_bytes=shutil.disk_usage(destination).free,requested_threads=threads)
    if threads>2 or load[0]+threads>.85*cpu or memory['MemAvailable']<8*1024**3:
        raise RuntimeError('Insufficient capacity for bounded tiny validation: '+json.dumps(record))
    if min(record['scratch_free_bytes'],record['persistent_free_bytes'])<4*1024**3:
        raise RuntimeError('Need at least4GiB free on each owned destination')
    return record


def stop(process):
    if process.poll() is not None:return
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait()


def copy_verified(source,destination,expected):
    if source.is_symlink() or destination.exists():raise ValueError('Ordinary source and fresh captured file required')
    before=source.stat();source_hash=sha(source)
    if before.st_size!=expected['bytes'] or source_hash!=expected['sha256']:raise ValueError('Stage/source hash mismatch')
    shutil.copy2(source,destination)
    after=source.stat();copied=sha(destination)
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or sha(source)!=source_hash or copied!=source_hash:
        raise ValueError('Source changed while capturing completed stage')
    return dict(source=str(source),bytes=before.st_size,recorded_input_sha256=expected['sha256'],source_sha256=source_hash,copied_sha256=copied)

MAKE_INPUTS = r'''
import hashlib,json,sys
from pathlib import Path
import nibabel as nib
import numpy as np
out=Path('/evidence/inputs');out.mkdir()
ref=nib.load('/fixture/fod.nii.gz'); labels=np.ones(ref.shape[:3],np.uint16)
labels[ref.shape[0]//2:]=2
nib.save(nib.Nifti1Image(labels,ref.affine),out/'JHU_labels.nii.gz')
# Deterministic single-shell diffusion signal: pure-fiber response regions
# surround a three-tensor crossing mixture. Full masks avoid mask-generation
# variability; native 1.25-mm signals are genuinely regridded before CSD.
shape=(16,18,20); affine=np.diag([1.25,1.25,1.25,1.]); affine[:3,3]=-np.array(shape)*1.25/2
n=64;i=np.arange(n);z=1-2*(i+.5)/n;phi=i*np.pi*(3-np.sqrt(5))
g=np.column_stack((np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z))
bvec=np.concatenate((np.zeros((6,3)),g));bval=np.r_[np.zeros(6),np.full(n,1000.)]
x,y,z=np.indices(shape,dtype=float);central=(x>3)&(x<12)&(y>3)&(y<14)&(z>3)&(z<16)
weights=np.zeros(shape+(3,));weights[...,0]=1
weights[central,0]=.48+.08*np.sin(y[central]/3)
weights[central,1]=.30+.05*np.cos(z[central]/4)
weights[central,2]=1-weights[central,0]-weights[central,1]
attenuation=np.exp(-bval[:,None]*(.0002+.0015*bvec**2))
signal=(950+60*np.cos(x/5)+30*np.sin(y/4))[...,None]*np.einsum('...k,nk->...n',weights,attenuation)
signal=signal.astype(np.float32)
nib.save(nib.Nifti1Image(signal,affine),out/'dwi.nii.gz')
nib.save(nib.Nifti1Image(np.ones(shape,np.uint8),affine),out/'dwi_mask.nii.gz')
np.savetxt(out/'dwi.bval',bval[None,:],fmt='%.8f');np.savetxt(out/'dwi.bvec',bvec.T,fmt='%.12f')
records={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in out.iterdir()}
(out/'generation.json').write_text(json.dumps(dict(shape=list(signal.shape),input_spacing_mm=1.25,
    b0_volumes=6,single_shell_directions=n,bvalue=1000,tensor_eigenvalues=[.0017,.0002,.0002],
    purpose='Software reconstruction/grid control; no biological validation claim',files=records),indent=2)+'\n')
'''

CHECK_SETTINGS = r'''
import json,sys
from pathlib import Path
import nibabel as nib
import numpy as np
root=Path(sys.argv[1]);expected=json.loads(sys.argv[2]);layout=json.loads((root/'provenance/layout.json').read_text())
manifest=json.loads((root/layout['laminarity']['output_manifest.json']).read_text())
assert manifest['scales_mm']==expected,(manifest['scales_mm'],expected)
assert manifest['scale_mode']==('fixed' if len(expected)==1 else 'max-selected')
assert manifest['configuration']['derivative_mm']==.9375 and manifest['configuration']['reference_mm']==1.25
for key in ('lamellarity','tubularity'):
    a=np.asarray(nib.load(root/layout['laminarity'][f'fixel_maps/{key}_scale_mm.nii.gz']).dataobj)
    assert np.isin(a[np.isfinite(a)],expected).all()
    assert np.isfinite(a).any()
print(json.dumps(dict(scales_mm=manifest['scales_mm'],mode=manifest['scale_mode'])))
'''

CHECK_ONE_MM = r'''
import json,sys
from pathlib import Path
import nibabel as nib
import numpy as np
root=Path('/evidence/dwi_1mm');layout=json.loads((root/'provenance/layout.json').read_text())
ref=nib.load(root/'shared/anatomy/FA.nii.gz');spacing=nib.affines.voxel_sizes(ref.affine)
assert np.array_equal(spacing,[1.,1.,1.]),spacing
checked=0
for module,mapping in layout.items():
    for old,new in mapping.items():
        if old.endswith(('.nii','.nii.gz')) and old.startswith(('fixel_maps/','voxel_maps/','tensors/','diagnostics/')):
            image=nib.load(root/new)
            assert image.shape[:3]==ref.shape and np.array_equal(image.affine,ref.affine),(module,old)
            checked+=1
profile=json.loads((root/'provenance/reconstruction/reconstruction.json').read_text())
assert profile['mode']=='sscsd' and profile['wm_lmax']==6 and profile['resampled']
assert profile['input_grid']['spacing_mm']==[1.25,1.25,1.25]
assert profile['output_grid']['spacing_mm']==[1.,1.,1.]
Path('/evidence/dwi_1mm_grid.json').write_text(json.dumps(dict(shape=list(ref.shape),spacing_mm=spacing.tolist(),
    checked_scientific_maps=checked,reconstruction=profile),indent=2)+'\n')
'''

CHECK_CUSTOM_QA = r'''
import json,sys
from pathlib import Path
from pypdf import PdfReader
root=Path('/qa/reports');record={}
for name in ('00_overview.pdf','01_sheets.pdf','02_lamellarity.pdf','03_dispersion.pdf'):
    reader=PdfReader(root/name);text=' '.join(' '.join(page.extract_text() or '' for page in reader.pages).split())
    record[name]=dict(pages=len(reader.pages),bytes=(root/name).stat().st_size)
    assert len(reader.pages)>0
    (root/(name+'.txt')).write_text(text)
    if name=='02_lamellarity.pdf':
        assert 'fixed integration sigma = 4 mm' in text and '0.9375' in text,text[:2000]
    if name=='03_dispersion.pdf':
        assert 'cap 25' in text and 'Relative amplitude 0.2' in text and 'minimum separation 40' in text,text[:3000]
        assert '724' in text and 'at most 4 fitted lobes' in text and '0.2' in text
    if name=='01_sheets.pdf':assert '5.25' in text and '80' in text and '0.05' in text
    if name=='00_overview.pdf':assert '4 mm' in text  # Supplied-label panel title is rasterized; inspect visually.
Path('/qa/pdf_checks.json').write_text(json.dumps(dict(reports=record,visual_review='required'),indent=2)+'\n')
'''


def scientific(name,module):
    groups=('fixel_maps/','voxel_maps/') if module!='sheets' else ('diagnostics/','tensors/','display/','legacy_SPI/')
    return name.endswith(('.nii','.nii.gz')) and (name.startswith(groups) or name in ('q.nii.gz','wm_mask.nii.gz'))


class Runner:
    def __init__(self,args):
        self.image,self.fixture,self.reference,self.out,self.scratch=(Path(getattr(args,k)).expanduser().resolve()
            for k in ('image','fixture','reference','out','scratch'))
        for path in (self.out,self.scratch):
            writable_path(path)
            if path.exists():raise FileExistsError('Fresh owned destination required: '+str(path))
            for source in (self.image,self.fixture,self.reference):
                if path==source or source in path.parents or path in source.parents:raise ValueError('Input/output overlap')
        if Path('/tmp').resolve() not in self.scratch.parents:raise ValueError('Scratch must be node-local /tmp')
        for path in (self.image,self.fixture/'fod.nii.gz',self.fixture/'brain_mask.nii.gz',self.fixture/'FA.nii.gz',
                     self.reference/'run.json',self.reference/'work/native/fixels/index.nii'):
            if not path.is_file():raise FileNotFoundError(path)
        self.runtime=shutil.which('apptainer') or shutil.which('singularity')
        if not self.runtime:raise FileNotFoundError('Container runtime')
        available=capacity(self.scratch.parent,self.out.parent,THREADS)
        self.out.mkdir();self.scratch.mkdir();(self.scratch/'.options_owned').write_text(str(self.out)+'\n')
        for name in ('logs','comparisons','baselines','cache'):(self.out/name).mkdir()
        self.deadline=time.monotonic()+args.timeout_hours*3600
        self.fixture_hashes=files(self.fixture)
        self.reference_record=json.loads((self.reference/'run.json').read_text())
        self.verify(self.reference,self.reference_record)
        self.record=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),threads=THREADS,
            image=dict(path=str(self.image),sha256=sha(self.image)),reference=str(self.reference),capacity=available,
            source_sha256={name:sha(Path(__file__).with_name(name)) for name in
                ('options_acceptance.py','acceptance.py','diagnose_identity.py','join_reported_identity.py')},
            commands=[],checks=[],tolerances=dict(absolute=0,relative=0,affine=0,dtype='exact',missingness='exact'),
            custom_q=CUSTOM_Q,custom_dispersion=CUSTOM_D,custom_laminarity_scales_mm=[4.],
            scope='Small installed-image software acceptance; original scientific engines unchanged')
        self.save()

    def save(self):write_json(self.out/getattr(self,'record_name','options_acceptance.json'),self.record)

    def check(self,name,passed,details=None):
        self.record['checks'].append(dict(name=name,status='passed' if passed else 'failed',details=details));self.save()
        if not passed:raise ValueError('Check failed: '+name)

    @staticmethod
    def verify(root,record):
        if record['status']!='complete':raise ValueError('Subject is incomplete')
        for name,item in record['public_files'].items():
            p=root/name
            if p.is_symlink() or not p.is_file() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:
                raise ValueError('Published integrity mismatch: '+str(p))

    def container(self,label,arguments,bindings=None,expected_failure=None):
        available=capacity(self.scratch,self.out,THREADS)
        temp=self.scratch/f'command-{len(self.record["commands"])+1}';temp.mkdir()
        bindings=bindings if bindings is not None else [(self.fixture,'/fixture','ro'),
            (self.reference,'/reference','ro'),(self.out,'/evidence','rw')]
        bindings=[*bindings,(self.scratch,'/scratch','rw')]
        argv=['nice','-n','10',self.runtime,'exec','--cleanenv','--containall','--no-home',
              '--no-mount','cwd,hostfs,bind-paths','--pwd','/']
        for source,target,mode in bindings:argv+=['--bind',f'{source}:{target}:{mode}']
        temporary='/scratch/'+temp.name
        for key,value in dict(TMPDIR=temporary,MPLCONFIGDIR=temporary+'/mpl',NUMBA_CACHE_DIR=temporary+'/numba',
                XDG_CACHE_HOME=temporary+'/cache',MRTRIX_TMPFILE_DIR=temporary,PYTHONDONTWRITEBYTECODE='1',
                NUMBA_NUM_THREADS=THREADS,ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=THREADS,
                OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,OMP_NUM_THREADS=1,ANTS_RANDOM_SEED=1).items():
            argv+=['--env',f'{key}={value}']
        argv += [str(self.image),*map(str,arguments)]
        env={k:v for k,v in os.environ.items() if not k.startswith(('APPTAINER','SINGULARITY'))}
        env.update(APPTAINER_CACHEDIR=str(self.out/'cache'),SINGULARITY_CACHEDIR=str(self.out/'cache'),
                   APPTAINER_TMPDIR=str(temp),SINGULARITY_TMPDIR=str(temp),TMPDIR=str(temp))
        log=self.out/'logs'/f'{len(self.record["commands"])+1:02d}-{label}.log'
        item=dict(label=label,argv=argv,log=str(log.relative_to(self.out)),capacity=available,expected_failure=expected_failure)
        self.record['commands'].append(item);self.save();start=time.monotonic()
        with log.open('w') as stream:
            process=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            try:code=process.wait(timeout=max(1,min(3600,self.deadline-time.monotonic())))
            except BaseException:stop(process);raise
        text=log.read_text(errors='replace');passed=(code!=0 and expected_failure in text) if expected_failure else code==0
        item.update(returncode=code,seconds=time.monotonic()-start,status='passed' if passed else 'failed');self.save()
        if not passed:raise RuntimeError(label+' failed: '+str(log)+'\n'+text[-3000:])

    def common(self,name,modules=None,jhu=False):
        command=['geometric-organization','--fod','/fixture/fod.nii.gz','--brain-mask','/fixture/brain_mask.nii.gz',
                 '--fa','/fixture/FA.nii.gz','--threads',THREADS,'--subject','acceptance','--out','/evidence/'+name]
        if modules:command+=['--modules',modules]
        if modules in (None,'sheets'):command+=['--wm-mask','/fixture/brain_mask.nii.gz']
        return command+(['--jhu-labels','/evidence/inputs/JHU_labels.nii.gz'] if jhu else ['--no-jhu'])

    def maps(self,root,prefix,module,standalone=False):
        if standalone:return {p:prefix+'/'+p for p in files(root) if scientific(p,module)}
        layout=json.loads((root/'provenance/layout.json').read_text())[module]
        return {old:prefix+'/'+new for old,new in layout.items() if scientific(old,module)}

    def compare(self,label,left,right,module,standalone=False,allow_scoped_ids=False):
        a=self.maps(self.out/left,'/evidence/'+left,module)
        b=self.maps(self.reference if right=='reference' else self.out/right,
                    '/reference' if right=='reference' else '/evidence/'+right,module,standalone)
        self.check(label+' map set',set(a)==set(b))
        job=self.out/'comparisons'/(label+'-input.json');report=self.out/'comparisons'/(label+'.json')
        write_json(job,dict(tolerances=self.record['tolerances'],pairs=[dict(module=module,relative=name,left=a[name],right=b[name]) for name in sorted(a)]))
        self.container(label,['geo-python','-c',CHECK_IMAGES,'/evidence/'+str(job.relative_to(self.out)),
                              '/evidence/'+str(report.relative_to(self.out))])
        record=json.loads(report.read_text())
        unexpected=[row['relative'] for row in record['comparisons'] if not row['exact'] and
                    not (allow_scoped_ids and row['relative']=='fixel_maps/native_fixel_id.nii.gz')]
        self.check(label+' exact science',not unexpected,dict(strict_report=str(report.relative_to(self.out)),
            strict_status=record['status'],unmatched=unexpected,identity_adjudication='separate complete native proof' if allow_scoped_ids else None))

    def compare_csv(self,left,right,module,standalone=False):
        a=self.out/left;b=self.reference if right=='reference' else self.out/right
        layout=json.loads((a/'provenance/layout.json').read_text())[module]
        other={} if standalone else json.loads((b/'provenance/layout.json').read_text())[module]
        for name,path in layout.items():
            if name in ('summary.csv','summary_eroded.csv','regions/JHU_region_means.csv','regions/JHU_scale_curves.csv'):
                target=b/(name if standalone else other[name])
                self.check(left+' '+module+' '+name+' exact CSV',(a/path).read_bytes()==target.read_bytes())

    def identity_reference(self):
        source=Path(__file__).with_name('diagnose_identity.py').read_text()
        self.container('default-all-native-proof',['geo-python','-c',source,'--native-left','/evidence/default/work/native/fixels',
            '--native-right','/reference/work/native/fixels','--out','/evidence/default_native_proof'])
        # Read-only views only arrange the old published ID image for the verified helper.
        view_code="from pathlib import Path; import json; m=json.loads(Path('/reference/provenance/layout.json').read_text()); "
        view_code+="[(lambda p,q:(p.parent.mkdir(parents=True,exist_ok=True),p.symlink_to(q)))(Path('/scratch/reference-view')/k/'fixel_maps/native_fixel_id.nii.gz',Path('/reference')/v['fixel_maps/native_fixel_id.nii.gz']) for k,v in m.items() if k in ('laminarity','dispersion')]"
        self.container('reference-views',['geo-python','-c',view_code])
        helper=Path(__file__).with_name('join_reported_identity.py').read_text()
        for module in ('laminarity','dispersion'):
            destination=f'/evidence/default_{module}_identity_join.json'
            self.container(module+'-reported-identity',['geo-python','-c',helper,'--crosswalk',
                '/evidence/default_native_proof/all_native_rows.npz','--integrated','/evidence/default',
                '--baseline','/scratch/reference-view/'+module,'--module',module,'--out',destination])
            self.check(module+' mapped IDs exact',json.loads((self.out/Path(destination).name).read_text())['status']=='passed')

    def capture_custom(self):
        destination=self.out/'custom_fixture';destination.mkdir();record={}
        work=self.out/'custom/work'
        layout=json.loads((self.out/'custom/provenance/layout.json').read_text())
        engine_record=self.out/'custom'/layout['sheets']['run.json']
        saved=json.loads(engine_record.read_text())['adapter']['shared_inputs']
        for relative,name,key in [('prepared/reconstruction/WM_FOD.nii.gz','fod.nii.gz','fod'),
            ('prepared/anatomy/brain_mask.nii.gz','brain_mask.nii.gz','brain_mask'),('prepared/anatomy/FA.nii.gz','FA.nii.gz','fa'),
            ('wm/wm_mask.nii.gz','wm_mask.nii.gz','wm_mask')]:
            expected=dict(bytes=(work/relative).stat().st_size,sha256=saved[key]['sha256'])
            record[name]=copy_verified(work/relative,destination/name,expected)
            record[name]['binding']='Saved Sheets adapter shared_inputs SHA256; completed stage source and captured bytes agree'
            record[name]['sheets_record_sha256']=sha(engine_record)
            (destination/name).chmod(0o444)
        write_json(destination/'capture.json',record);(destination/'capture.json').chmod(0o444);destination.chmod(0o555)
        self.record['custom_ready']=dict(subject=str(self.out/'custom'),fixture=str(destination));self.save()
        print('CUSTOM_READY '+json.dumps(self.record['custom_ready']),flush=True)

    def custom_baselines(self):
        base='/evidence/custom/work';native=base+'/native';prepared=base+'/prepared';labels='/evidence/custom/shared/regions/JHU_labels_native.nii.gz'
        make_config="from pathlib import Path; import json; p=Path('/opt/laminarity/transverse/final_config.json'); c=json.loads(p.read_text()); c['scales']=[4.]; Path('/evidence/direct_laminarity_config.json').write_text(json.dumps(c,indent=2)+'\\n')"
        self.container('direct-L-config',['geo-python','-c',make_config])
        self.container('direct-L-compute',['laminarity-python','/opt/laminarity/transverse/pipeline.py','--input',native,
            '--out','/scratch/direct-L-analysis','--config','/evidence/direct_laminarity_config.json','--workers',THREADS,'--subject','acceptance'])
        self.container('direct-L-export',['laminarity-python','/opt/laminarity/transverse/save_outputs.py','--input','/scratch/direct-L-analysis',
            '--prepared-input',native,'--out','/evidence/baselines/laminarity'])
        self.container('direct-L-regions',['laminarity-python','/opt/laminarity/transverse/regions.py','--input','/scratch/direct-L-analysis',
            '--fa',prepared+'/anatomy/FA.nii.gz','--brain-mask',prepared+'/anatomy/brain_mask.nii.gz','--out','/evidence/baselines/laminarity/regions',
            '--atlas-dir','/opt/laminarity/transverse/atlas','--native-labels',labels,'--subject','acceptance','--threads',THREADS])
        command=['dispersion-python','/opt/dispersion/pipeline/compute.py','--fod',prepared+'/reconstruction/WM_FOD.nii.gz',
            '--brain-mask',prepared+'/anatomy/brain_mask.nii.gz','--native-fixels',native+'/fixels',
            '--out','/evidence/baselines/dispersion','--threads',THREADS,'--save-parameters']
        for name,value in CUSTOM_D.items():command+=['--'+name.replace('_','-'),value]
        self.container('direct-D-compute',command)
        code="from pathlib import Path; import shutil,sys; sys.path.insert(0,'/opt/dispersion/pipeline'); from regions import run; r=Path('/evidence/baselines/dispersion'); (r/'anatomy').mkdir(); [shutil.copy2(Path('/evidence/custom/work/prepared/anatomy')/n,r/'anatomy'/n) for n in ('FA.nii.gz','brain_mask.nii.gz')]; run(r,Path('/opt/dispersion/pipeline/atlas'),2,'acceptance',Path('/evidence/custom/shared/regions/JHU_labels_native.nii.gz'))"
        self.container('direct-D-regions',['dispersion-python','-c',code])
        for module in ('laminarity','dispersion'):
            self.compare('custom-'+module,'custom','baselines/'+module,module,True)
            self.compare_csv('custom','baselines/'+module,module,True)

    def portable_qa(self):
        original=self.out/'custom';record=json.loads((original/'run.json').read_text());portable=self.out/'portable';portable.mkdir()
        for name in [*record['public_files'],'run.json']:
            target=portable/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original/name,target)
        before=files(portable);qa=self.out/'portable_qa';qa.mkdir()
        bindings=[(portable,'/subject','ro'),(qa,'/qa','rw')]
        self.container('portable-custom-QA',['geometric-organization','qa','/subject','--out','/qa/reports'],bindings=bindings)
        self.container('portable-custom-QA-settings',['geo-python','-c',CHECK_CUSTOM_QA],bindings=bindings)
        self.check('portable QA input untouched',before==files(portable))

    def remaining(self):
        custom=self.common('custom',jhu=True)+['--keep-work','--save-fod','--no-report','--laminarity-fixed-scale-mm',4]
        for name,value in CUSTOM_Q.items():custom += ['--sheets-'+name.replace('_','-'),value]
        for name,value in CUSTOM_D.items():custom += ['--dispersion-'+name.replace('_','-'),value]
        state=json.loads((self.out/'custom/run.json').read_text());self.verify(self.out/'custom',state)
        self.capture_custom()
        self.container('custom-fixed-setting',['geo-python','-c',CHECK_SETTINGS,'/evidence/custom','[4.0]'])
        before=state['public_files'];self.container('custom-resume',custom+['--resume'])
        self.check('custom resume preserves outputs',before==json.loads((self.out/'custom/run.json').read_text())['public_files'])
        changed=list(custom);changed[changed.index('--laminarity-fixed-scale-mm')+1]=3
        self.container('changed-custom-resume',changed+['--resume'],expected_failure='Inputs, settings, or source changed')
        self.custom_baselines()
        for name,options,scales in [('minimum3',['--laminarity-min-scale-mm',3],[3.,4.,6.,8.]),
                ('unsorted_list',['--laminarity-scales-mm',8,3,4],[3.,4.,8.])]:
            self.container(name,self.common(name,'laminarity')+['--no-report',*options])
            self.container(name+'-settings',['geo-python','-c',CHECK_SETTINGS,'/evidence/'+name,json.dumps(scales)])
        self.portable_qa()
        self.dwi_checks()

    def dwi_checks(self):
        self.container('real-DWI-1mm',['geometric-organization','--dwi','/evidence/inputs/dwi.nii.gz',
            '--bval','/evidence/inputs/dwi.bval','--bvec','/evidence/inputs/dwi.bvec','--brain-mask','/evidence/inputs/dwi_mask.nii.gz',
            '--wm-mask','/evidence/inputs/dwi_mask.nii.gz','--spacing',1,'--csd-mode','sscsd','--no-jhu','--no-report',
            '--sheets-draws',8,'--sheets-radius-mm',3,'--laminarity-fixed-scale-mm',2,'--threads',THREADS,
            '--subject','dwi_1mm','--out','/evidence/dwi_1mm'])
        self.container('real-DWI-1mm-grid',['geo-python','-c',CHECK_ONE_MM])
        self.container('real-DWI-1mm-coverage',['geo-python','-c',CHECK_COVERAGE,'/evidence/dwi_1mm','/evidence/dwi_1mm_coverage.json'])
        self.check('real DWI 1mm finite q L T ODI',json.loads((self.out/'dwi_1mm_coverage.json').read_text())['status']=='passed')
        self.check('fixture unchanged',self.fixture_hashes==files(self.fixture));self.verify(self.reference,self.reference_record)
        self.record.update(status='passed',visual_review='Generated custom reports require human visual review; custom Sheets parity reported separately')

    def run(self):
        try:
            self.container('installed-runtime',['geo-python','/opt/geometric-organization/container/check_runtime.py'])
            self.container('bounded-fixture',['geo-python','-c',CHECK_FIXTURE,'/fixture/fod.nii.gz',
                '/fixture/brain_mask.nii.gz','/fixture/FA.nii.gz','/evidence/fixture.json'])
            self.container('make-small-inputs',['geo-python','-c',MAKE_INPUTS])
            self.container('all-default',self.common('default')+['--keep-work'])
            self.verify(self.out/'default',json.loads((self.out/'default/run.json').read_text()))
            for module in ('sheets','laminarity','dispersion'):
                self.compare('default-'+module,'default','reference',module,allow_scoped_ids=module!='sheets')
                self.compare_csv('default','reference',module)
            self.identity_reference()
            custom=self.common('custom',jhu=True)+['--keep-work','--save-fod','--no-report','--laminarity-fixed-scale-mm',4]
            for name,value in CUSTOM_Q.items():custom+=['--sheets-'+name.replace('_','-'),value]
            for name,value in CUSTOM_D.items():custom+=['--dispersion-'+name.replace('_','-'),value]
            self.container('all-custom',custom)
            state=json.loads((self.out/'custom/run.json').read_text());self.verify(self.out/'custom',state)
            self.check('no-report suppresses public PDFs',not list((self.out/'custom/QA').glob('*.pdf')))
            registration=json.loads((self.out/'custom/shared/regions/registration.json').read_text())
            self.check('supplied JHU skips registration',registration.get('registration_performed') is False and
                not (self.out/'custom/shared/regions/JHU_template_FA_native.nii.gz').exists(),registration)
            self.remaining()
        except BaseException as error:
            self.record.update(status='failed',error=type(error).__name__+': '+str(error));raise
        finally:
            self.record['completed_utc']=datetime.now(timezone.utc).isoformat();self.save()
        return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('image','fixture','reference','out','scratch'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--timeout-hours',type=float,default=2.)
    continuation=p.add_mutually_exclusive_group()
    continuation.add_argument('--continue-after-capture-error',action='store_true',help='Preserve the failed capture harness record and continue its completed custom run')
    continuation.add_argument('--continue-after-qa-text-error',action='store_true',help='Preserve the raster-title text-check failure and continue without regenerating PDFs')
    args=p.parse_args()
    if not (args.continue_after_capture_error or args.continue_after_qa_text_error):return Runner(args).run()
    runner=Runner.__new__(Runner)
    for name in ('image','fixture','reference','out','scratch'):
        setattr(runner,name,getattr(args,name).expanduser().resolve())
    prior=runner.out/('options_acceptance_continuation.json' if args.continue_after_qa_text_error else 'options_acceptance.json');record=json.loads(prior.read_text())
    known_failure=(record.get('error')=="KeyError: 'prepared/reconstruction/WM_FOD.nii.gz'") if args.continue_after_capture_error else (
        record.get('commands',[])[-1].get('label')=='portable-custom-QA-settings' and 'line 17' in record.get('error',''))
    if record.get('status')!='failed' or not known_failure:
        raise ValueError('Continuation is limited to the recorded harness-only failure')
    if (runner.scratch/'.options_owned').read_text()!=str(runner.out)+'\n':raise ValueError('Scratch ownership mismatch')
    runner.record_name='options_acceptance_remaining.json' if args.continue_after_qa_text_error else 'options_acceptance_continuation.json'
    if (runner.out/runner.record_name).exists():raise FileExistsError('Continuation record already exists')
    if sha(runner.image)!=record['image']['sha256']:raise ValueError('Installed image changed')
    if args.continue_after_capture_error:
        empty=runner.out/'custom_fixture'
        if not empty.is_dir() or any(empty.iterdir()):raise ValueError('Expected only the empty failed capture directory')
        empty.rename(runner.out/'custom_fixture_capture_failure')
    runner.reference_record=json.loads((runner.reference/'run.json').read_text());runner.verify(runner.reference,runner.reference_record)
    runner.fixture_hashes=files(runner.fixture)
    default=json.loads((runner.out/'default/run.json').read_text());runner.verify(runner.out/'default',default)
    for item in default['recipe']['inputs'].values():
        source=runner.fixture/Path(item['path']).name
        if sha(source)!=item['sha256']:raise ValueError('Fixture changed since the passed default run')
    runner.runtime=shutil.which('apptainer') or shutil.which('singularity')
    elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(record['started_utc'])).total_seconds()
    runner.deadline=time.monotonic()+max(1,args.timeout_hours*3600-elapsed)
    runner.record=record.copy();runner.record.pop('error',None);runner.record.pop('completed_utc',None)
    runner.record.update(status='running',continuation_of=dict(path=prior.name,sha256=sha(prior)),
                        continuation_source_sha256=sha(Path(__file__)),continuation_reason=(
                            'Overview supplied-label title is a raster; text-check numerical settings and visually review panel' if args.continue_after_qa_text_error else
                            'Capture from saved adapter input hashes; finalized stage inventories are intentionally removed'))
    runner.save()
    try:
        if args.continue_after_qa_text_error:
            portable=runner.out/'portable';before=files(portable)
            runner.verify(portable,json.loads((portable/'run.json').read_text()))
            runner.container('portable-custom-QA-settings-corrected',['geo-python','-c',CHECK_CUSTOM_QA],
                bindings=[(portable,'/subject','ro'),(runner.out/'portable_qa','/qa','rw')])
            runner.check('portable QA input untouched',before==files(portable))
            runner.dwi_checks()
        else:runner.remaining()
    except BaseException as error:
        runner.record.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:
        runner.record['completed_utc']=datetime.now(timezone.utc).isoformat();runner.save()
    return 0


if __name__=='__main__':raise SystemExit(main())

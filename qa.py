"""Compact integration overview plus the unchanged module QA reports."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import tempfile


def regenerate(root, out, command=None):
    """Rebuild QA from a verified portable subject, with no FOD or old work."""
    import publish
    root, out = Path(root).expanduser().resolve(), Path(out).expanduser().resolve()
    if out.exists() or out == root or root in out.parents or out in root.parents:
        raise ValueError('Use a fresh QA destination outside the input subject')
    if out == Path('/nfs2') or Path('/nfs2') in out.parents:
        raise ValueError('/nfs2 is read-only')
    run = read_json(root/'run.json')
    if run.get('status') != 'complete' or not run.get('public_files'):
        raise ValueError('QA regeneration requires a completed subject with saved-file hashes')
    records = run['public_files']
    for name in records:
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Invalid published file reference')
    publish.verify(root, records)
    layout = read_json(root/'provenance/layout.json')
    if not layout or set(layout)-{'sheets','laminarity','dispersion'}:
        raise ValueError('Missing or invalid compact module layout')
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.qa-regeneration-', dir=out.parent))
    log = temporary/'commands.log'

    def execute(cmd, cwd=None, env=None):
        if command is not None: return command(cmd, cwd=cwd, env=env)
        cmd = list(map(str,cmd))
        with log.open('a') as stream:
            stream.write('\n'+shlex.join(cmd)+'\n'); stream.flush()
            result = subprocess.run(cmd, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(f'Report command failed ({result.returncode}); see {log}')

    try:
        modules = {name: publish.restore(root,name,temporary/'modules'/name) for name in layout}
        prepared = temporary/'prepared'; prepared.mkdir()
        (prepared/'anatomy').symlink_to(root/'shared/anatomy', target_is_directory=True)
        (prepared/'reconstruction').mkdir()
        (prepared/'reconstruction/reconstruction.json').symlink_to(root/'provenance/reconstruction/reconstruction.json')
        regions = root/'shared/regions'
        paths = build_reports(modules,prepared,regions if regions.is_dir() else None,
                              out,run.get('subject',root.name),run,execute)
        publish.verify(root,records)
        shutil.rmtree(temporary)
        return paths
    except Exception as error:
        (temporary/'failure.json').write_text(json.dumps(dict(error=str(error),input=str(root),output=str(out)),indent=2)+'\n')
        raise RuntimeError(f'QA regeneration failed; evidence retained in {temporary}: {error}') from error


def read_json(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else {}


def executable(module):
    return os.environ.get('GEO_' + module.upper() + '_PYTHON', module + '-python')


def engine_source(module):
    fallback = '/opt/laminarity/transverse' if module == 'laminarity' else '/opt/dispersion/pipeline'
    return Path(os.environ.get('GEO_' + module.upper() + '_SOURCE', fallback))


def guide_path(module, relative, shared_identity):
    """Translate display paths with the same publication rules as publish.build."""
    import publish
    path=Path(relative); group=path.parts[0]
    if group=='anatomy': return str(Path('shared/anatomy')/path.name)
    if group=='fixel_maps' and path.name in publish.IDENTITY and shared_identity:
        return str(Path('shared/fixels')/path.name)
    if group=='regions' and path.name in publish.REGIONS:
        return str(Path('shared/regions')/path.name)
    if group in ('scales','mrtrix_fixels','parameters'):
        return str(Path('extras')/module/path)
    if group=='QA': return 'QA/'+{'laminarity':'02_lamellarity.pdf','dispersion':'03_dispersion.pdf'}[module]
    if group=='provenance' or path.name in ('manifest.json','output_manifest.json','run.json'):
        return str(Path('provenance/modules')/module/path)
    return str(Path(module)/path)


def dispersion_path_wrapper(stage, root, shared_identity, regional):
    """Patch only frozen report text at render time; fail on guide drift."""
    identity='shared/fixels' if shared_identity else 'dispersion/fixel_maps'
    replacements={
        'fixel_maps/odi_total.nii.gz<br/>(2/pi) atan[1/sqrt(kb kn)]':
            'dispersion/fixel_maps/odi_total.nii.gz<br/>(2/pi) atan[1/sqrt(kb kn)]',
        'fixel_maps/dispersion_anisotropy.nii.gz<br/>(2/pi) atan[(kn-kb)/kb]':
            'dispersion/fixel_maps/dispersion_anisotropy.nii.gz<br/>(2/pi) atan[(kn-kb)/kb]',
        'fixel_maps/hwhm_{broad,narrow}_deg.nii.gz<br/>fitted_peak; broad_axis; narrow_axis':
            'dispersion/fixel_maps/<br/>hwhm_broad_deg.nii.gz; hwhm_narrow_deg.nii.gz<br/>fitted_peak.nii.gz; broad_axis.nii.gz; narrow_axis.nii.gz',
        'fixel_maps/fiber_spread_FS.nii.gz<br/>Integral(F)/peak amplitude':
            'dispersion/fixel_maps/fiber_spread_FS.nii.gz<br/>Integral(F)/peak amplitude',
        'voxel_maps/{metric}_{afd_weighted_mean,max}.nii.gz<br/>valid_afd_fraction.nii.gz':
            'dispersion/voxel_maps/<br/>odi_total_afd_weighted_mean.nii.gz; odi_total_max.nii.gz<br/>dispersion_anisotropy_afd_weighted_mean.nii.gz; dispersion_anisotropy_max.nii.gz<br/>fiber_spread_FS_afd_weighted_mean.nii.gz; fiber_spread_FS_max.nii.gz<br/>valid_afd_fraction.nii.gz',
        'fixel_maps/native_fixel_id; afd; fiber_direction<br/>metric_valid; width_valid; axis_identifiable':
            identity+'/<br/>native_fixel_id.nii.gz; afd.nii.gz; fiber_direction.nii.gz<br/>dispersion/fixel_maps/<br/>metric_valid.nii.gz; width_valid.nii.gz; axis_identifiable.nii.gz',
        'anatomy/FA; brain_mask<br/>regions/JHU_labels_native; JHU_labels_eroded_1voxel<br/>regions/JHU_region_means.csv':
            'shared/anatomy/FA.nii.gz<br/>shared/anatomy/brain_mask.nii.gz'+
            ('<br/>shared/regions/JHU_labels_native.nii.gz<br/>shared/regions/JHU_labels_eroded_1voxel.nii.gz<br/>dispersion/regions/JHU_region_means.csv' if regional else '<br/>JHU not requested')}
    spec=stage/'published_guide.json'; spec.write_text(json.dumps(replacements,indent=2)+'\n')
    wrapper=stage/'render_dispersion.py'
    wrapper.write_text('''import json, sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import qa_report as frozen
replacements=json.loads(Path(sys.argv[2]).read_text())
seen=dict(guide=0,page=0,footer=0,regions=0)
original_start,original_table,original_finish=frozen.Report.start,frozen.Report.table,frozen.Report.finish_page
original_load=frozen.load_compact
def load_compact(root):
    values=list(original_load(root))
    for record in values[-2:]:
        for name in ('settings','options'):
            settings=record.get(name,{})
            if 'min_sep' in settings: settings.setdefault('min_separation',settings['min_sep'])
            if 'relative_peak' in settings: settings.setdefault('peak_threshold',settings['relative_peak'])
    return tuple(values)
frozen.load_compact=load_compact
def start(self,title,height=612):
    if title=='Dispersion | output guide and distinct meanings':
        seen['page']+=1; height=475
    return original_start(self,title,height)
def table(self,rows,widths,top,font=9,padding=5):
    if rows and tuple(rows[0])==('<b>Measure</b>','<b>Saved paths / definition</b>','<b>Interpretation</b>'):
        seen['guide']+=1; rows=[list(row) for row in rows]
        for old,new in replacements.items():
            matches=[(i,j) for i,row in enumerate(rows) for j,cell in enumerate(row) if cell==old]
            if len(matches)!=1: raise ValueError('Frozen Dispersion guide changed: '+old)
            i,j=matches[0]; rows[i][j]=new
        top+=119
    return original_table(self,rows,widths,top,font,padding)
def finish(self,text):
    old='Paths are relative to this output folder; .nii.gz omitted where needed for space. Published Bingham indices do not remove fitting limitations.'
    if text==old:
        seen['footer']+=1
        text='Paths are relative to the subject folder; directory headings apply to the filenames listed below them. Published Bingham indices retain their fitting limitations.'
    if 'complete statistics: regions/' in text:
        seen['regions']+=1
        text=text.replace('complete statistics: regions/','complete statistics: dispersion/regions/')
    return original_finish(self,text)
frozen.Report.start,frozen.Report.table,frozen.Report.finish_page=start,table,finish
sys.argv=[str(Path(sys.argv[1])/'qa_report.py'),*sys.argv[3:]]
frozen.main()
if seen!={'guide':1,'page':1,'footer':1,'regions':__REGIONAL__}: raise ValueError('Frozen Dispersion output-guide hooks changed: '+str(seen))
'''.replace('__REGIONAL__',str(int(regional))))
    return [executable('dispersion'),wrapper,root,spec]


def build_reports(module_dirs, prepared, regions, out, subject, run_record, command):
    """Read compact module outputs; write fresh PDFs/JSON, never change sources.

    command accepts an argv list and optional cwd/env keyword arguments. The
    caller records the artifact-operation marker before the first PDF build.
    Failed staging is deliberately retained; only successful staging is removed.
    """
    prepared, out = Path(prepared).resolve(), Path(out).resolve()
    module_dirs = {key: Path(value).resolve() for key, value in module_dirs.items()}
    regions = Path(regions).resolve() if regions else None
    if out.exists(): raise FileExistsError(f'Use a fresh QA destination: {out}')
    if not subject or len(subject) > 100:
        raise ValueError('Use a nonempty subject label of at most 100 characters')
    if not module_dirs or set(module_dirs) - {'sheets', 'laminarity', 'dispersion'}:
        raise ValueError('Select at least one recognized module for QA')
    for source in [prepared, *module_dirs.values()] + ([regions] if regions else []):
        if out == source or source in out.parents or out in source.parents:
            raise ValueError('QA destination must be separate from source results')
    out.mkdir(parents=True)
    staging = out/'.staging'; staging.mkdir()
    paths = {}
    import publish
    shared_identity=not {'laminarity','dispersion'}.issubset(module_dirs) or all(
        publish.equal_image(module_dirs['laminarity']/'fixel_maps'/name,module_dirs['dispersion']/'fixel_maps'/name)
        for name in publish.IDENTITY)
    names = {'sheets': '01_sheets.pdf', 'laminarity': '02_lamellarity.pdf', 'dispersion': '03_dispersion.pdf'}
    for name, folder in module_dirs.items():
        stage = staging/name; stage.mkdir()
        pdf = stage/names[name]
        env = os.environ.copy()
        env.update(MPLCONFIGDIR=str(stage/'mpl'), NUMBA_CACHE_DIR=str(stage/'numba'),
                   XDG_CACHE_HOME=str(stage/'cache'), TMPDIR=str(stage), MPLBACKEND='Agg',
                   PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        if name == 'sheets':
            cmd = [executable(name), Path(__file__).with_name('sheets_module.py'),
                   '--qa-from', folder, '--qa-out', pdf, '--published-paths']
        else:
            root = engine_source(name)
            cmd = [executable(name), root/'qa_report.py', '--compact-output', folder,
                   '--out', pdf, '--subject', subject]
            if name=='laminarity':
                manifest=read_json(folder/'output_manifest.json')
                for row in manifest['guide']:
                    row['paths']=[guide_path(name,path,shared_identity) for path in row['paths']]
                guide=stage/'published_output_manifest.json'; guide.write_text(json.dumps(manifest,indent=2)+'\n')
                cmd+=['--output-manifest',guide]
            else:
                cmd=dispersion_path_wrapper(stage,root,shared_identity,(folder/'regions/JHU_labels_native.nii.gz').exists())+cmd[2:]
            if name == 'laminarity' and (folder/'regions/JHU_labels_native.nii.gz').is_file():
                overlay = stage/'JHU_QA.png'
                code = ('import sys; import nibabel as n; import numpy as np; '
                        'sys.path.insert(0,sys.argv[1]); from regions import qa_overlay; '
                        'qa_overlay(n.load(sys.argv[2]),np.asarray(n.load(sys.argv[3]).dataobj),'
                        'np.asarray(n.load(sys.argv[4]).dataobj),sys.argv[5],sys.argv[6] or None)')
                template = folder/'regions/JHU_template_FA_native.nii.gz'
                if not template.exists(): template = folder/'regions/registration/template_in_native_FA.nii.gz'
                command([executable(name), '-c', code, root, folder/'anatomy/FA.nii.gz',
                         folder/'regions/JHU_labels_native.nii.gz', folder/'regions/JHU_labels_eroded_1voxel.nii.gz',
                         overlay, template if template.exists() else ''], cwd=stage, env=env)
                cmd += ['--regions-qa', overlay]
        command(cmd, cwd=stage, env=env)
        shutil.copy2(pdf, out/pdf.name)
        paths[name] = out/pdf.name
        for note in stage.glob('*.json'):
            if not note.name.startswith('published_'): shutil.copy2(note, out/(name + '_' + note.name))
    paths['overview'] = out/'00_overview.pdf'
    overview(module_dirs, prepared, regions, paths['overview'], subject, run_record, shared_identity)
    from pypdf import PdfReader
    records = {}
    for name, pdf in paths.items():
        pages = len(PdfReader(pdf).pages)
        if pages < 1 or (name == 'overview' and pages != 2):
            raise ValueError(f'Unexpected report page count: {pdf}')
        records[name] = dict(path=pdf.name, pages=pages, bytes=pdf.stat().st_size,
                             sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
                             published_subject_paths=True,
                             display_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'reports.json').write_text(json.dumps(records, indent=2)+'\n')
    shutil.rmtree(staging)
    return paths


def overview(module_dirs, prepared, regions, pdf, subject, run_record, shared_identity=True):
    """Two dense pages; scalar interpolation is for display only."""
    import nibabel as nib
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from scipy.ndimage import map_coordinates
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph
    from xml.sax.saxutils import escape

    ref = nib.load(prepared/'anatomy/FA.nii.gz')
    supplied_labels = bool(regions and read_json(regions/'registration.json').get('registration_performed') is False)

    def load(path, channel=None):
        if path is None or not Path(path).is_file(): return None
        image = nib.load(path)
        if image.shape[:3] != ref.shape or not np.allclose(image.affine, ref.affine, atol=1e-5, rtol=0):
            raise ValueError(f'Overview input is not on the shared grid: {path}')
        values = np.asarray(image.dataobj, dtype=np.float32)
        if channel is not None: values = values[..., channel]
        if values.ndim != 3: raise ValueError(f'Overview needs a scalar image: {path}')
        return values

    fa = load(prepared/'anatomy/FA.nii.gz')
    mask_path = prepared/'anatomy/brain_mask.nii.gz'
    if not mask_path.is_file(): raise FileNotFoundError(mask_path)
    brain = load(mask_path) > 0
    occupied = np.argwhere(brain)
    if not len(occupied): raise ValueError('Empty shared observation mask')
    world = nib.affines.apply_affine(ref.affine, occupied)
    low, high = world.min(0)-4, world.max(0)+4
    center = world.mean(0); center[0] = np.clip(center[0]+20, low[0]+4, high[0]-4)
    inverse = np.linalg.inv(ref.affine)
    wm = load(module_dirs['sheets']/'wm_mask.nii.gz') if 'sheets' in module_dirs else None
    labels = load(regions/'JHU_labels_native.nii.gz') if regions else None
    template = load(regions/'JHU_template_FA_native.nii.gz') if regions else None
    if 'sheets' in module_dirs and wm is None:
        raise FileNotFoundError(module_dirs['sheets']/'wm_mask.nii.gz')
    if regions and labels is None: raise FileNotFoundError(regions/'JHU_labels_native.nii.gz')
    profile = read_json(prepared/'reconstruction/reconstruction.json')
    spacing = nib.affines.voxel_sizes(ref.affine)
    observations = int(brain.sum())
    metadata = dict(subject=subject, shared_grid=dict(shape=list(ref.shape), affine=ref.affine.tolist(),
                    spacing_mm=spacing.tolist()), display_coordinates_RAS_mm=center.tolist(),
                    interpolation='Nearest metric/mask/labels; linear FA/template; display only',
                    brain_voxels=observations, wm_voxels=int((wm>0).sum()) if wm is not None else None,
                    modules=list(module_dirs), shared_registration=labels is not None, panels={})

    def sample(values, axis, order=0):
        dims = [d for d in range(3) if d != axis]
        scale = 400/max(high[dims]-low[dims])
        shape = np.maximum(2, np.ceil((high[dims]-low[dims])*scale).astype(int))
        x = np.linspace(low[dims[0]], high[dims[0]], shape[0])
        y = np.linspace(low[dims[1]], high[dims[1]], shape[1])
        xx, yy = np.meshgrid(x, y)
        coordinates = np.zeros(xx.shape+(3,)); coordinates[..., axis] = center[axis]
        coordinates[..., dims[0]], coordinates[..., dims[1]] = xx, yy
        ijk = nib.affines.apply_affine(inverse, coordinates)
        return map_coordinates(values.astype(float), ijk.reshape(-1, 3).T, order=order,
                               mode='constant', cval=np.nan, prefilter=False).reshape(xx.shape)

    def panel(ax, axis, title):
        ax.set_facecolor('#34383c')
        ax.imshow(sample(fa, axis, 1), origin='lower', cmap='gray', vmin=0, vmax=.8)
        ax.set_title(title, fontsize=10, loc='left', pad=5)
        left, right = ('P', 'A') if axis == 0 else ('L', 'R')
        for x, text, align in [(0.02,left,'left'),(.98,right,'right')]:
            ax.text(x,.04,text,transform=ax.transAxes,color='white',fontsize=8,ha=align)
        ax.text(.98,.96,'S' if axis != 2 else 'A',transform=ax.transAxes,color='white',fontsize=8,ha='right',va='top')
        ax.set_xticks([]); ax.set_yticks([])

    def contour(ax, values, axis, color):
        image = sample(values, axis)
        if np.any(image > 0) and np.any(np.nan_to_num(image) == 0):
            ax.contour(np.nan_to_num(image), levels=[.5], colors=[color], linewidths=.7)

    page = canvas.Canvas(str(pdf), pagesize=(792,612))
    style = ParagraphStyle('caption', fontName='Helvetica', fontSize=9, leading=12, textColor='#263746')

    def text(value, y, size=9):
        style.fontSize=size; style.leading=size+3
        paragraph=Paragraph(value,style); _,height=paragraph.wrap(756,100)
        paragraph.drawOn(page,18,y-height)

    def header(title, page_number):
        page.setFillColorRGB(.07,.16,.21); page.setFont('Helvetica-Bold',18)
        page.drawString(18,584,title)
        page.setFont('Helvetica',9); page.drawRightString(774,585,f'{subject} | {page_number}/2')

    def draw(fig, x, y, width, height):
        stream=io.BytesIO(); fig.savefig(stream,format='png',dpi=180,facecolor='white',bbox_inches='tight',pad_inches=.06)
        plt.close(fig); stream.seek(0)
        page.drawImage(ImageReader(stream),x,y,width,height,preserveAspectRatio=True,anchor='c',mask='auto')

    header('Geometric organization | shared preparation',1)
    mode=profile.get('mode','shared prepared FOD')
    grid=' x '.join(f'{v:g}' for v in spacing)
    locations=['shared/anatomy/']+(['shared/regions/'] if regions else [])
    if shared_identity and {'laminarity','dispersion'}.intersection(module_dirs): locations.append('shared/fixels/')
    locations += [name+'/' for name in module_dirs]
    text(f'<b>One FOD / one grid:</b> {escape(str(mode))}; {grid} mm; '
         f'{ref.shape[0]} x {ref.shape[1]} x {ref.shape[2]} voxels. '
         '<b>Modules:</b> '+escape(', '.join(module_dirs))+'.<br/>'
         '<b>Saved locations:</b> '+escape(' | '.join(locations)),563)
    fig,axes=plt.subplots(2,3,figsize=(12.5,7.0))
    for axis,name in enumerate(('Sagittal','Coronal','Axial')):
        panel(axes[0,axis],axis,f'{name} | domains on FA')
        contour(axes[0,axis],brain,axis,'#5de4f5')
        if wm is not None: contour(axes[0,axis],wm,axis,'#ffbc42')
        panel(axes[1,axis],axis,'Shared JHU | '+('supplied labels on FA' if supplied_labels else 'atlas-to-FA alignment' if labels is not None else 'not requested'))
        if labels is not None:
            lab=sample(labels,axis); border=np.zeros(lab.shape,bool)
            for dim in (0,1): border|=(lab != np.roll(lab,1,dim)) & (lab>0)
            if template is not None:
                warped=sample(template,axis,1); yy,xx=np.indices(warped.shape)
                checker=np.ma.masked_where((xx//32+yy//32)%2==0,warped)
                axes[1,axis].imshow(checker,origin='lower',cmap='gray',vmin=0,vmax=.8)
            overlay=np.zeros(lab.shape+(4,)); overlay[border]=[1,.74,.26,.9]
            axes[1,axis].imshow(overlay,origin='lower')
        else: axes[1,axis].text(.5,.5,'No registration',transform=axes[1,axis].transAxes,ha='center',color='white')
    fig.subplots_adjust(left=.01,right=.99,top=.97,bottom=.01,wspace=.025,hspace=.12)
    draw(fig,12,105,768,420)
    text(f'<b>Cyan:</b> observed brain ({observations:,} voxels). <b>Gold:</b> Sheets measurement centers '
         f'({metadata["wm_voxels"]:,} voxels).' if wm is not None else
         f'<b>Cyan:</b> observed brain ({observations:,} voxels). No Sheets WM domain is required.',96)
    text('Lamellarity and dispersion use brain observations; the WM mask never clips their calculation. '
         'Bottom row: JHU parcel borders on subject FA, with warped-template FA checkerboard when available. '
         'Inspect anatomical alignment; shared labels do not make module denominators identical.',69)
    text('Physical RAS views; image left is anatomical left in coronal/axial views. '
         'FA window 0-0.8. Resampling standardizes the grid; it does not recover acquired detail.',31,8)
    page.showPage(); header('Geometric organization | complementary measures',2)
    text('Matched axial view. Fixed scales below are display windows, not shared scientific units. '
         'Gray indicates unavailable measurements; genuine zero retains its colormap color.',561)
    specs=[('FA',None,None,None,(0,.8),'gray',fa),
           ('q pair12','sheets','q.nii.gz',0,(0,.1),'viridis_r',None),
           ('Lamellarity L','laminarity','voxel_maps/lamellarity_afd_weighted_mean.nii.gz',None,(0,1),'viridis',None),
           ('Tubularity T','laminarity','voxel_maps/tubularity_afd_weighted_mean.nii.gz',None,(0,1),'viridis',None),
           ('Dispersion ODI','dispersion','voxel_maps/odi_total_afd_weighted_mean.nii.gz',None,(0,.25),'viridis',None),
           ('Dispersion anisotropy DA_B','dispersion','voxel_maps/dispersion_anisotropy_afd_weighted_mean.nii.gz',None,(0,1),'plasma',None)]
    fig,axes=plt.subplots(2,3,figsize=(12.5,7.1))
    for ax,(title,module,relative,channel,limits,cmap,values) in zip(axes.flat,specs):
        panel(ax,2,title)
        if module in module_dirs:
            values=load(module_dirs[module]/relative,channel)
            if values is None: raise FileNotFoundError(module_dirs[module]/relative)
        if values is None:
            ax.text(.5,.5,'Module not selected',transform=ax.transAxes,ha='center',color='white'); continue
        domain=wm>0 if module=='sheets' else brain
        shown=sample(np.where(domain,values,np.nan),2)
        palette=plt.get_cmap(cmap).copy(); palette.set_bad((.47,.49,.5,.65))
        view=ax.imshow(np.ma.masked_invalid(shown),origin='lower',cmap=palette,vmin=limits[0],vmax=limits[1])
        bar=fig.colorbar(view,ax=ax,fraction=.033,pad=.015,shrink=.82,
                        extend='max' if title in ('q pair12','Dispersion ODI') else 'neither')
        bar.ax.tick_params(labelsize=8)
        finite=int(np.sum(np.isfinite(values)&domain)); denominator=int(domain.sum())
        units='mm^-1; lower = more compatible' if module=='sheets' else 'dimensionless'
        ax.set_xlabel(f'{units} | finite {finite:,}/{denominator:,}',fontsize=8)
        metadata['panels'][title]=dict(window=list(limits),finite_voxels=finite,denominator_voxels=denominator,
                                      denominator='Sheets WM centers' if module=='sheets' else 'observed brain')
    fig.subplots_adjust(left=.01,right=.98,top=.97,bottom=.05,wspace=.15,hspace=.22)
    draw(fig,12,92,768,437)
    scale_text='of independently maximized fixel scores over 2.5, 3, 4, 6, 8 mm.'
    if 'laminarity' in module_dirs:
        saved=read_json(module_dirs['laminarity']/'output_manifest.json')
        scales=saved.get('scales_mm',saved.get('configuration',{}).get('scales',[2.5,3,4,6,8]))
        values=', '.join(f'{float(value):g}' for value in scales)
        scale_text=(f'of fixed-scale fixel scores at {values} mm.' if len(scales)==1 else
                    f'of independently maximized fixel scores over {values} mm.')
        metadata['laminarity_scales_mm']=scales
    text('<b>q:</b> fixed primary reference pair12, never a minimum-q winner. <b>L / T:</b> AFD-weighted voxel means '
         +scale_text+' <b>ODI / DA_B:</b> '
         'AFD-weighted voxel means of fitted angular spread and its anisotropy.',80)
    text('Counts above use finite voxel estimates / the stated spatial domain; they are not AFD coverage. '
         'See module reports and saved coverage maps for missing fixels, matched lobes and scale support. '
         'These measures describe different properties; none is a histological classification.',43,8)
    page.save()
    pdf.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')

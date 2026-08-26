"""
=====================================================================
  LIVE DEMO - Fracture-Surface Segmentation with DiffusionNet
  3D Surface Fracture Segmentation Demo - USTH
=====================================================================
Pick a broken object -> the model predicts its fracture surface LIVE ->
an interactive 3D view opens in the browser (drag to rotate, scroll to
zoom). Use the GROUND TRUTH / PREDICTION buttons to compare.

Menu:
  0-5  : pre-loaded objects (instant, operators cached)
  b    : a RANDOM new Breaking Bad object   (computed live, ~10-15s)
  f    : a RANDOM new Fantastic Breaks shard (computed live, ~10s)
  q    : quit

Run:  python demo.py            (interactive menu)
      python demo.py 3          (directly show object 3)
      python demo.py b          (a random Breaking Bad object)
Prerequisite (run once):  python build_demo.py
"""
import sys, io, time, random, zipfile, re
from pathlib import Path
from collections import defaultdict
import numpy as np, torch
import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
DIFFNET_SRC = 'E:/diffusion-net/src'         # <- repo path (change if you move machines)
BB_ZIP  = 'E:/artifact_compressed.zip'
FB_ROOT = Path('E:/Fantastic_Breaks_v1')

sys.path.append(DIFFNET_SRC)
import diffusion_net

DATA = HERE/'data'
OPC  = HERE/'cache'/'op_cache'; OPC.mkdir(parents=True, exist_ok=True)
K_EIG, N_HKS = 128, 16
EXPLODE, THR_RATIO, DECIMATE_VERTS = 0.35, 0.005, 22000
device = torch.device('cpu')

def load_model(pt):
    ck = torch.load(pt, map_location='cpu', weights_only=False)
    m = diffusion_net.layers.DiffusionNet(C_in=20, C_out=2, C_width=128, N_block=4,
                                          outputs_at='vertices', dropout=True)
    m.load_state_dict(ck['model_state_dict']); m.eval(); return m
MODELS = {'bb': load_model(HERE/'model_bb.pt'), 'fb': load_model(HERE/'model_fb.pt')}
SAMPLES = sorted(DATA.glob('*.npz'))

# ---------------- model forward ----------------
def _sl(x):
    x = torch.log(x.clamp_min(1e-8)); return (x - x.mean(0,keepdim=True))/(x.std(0,keepdim=True)+1e-6)

def predict(verts, faces, model_key):
    v = torch.tensor(verts); f = torch.tensor(faces, dtype=torch.long)
    _, mass, L, evals, evecs, gX, gY = diffusion_net.geometry.get_operators(
        v, f, k_eig=K_EIG, op_cache_dir=str(OPC))
    hks = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, N_HKS)
    Hv = torch.sparse.mm(L, v)/mass.clamp_min(1e-8).unsqueeze(-1)
    x = torch.cat([v, _sl(hks), _sl(Hv.norm(dim=-1,keepdim=True))], dim=-1)
    with torch.no_grad():
        out = MODELS[model_key](x_in=x, mass=mass, L=L, evals=evals, evecs=evecs,
                                gradX=gX, gradY=gY, faces=f)
    return out.argmax(-1).numpy()

def iou(p, y):
    tp=((p==1)&(y==1)).sum(); fp=((p==1)&(y==0)).sum(); fn=((p==0)&(y==1)).sum()
    prec=tp/(tp+fp) if (tp+fp) else 0; rec=tp/(tp+fn) if (tp+fn) else 0
    return (tp/(tp+fp+fn) if (tp+fp+fn) else 0), prec, rec

def explode(v, pid):
    v=v.copy(); gc=v.mean(0)
    for i in np.unique(pid):
        mk=pid==i; dvec=v[mk].mean(0)-gc; n=np.linalg.norm(dvec)
        if n>1e-6: v[mk]+=(dvec/n)*EXPLODE
    return v

# ---------------- live samplers (arbitrary objects) ----------------
def _rdf(F):
    Fs=np.sort(F,axis=1); _,idx=np.unique(Fs,axis=0,return_index=True); return F[np.sort(idx)]

_BB_INDEX=None
def _bb_index(z):
    """Build {object: [fracture,...]} once from the zip listing (no .npy reads)."""
    global _BB_INDEX
    if _BB_INDEX is None:
        idx=defaultdict(set)
        for n in z.namelist():
            m=re.match(r'artifact_compressed/([^/]+)/(fractured_\d+)/',n)
            if m: idx[m.group(1)].add(m.group(2))
        _BB_INDEX={o:sorted(f) for o,f in idx.items()}
    return _BB_INDEX

def _frac_pieces(z, obj, frac):
    a=np.load(io.BytesIO(z.read(f'artifact_compressed/{obj}/{frac}/compressed_fracture.npy')))
    return int(a.max()+1)

def _bb_decompress(z, obj, frac):
    """Decompress one (object, fracture) -> sample dict, or None if too large.
    The ACTUAL rendered fragment count is len(unique(pid))."""
    import igl
    from scipy.sparse import load_npz
    from scipy.spatial import cKDTree
    tmp=HERE/'_tmp'; tmp.mkdir(exist_ok=True)
    (tmp/'m.obj').write_bytes(z.read(f'artifact_compressed/{obj}/compressed_mesh.obj'))
    (tmp/'d.npz').write_bytes(z.read(f'artifact_compressed/{obj}/compressed_data.npz'))
    V,Fc=igl.read_triangle_mesh(str(tmp/'m.obj')); M=load_npz(str(tmp/'d.npz'))
    pl=np.load(io.BytesIO(z.read(f'artifact_compressed/{obj}/{frac}/compressed_fracture.npy')))
    vlab=(M@pl).astype(int); tri=vlab[Fc[:,0]]; pieces=[]
    for i in range(int(pl.max()+1)):
        fi=Fc[tri==i]
        if len(fi)==0: continue
        vi,ff=igl.remove_unreferenced(V,fi)[:2]
        ui,I,J,_=igl.remove_duplicate_vertices(vi,ff,1e-10)
        nv,nf=igl.remove_unreferenced(ui,_rdf(J[ff]))[:2]
        pieces.append((np.asarray(nv,np.float32),np.asarray(nf,np.int64)))
    if len(pieces)<2: return None
    allv=np.concatenate([v for v,_ in pieces]); thr=np.linalg.norm(allv.max(0)-allv.min(0))*THR_RATIO
    mv,mf,ml,pid,off=[],[],[],[],0
    for i,(v,f) in enumerate(pieces):
        oth=[pv for j,(pv,_) in enumerate(pieces) if j!=i]
        lab=(cKDTree(np.concatenate(oth)).query(v,k=1)[0]<thr).astype(np.int64) if oth else np.zeros(len(v),int)
        mv.append(v); mf.append(f+off); ml.append(lab); pid.append(np.full(len(v),i)); off+=len(v)
    verts=np.concatenate(mv); faces=np.concatenate(mf).astype(np.int64)
    labels=np.concatenate(ml); pids=np.concatenate(pid)
    if len(verts)>28000: return None          # keep CPU time reasonable
    verts=verts-verts.mean(0); sc=np.linalg.norm(verts.max(0)-verts.min(0)); verts=(verts/sc).astype(np.float32)
    return dict(verts=verts,faces=faces,labels=labels,pid=pids,explode=1,model='bb',
                name=f'Breaking Bad (random) - {obj}/{frac} - {len(pieces)} fragments', npieces=len(pieces))

def random_bb(n_frags=None):
    """Random Breaking Bad object. If n_frags is given, decompress several
    candidates and return the one whose ACTUAL fragment count is closest to it."""
    z=zipfile.ZipFile(BB_ZIP)
    index=_bb_index(z)
    objs=list(index.keys()); random.shuffle(objs)

    if n_frags is None:
        for o in objs:
            d=_bb_decompress(z, o, random.choice(index[o]))
            if d is not None: return d
        raise RuntimeError('no suitable BB object found')

    # shortlist fractures whose label-count is >= n_frags (actual <= label-count),
    # decompress them, and keep the one with the closest ACTUAL fragment count.
    best=None; reads=0; tried=0
    for obj in objs:
        fr=index[obj][:]; random.shuffle(fr)
        for frac in fr:
            if reads>=200 or tried>=12: break
            reads+=1
            nc=_frac_pieces(z, obj, frac)
            if nc < n_frags or nc > n_frags+5:      # actual can only drop below nc
                continue
            tried+=1
            d=_bb_decompress(z, obj, frac)
            if d is None: continue
            if d['npieces']==n_frags: return d      # exact actual match
            if best is None or abs(d['npieces']-n_frags)<abs(best['npieces']-n_frags):
                best=d
        if reads>=200 or tried>=12: break
    if best is not None: return best
    return random_bb(None)                          # fallback: any object

def random_fb():
    import trimesh, fast_simplification
    from scipy.spatial import cKDTree
    dirs=sorted(set(m.parent for m in FB_ROOT.rglob('meta_*.npz')))
    sd=random.choice(dirs)
    mb=sorted(sd.glob('model_b_*.ply'))[0]; mt=sorted(sd.glob('meta_*.npz'))[0]
    m=trimesh.load(str(mb),process=False)
    V0=np.ascontiguousarray(np.asarray(m.vertices,np.float64)); F0=np.ascontiguousarray(np.asarray(m.faces,np.int32))
    mask=np.load(str(mt))['mask'].astype(bool)
    red=max(0.05,min(0.99,1.0-(DECIMATE_VERTS*2)/len(F0)))
    Vd,Fd=fast_simplification.simplify(V0,F0,target_reduction=red); Vd=np.asarray(Vd,np.float32); Fd=np.asarray(Fd,np.int64)
    labels=mask[cKDTree(V0).query(Vd,k=1)[1]].astype(np.int64)
    adj=defaultdict(set)
    for a,b,c in Fd: adj[a].update((b,c)); adj[b].update((a,c)); adj[c].update((a,b))
    grow=set()
    for v in np.where(labels==1)[0]: grow.update(adj[int(v)])
    if grow: labels[list(grow)]=1
    Vd=Vd-Vd.mean(0); sc=np.linalg.norm(Vd.max(0)-Vd.min(0)); Vd=(Vd/sc).astype(np.float32)
    rel=sd.relative_to(FB_ROOT).as_posix()
    return dict(verts=Vd,faces=Fd,labels=labels,pid=np.zeros(len(Vd),int),explode=0,model='fb',
                name=f'Fantastic Breaks (random real shard) - {rel}')

# ---------------- view ----------------
def show(d, pred, io_):
    v=np.asarray(d['verts']); f=np.asarray(d['faces']); gt=np.asarray(d['labels']); pid=np.asarray(d['pid'])
    if int(d['explode']): v=explode(v, pid)
    gt_c=gt.astype(float); pr_c=pred.astype(float)
    base=dict(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],
              colorscale=[[0,'#cfcfcf'],[1,'crimson']],cmin=0,cmax=1,flatshading=True,showscale=False)
    fig=go.Figure(go.Mesh3d(intensity=gt_c, **base))
    fig.update_layout(
        title=f"{str(d['name'])}<br><sub>Prediction IoU={io_[0]:.3f} | Precision={io_[1]:.3f} Recall={io_[2]:.3f} (grey=original, red=fracture)</sub>",
        scene=dict(aspectmode='data'), width=1000, height=750,
        updatemenus=[dict(type='buttons', direction='right', x=0.5, y=1.05, xanchor='center',
            buttons=[dict(label='GROUND TRUTH', method='restyle', args=[{'intensity':[gt_c]}]),
                     dict(label='PREDICTION',   method='restyle', args=[{'intensity':[pr_c]}])])])
    fig.show()

def run_sample(d):
    print(f"\n>> {str(d['name'])}")
    print('   Running DiffusionNet on the mesh ...', flush=True)
    t=time.time(); pred=predict(d['verts'], d['faces'], str(d['model'])); io_=iou(pred, np.asarray(d['labels']))
    print(f"   Done in {time.time()-t:.2f}s  ->  IoU={io_[0]:.3f}  Precision={io_[1]:.3f}  Recall={io_[2]:.3f}")
    print('   Opening interactive 3D view (drag=rotate, scroll=zoom; buttons toggle GT/Prediction)...')
    show(d, pred, io_)

def handle(choice):
    choice=choice.strip().lower()
    if choice.startswith('b'):
        rest=choice[1:].strip()
        if rest=='':
            print('   Picking a random Breaking Bad object, computing operators (~10-15s)...', flush=True)
            run_sample(random_bb())
        elif rest.isdigit():
            n=int(rest)
            print(f'   Looking for a Breaking Bad object broken into {n} fragments, computing operators (~10-15s)...', flush=True)
            run_sample(random_bb(n_frags=n))
        else:
            print('   invalid choice (use b, or b<number> e.g. b5)')
    elif choice=='f':
        print('   Picking a random Fantastic Breaks shard, computing operators (~10s)...', flush=True)
        run_sample(random_fb())
    elif choice.isdigit() and int(choice)<len(SAMPLES):
        run_sample(np.load(SAMPLES[int(choice)], allow_pickle=True))
    else:
        print('   invalid choice')

def menu():
    print('='*62); print('  FRACTURE-SURFACE SEGMENTATION - LIVE DEMO'); print('='*62)
    for i,s in enumerate(SAMPLES):
        print(f'  [{i}] {str(np.load(s, allow_pickle=True)["name"])}')
    print('  [b]  random NEW Breaking Bad object        (live, ~10-15s)')
    print('  [bN] Breaking Bad broken into N fragments  (e.g. b5, b10)')
    print('  [f]  random NEW Fantastic Breaks shard     (live, ~10s)')
    print('  [q]  quit')
    while True:
        c=input('\nSelect: ').strip().lower()
        if c=='q': break
        try: handle(c)
        except Exception as e: print('   error:', e)

if __name__=='__main__':
    if not SAMPLES:
        print('No demo data. Run:  python build_demo.py'); sys.exit(1)
    if len(sys.argv)>1: handle(sys.argv[1])
    else: menu()

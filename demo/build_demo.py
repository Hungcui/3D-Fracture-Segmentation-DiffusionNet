"""Build the demo data package: decompress/preprocess the curated samples,
pre-compute (warm) the spectral operators into cache/op_cache, and save each
sample's mesh + ground-truth labels into data/. Run ONCE before the demo.

After this, demo.py runs the model live (forward pass <1s) with no heavy compute.
"""
import sys, io, zipfile
from pathlib import Path
import numpy as np, torch, igl
from scipy.sparse import load_npz
from scipy.spatial import cKDTree
import fast_simplification

sys.path.append('E:/diffusion-net/src')
import diffusion_net

HERE = Path(__file__).resolve().parent
DATA = HERE/'data'; DATA.mkdir(exist_ok=True)
OPC  = HERE/'cache'/'op_cache'; OPC.mkdir(parents=True, exist_ok=True)
K_EIG=128; THR_RATIO=0.005; DECIMATE_VERTS=22000; DILATE_RINGS=1

BB_ZIP='E:/artifact_compressed.zip'
FB_ROOT=Path('E:/Fantastic_Breaks_v1')

# curated demo samples (validated IoU)
BB_SAMPLES=[('75663_sf','fractured_62'),('75663_sf','fractured_5'),('75663_sf','fractured_65')]
FB_SAMPLES=['18/18004','02/02001','12/12001']

device=torch.device('cpu')

# ---------- BB decompress (in-memory, with full dedup chain) ----------
def _resolve_dup_faces(F):
    Fs=np.sort(F,axis=1); _,idx=np.unique(Fs,axis=0,return_index=True); return F[np.sort(idx)]

def bb_decompress(z, obj, frac):
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
        ui,I,J,_=igl.remove_duplicate_vertices(vi,ff,1e-10); gi=J[ff]
        ffi=_resolve_dup_faces(gi); nv,nf=igl.remove_unreferenced(ui,ffi)[:2]
        pieces.append((np.asarray(nv,np.float32),np.asarray(nf,np.int64)))
    # merge + proximity GT labels + piece ids
    allv=np.concatenate([v for v,_ in pieces]); thr=np.linalg.norm(allv.max(0)-allv.min(0))*THR_RATIO
    mv,mf,ml,pid,off=[],[],[],[],0
    for i,(v,f) in enumerate(pieces):
        oth=[pv for j,(pv,_) in enumerate(pieces) if j!=i]
        lab=(cKDTree(np.concatenate(oth)).query(v,k=1)[0]<thr).astype(np.int64) if oth else np.zeros(len(v),np.int64)
        mv.append(v); mf.append(f+off); ml.append(lab); pid.append(np.full(len(v),i)); off+=len(v)
    verts=np.concatenate(mv); faces=np.concatenate(mf).astype(np.int64)
    labels=np.concatenate(ml); pids=np.concatenate(pid)
    verts=verts-verts.mean(0); sc=np.linalg.norm(verts.max(0)-verts.min(0))
    if sc>0: verts=verts/sc
    return verts.astype(np.float32),faces,labels,pids,len(pieces)

# ---------- FB preprocess ----------
def fb_preprocess(sd):
    sd=FB_ROOT/sd
    mb=sorted(sd.glob('model_b_*.ply')); mt=sorted(sd.glob('meta_*.npz'))
    m=__import__('trimesh').load(str(mb[0]),process=False)
    V0=np.ascontiguousarray(np.asarray(m.vertices,np.float64)); F0=np.ascontiguousarray(np.asarray(m.faces,np.int32))
    mask=np.load(str(mt[0]))['mask'].astype(bool)
    red=max(0.05,min(0.99,1.0-(DECIMATE_VERTS*2)/len(F0)))
    Vd,Fd=fast_simplification.simplify(V0,F0,target_reduction=red)
    Vd=np.asarray(Vd,np.float32); Fd=np.asarray(Fd,np.int64)
    labels=mask[cKDTree(V0).query(Vd,k=1)[1]].astype(np.int64)
    from collections import defaultdict
    adj=defaultdict(set)
    for a,b,c in Fd: adj[a].update((b,c)); adj[b].update((a,c)); adj[c].update((a,b))
    for _ in range(DILATE_RINGS):
        grow=set()
        for v in np.where(labels==1)[0]: grow.update(adj[int(v)])
        if grow: labels[list(grow)]=1
    Vd=Vd-Vd.mean(0); sc=np.linalg.norm(Vd.max(0)-Vd.min(0))
    if sc>0: Vd=Vd/sc
    return Vd.astype(np.float32),Fd,labels,np.zeros(len(Vd),int),1

def warm(verts,faces):
    v=torch.tensor(verts); f=torch.tensor(faces,dtype=torch.long)
    diffusion_net.geometry.get_operators(v,f,k_eig=K_EIG,op_cache_dir=str(OPC))

idx=0
z=zipfile.ZipFile(BB_ZIP)
for obj,frac in BB_SAMPLES:
    verts,faces,labels,pid,npc=bb_decompress(z,obj,frac)
    warm(verts,faces)
    name=f'Breaking Bad (synthetic) - {npc} fragments'
    np.savez(DATA/f'{idx:02d}_bb.npz',verts=verts,faces=faces,labels=labels,pid=pid,
             name=name,model='bb',npc=npc,explode=1)
    print(f'[{idx}] {name}  ({obj}/{frac}, v={len(verts)})',flush=True); idx+=1
for sd in FB_SAMPLES:
    verts,faces,labels,pid,npc=fb_preprocess(sd)
    warm(verts,faces)
    name=f'Fantastic Breaks (real ceramic) - shard {sd}'
    np.savez(DATA/f'{idx:02d}_fb.npz',verts=verts,faces=faces,labels=labels,pid=pid,
             name=name,model='fb',npc=npc,explode=0)
    print(f'[{idx}] {name}  (v={len(verts)})',flush=True); idx+=1
print('BUILD DONE -> ',DATA,flush=True)

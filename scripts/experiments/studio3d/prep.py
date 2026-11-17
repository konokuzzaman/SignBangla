import numpy as np, json
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
U="/mnt/user-data/uploads/signbangla_phase1_Only_Code/work/sign_dictionary/"
rest=np.load(U+"rest_pose.npy").astype(np.float32)
WORDS=[("W1",38,"বাবা","Father"),("W3",42,"ভাই","Brother"),("W2",25,"আত্মীয়","Relative")]
def mj(n): s=np.linspace(0,1,n); return 10*s**3-15*s**4+6*s**5
def tr(a,b,n): w=mj(n)[:,None,None]; return a[None]*(1-w)+b[None]*w
segs=[np.repeat(rest[None],10,0)]; lab=[-1]*10; prev=rest
for wi,(w,nf,_,_) in enumerate(WORDS):
    s=np.load(U+w+".npy")[:nf].astype(np.float32).copy()
    for sl in (slice(25,46),slice(46,67)):
        for t in range(len(s)):
            if np.abs(s[t,sl]).sum()==0: s[t,sl]=rest[sl]
    s=savgol_filter(s,7,2,axis=0)
    n=16 if wi==0 else 14
    segs+=[tr(prev,s[0],n)[1:],s,np.repeat(s[-1:],6,0)]
    lab+=[lab[-1]]*(n//2)+[wi]*(n-1-n//2)+[wi]*(len(s)+6); prev=s[-1]
segs+=[tr(prev,rest,16)[1:],np.repeat(rest[None],12,0)]; lab+=[len(WORDS)-1]*7+[-1]*20
seq=np.concatenate(segs); assert len(seq)==len(lab),(len(seq),len(lab))
t=np.arange(len(seq)); N=len(seq)*2-1
seq=CubicSpline(t,seq,axis=0)(np.linspace(0,len(seq)-1,N)).astype(np.float32)
lab=[lab[min(len(lab)-1,round(i/2))] for i in range(N)]
np.save("seq2d.npy",seq); json.dump({"lab":lab,"words":[[w[2],w[3]] for w in WORDS]},open("meta.json","w"),ensure_ascii=False)
# ---- lift to 3D (three.js coords: x right, y up, z toward camera)
P=seq.copy(); out=np.zeros_like(P); out[...,0]=P[...,0]; out[...,1]=-P[...,1]; out[...,2]=0
def lift(par,ch,par3,mpz_par,mpz_ch,chxy=None):
    cxy=P[:,ch,:2] if chxy is None else chxy
    d=np.linalg.norm(cxy-P[:,par,:2],axis=1); L=np.percentile(d,97)*1.03
    fz=np.sqrt(np.clip(L*L-d*d,0,None))
    sgn=np.tanh(-(mpz_ch-mpz_par)*3.0)       # mediapipe: negative z = toward camera
    sgn=np.where(np.abs(sgn)<0.3,0.3,sgn)
    z=savgol_filter(fz*sgn,21,2)
    return par3+z
for sh,el,wr,hs in ((11,13,15,slice(25,46)),(12,14,16,slice(46,67))):
    ez=lift(sh,el,0,P[:,sh,2],P[:,el,2]); out[:,el,2]=ez
    hw=P[:,hs][:,0,:2]
    wz=lift(el,wr,ez,P[:,el,2],P[:,wr,2],chxy=hw)
    wz=np.maximum(wz,0.12)                    # keep hands in front of the body
    out[:,wr,0]=hw[:,0]; out[:,wr,1]=-hw[:,1]; out[:,wr,2]=wz
    h=P[:,hs]; out[:,hs,2]=wz[:,None]-(h[...,2]-h[:,:1,2])*1.6
json.dump({"frames":np.round(out,3).tolist(),"lab":lab,"words":[[w[2],w[3]] for w in WORDS]},open("seq3d.json","w"),ensure_ascii=False)
print(N, out[...,2].min(), out[...,2].max())

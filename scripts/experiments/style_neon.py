"""Neon motion-capture style for SignBangla Module 2.
Glowing cyan body lines on a dark background, orange (right) and magenta (left) hand bones with white joints,
head placed from the face keypoints. Same smooth motion as style_bplus.py
(Savitzky-Golay smoothing, minimum-jerk transitions, end hold, 60 fps).
Usage from the project folder:
    python scripts/experiments/style_neon.py work/style_previews/neon_ma W20 "মা" 23
Args: output prefix, word file, caption, number of frames of variant 0 [, comma list of frames for a test sheet]
Needs numpy, pillow (with Bangla text shaping), scipy, imageio, imageio-ffmpeg.
"""
import numpy as np, sys, os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
ROOT=Path(__file__).resolve().parents[2]
U=str(ROOT/"work"/"sign_dictionary")+os.sep
FONT=str(ROOT/"assets"/"fonts"/"NotoSansBengali.ttf")
OUT,WORD,CAP,NF=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]); ONLY=sys.argv[5] if len(sys.argv)>5 else None
rest=np.load(U+"rest_pose.npy").astype(np.float32); sig=np.load(U+f"{WORD}.npy")[:NF].astype(np.float32)
for sl in (slice(25,46),slice(46,67)):
    for t in range(len(sig)):
        if np.abs(sig[t,sl]).sum()==0: sig[t,sl]=rest[sl]
if len(sig)>=7: sig=savgol_filter(sig,7,2,axis=0)
def mj(n): s=np.linspace(0,1,n); return 10*s**3-15*s**4+6*s**5
def tr(a,b,n): w=mj(n)[:,None,None]; return a[None]*(1-w)+b[None]*w
seq=np.concatenate([np.repeat(rest[None],10,0),tr(rest,sig[0],16)[1:],sig,np.repeat(sig[-1:],8,0),tr(sig[-1],rest,16)[1:],np.repeat(rest[None],12,0)])
t=np.arange(len(seq)); seq=CubicSpline(t,seq,axis=0)(np.linspace(0,len(seq)-1,len(seq)*2-1)).astype(np.float32)
W,H,S=640,736,3; CAPH=92
CX,CY,SC=W/2,(H-CAPH)*0.40,205
def P(p): return np.array([(CX+p[0]*SC)*S,(CY+p[1]*SC)*S])
def taper(d,a,b,ra,rb,fill):
    a=np.asarray(a,float); b=np.asarray(b,float); v=b-a; L=np.hypot(*v)
    if L<1e-6: d.ellipse([a[0]-ra,a[1]-ra,a[0]+ra,a[1]+ra],fill=fill); return
    n=np.array([-v[1],v[0]])/L
    d.polygon([tuple(a+n*ra),tuple(b+n*rb),tuple(b-n*rb),tuple(a-n*ra)],fill=fill)
    for c,r in ((a,ra),(b,rb)): d.ellipse([c[0]-r,c[1]-r,c[0]+r,c[1]+r],fill=fill)
def chaikin(pts,n=3):
    pts=np.asarray(pts,float)
    for _ in range(n):
        q=[]
        for i in range(len(pts)):
            a,b=pts[i],pts[(i+1)%len(pts)]; q+=[a*0.75+b*0.25,a*0.25+b*0.75]
        pts=np.array(q)
    return [tuple(p) for p in pts]
FING=[(1,2,3,4),(5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20)]
def hand(d,h,col,dark,ol=1.3,hi=None,crease=None):
    Q=[P(p) for p in h]; z=h[:,2]; ol=ol*S
    pw=max(np.linalg.norm(Q[5]-Q[17]),np.linalg.norm(Q[9]-Q[0])*0.55,14*S); hl=np.linalg.norm(Q[9]-Q[0])
    pr=max(pw*0.5,hl*0.26)*0.92
    pts=np.array([Q[0],Q[1],Q[2]*0.35+Q[1]*0.65,Q[5],Q[9],Q[13],Q[17]])
    try: palm=chaikin(pts[ConvexHull(pts).vertices],3)
    except Exception: palm=None
    taper(d,Q[0],Q[9]*0.8+Q[0]*0.2,pr*0.75+ol,pr+ol,dark)
    if palm:
        pc=np.mean(np.array(palm),axis=0)
        d.polygon([tuple(pc+(np.array(p)-pc)*(1+ol/(np.linalg.norm(np.array(p)-pc)+1e-6))) for p in palm],fill=dark)
    taper(d,Q[0],Q[9]*0.8+Q[0]*0.2,pr*0.75,pr,col)
    if palm: d.polygon(palm,fill=col)
    for f in sorted(FING,key=lambda f:-np.mean(z[list(f)])):
        r=[pw*0.15,pw*0.13,pw*0.115,pw*0.10] if f[0]==1 else ([pw*0.095,pw*0.085,pw*0.075,pw*0.068] if f[0]==17 else [pw*0.115,pw*0.10,pw*0.09,pw*0.08])
        pts_=[Q[k] for k in f]; v=pts_[3]-pts_[2]; L=np.linalg.norm(v)+1e-6; pts_[3]=pts_[3]-v/L*r[3]*0.3
        for i in range(3): taper(d,pts_[i],pts_[i+1],r[i]+ol,r[i+1]+ol,dark)
        for i in range(3): taper(d,pts_[i],pts_[i+1],r[i],r[i+1],col)
        if hi:
            for i in range(3):
                a,b=pts_[i],pts_[i+1]; v=b-a; L=np.linalg.norm(v)+1e-6; n=np.array([-v[1],v[0]])/L
                taper(d,a+n*r[i]*0.35,b+n*r[i+1]*0.35,r[i]*0.26,r[i+1]*0.26,hi)
        if crease:
            for i in (1,2):
                p=pts_[i]; v=pts_[i+1]-pts_[i-1]; L=np.linalg.norm(v)+1e-6; n=np.array([-v[1],v[0]])/L
                d.line([tuple(p+n*r[i]*0.6),tuple(p-n*r[i]*0.6)],fill=crease,width=max(1,int(1.1*S)))
font=ImageFont.truetype(FONT,46)
def caption(img,dark=False):
    dd=ImageDraw.Draw(img); bg=(20,24,40) if dark else (248,249,251); fg=(230,240,255) if dark else (28,32,40)
    dd.rectangle([0,H-CAPH,W,H],fill=bg); tw=dd.textlength(CAP,font=font); dd.text(((W-tw)/2,H-CAPH+12),CAP,fill=fg,font=font)
def geo(p):
    g={}
    for k,i in dict(ls=11,rs=12,le=13,re=14,nose=0,eL=2,eR=5,earL=7,earR=8,mL=9,mR=10).items(): g[k]=P(p[i])
    g['lh'],g['rh']=p[25:46],p[46:67]; g['lw'],g['rw']=P(g['lh'][0]),P(g['rh'][0])
    g['mid']=(g['ls']+g['rs'])/2; g['sw']=abs(g['ls'][0]-g['rs'][0])
    hw=max(abs(g['earL'][0]-g['earR'][0])*1.05,g['sw']*0.42); g['hw']=hw; g['hh']=hw*1.3
    g['eyey']=(g['eL'][1]+g['eR'][1])/2; g['mouthy']=(g['mL'][1]+g['mR'][1])/2; g['hcx']=(g['earL'][0]+g['earR'][0])/2
    g['hcy']=(g['eyey']+g['mouthy'])/2-g['hh']*0.06
    arms=[(g['ls'],g['le'],g['lw'],g['lh'],'L'),(g['rs'],g['re'],g['rw'],g['rh'],'R')]; arms.sort(key=lambda a:-a[2][1]); g['arms']=arms
    return g

# ---------------- NEON ----------------
def neon(p):
    g=geo(p); base=Image.new("RGB",(W*S,H*S),(10,14,30)); glow=Image.new("RGB",(W*S,H*S),(0,0,0))
    for im in (glow,base):
        d=ImageDraw.Draw(im)
        CY_=(70,220,255); MG=(255,70,190); OR=(255,170,60)
        hb=[g['hcx']-g['hw']/2,g['hcy']-g['hh']/2,g['hcx']+g['hw']/2,g['hcy']+g['hh']/2]
        d.ellipse(hb,outline=CY_,width=int(4*S))
        for e in (g['eL'],g['eR']): d.ellipse([e[0]-4*S,e[1]-4*S,e[0]+4*S,e[1]+4*S],fill=CY_)
        mid=g['mid']; d.line([tuple(mid),(g['hcx'],hb[3])],fill=CY_,width=int(5*S)); d.line([tuple(g['ls']),tuple(g['rs'])],fill=CY_,width=int(5*S))
        d.line([tuple(mid),(mid[0],mid[1]+g['sw']*1.2)],fill=(40,120,170),width=int(4*S))
        for sh_,el,wr,h,side in g['arms']:
            d.line([tuple(sh_),tuple(el),tuple(wr)],fill=CY_,width=int(6*S),joint="curve")
            for j in (sh_,el): d.ellipse([j[0]-7*S,j[1]-7*S,j[0]+7*S,j[1]+7*S],fill=(255,255,255))
            col=OR if side=='R' else MG; Q=[P(q) for q in h]
            for f in FING: d.line([tuple(Q[0])]+[tuple(Q[k]) for k in f],fill=col,width=int(4*S),joint="curve")
            for k in range(21): d.ellipse([Q[k][0]-3*S,Q[k][1]-3*S,Q[k][0]+3*S,Q[k][1]+3*S],fill=(255,255,255))
    glow=glow.filter(ImageFilter.GaussianBlur(9*S))
    out=Image.fromarray(np.clip(np.asarray(base,np.int32)+np.asarray(glow,np.int32)*1.4,0,255).astype(np.uint8))
    # grid floor lines
    d=ImageDraw.Draw(out)
    for y in range(0,H*S,40*S): d.line([(0,y),(W*S,y)],fill=(18,24,48),width=S)
    out=out.resize((W,H),Image.LANCZOS); caption(out,dark=True); return out

Path(OUT).parent.mkdir(parents=True,exist_ok=True)
if ONLY:
    idx=[int(x) for x in ONLY.split(",")]; fr=[neon(seq[i]) for i in idx]; w,h=fr[0].size
    sh=Image.new("RGB",(w*len(fr),h),"white")
    for k,f_ in enumerate(fr): sh.paste(f_,(k*w,0))
    sh.save(OUT+"_test.png"); sys.exit()
Path(OUT).parent.mkdir(parents=True,exist_ok=True)
frames=[neon(q) for q in seq]
frames[0].save(OUT+".gif",save_all=True,append_images=frames[1:],duration=17,loop=0)
import imageio.v3 as iio
iio.imwrite(OUT+".mp4",np.stack([np.asarray(f) for f in frames]),fps=60,codec="libx264")
print(len(frames))

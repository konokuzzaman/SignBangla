"""Prototype: illustrated Bangladeshi man (panjabi) driven by SignBangla keypoints.
Usage from the project folder:
    python scripts/experiments/man_render.py work/style_previews/man_ma W20 "ma" 23
Args: output prefix, word file, caption, number of frames of variant 0, [optional comma list of frames for a test sheet]
Needs numpy, pillow, scipy, imageio, imageio-ffmpeg.
"""
import numpy as np, sys, math, os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
U=str(Path(__file__).resolve().parents[2]/"work"/"sign_dictionary")+os.sep
OUT, WORD, CAP, NF = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
ONLY = sys.argv[5] if len(sys.argv)>5 else None
rest=np.load(U+"rest_pose.npy").astype(np.float32)
sig=np.load(U+f"{WORD}.npy")[:NF].astype(np.float32)
for sl in (slice(25,46),slice(46,67)):
    for t in range(len(sig)):
        if np.abs(sig[t,sl]).sum()==0: sig[t,sl]=rest[sl]
if len(sig)>=7: sig=savgol_filter(sig,7,2,axis=0)
def mj(n): s=np.linspace(0,1,n); return 10*s**3-15*s**4+6*s**5
def trans(a,b,n): w=mj(n)[:,None,None]; return a[None]*(1-w)+b[None]*w
seq=np.concatenate([np.repeat(rest[None],8,0),trans(rest,sig[0],16)[1:],sig,np.repeat(sig[-1:],8,0),
                    trans(sig[-1],rest,16)[1:],np.repeat(rest[None],10,0)])
mid_s=(seq[:,11]+seq[:,12])/2
off=seq[:,0:11]-mid_s[:,None]; med=np.median(off,axis=0)
seq[:,0:11]=mid_s[:,None]+med[None]+0.85*(off-med[None])
t=np.arange(len(seq)); seq=CubicSpline(t,seq,axis=0)(np.linspace(0,len(seq)-1,len(seq)*2-1)).astype(np.float32)

W,H,S=640,720,3
CX,CY,SC=W/2,H*0.37,215
def P(p): return np.array([(CX+p[0]*SC)*S,(CY+p[1]*SC)*S])
SKIN=(198,140,102); SKIN_SH=(170,112,78); SKIN_HI=(214,160,122); LINE=(120,74,48)
NAIL=(222,178,150); HAIR=(28,22,20); SHIRT=(232,226,212); SHIRT_SH=(206,198,180); SHIRT_LN=(150,140,120)
BG_TOP=(226,233,241); BG_BOT=(205,215,227); LIP=(150,88,70); EYE_W=(245,242,238); IRIS=(52,34,26)

def tapered(d,a,b,ra,rb,fill):
    a=np.asarray(a,float); b=np.asarray(b,float); v=b-a; L=np.hypot(*v)
    if L<1e-6: d.ellipse([a[0]-ra,a[1]-ra,a[0]+ra,a[1]+ra],fill=fill); return
    n=np.array([-v[1],v[0]])/L
    d.polygon([tuple(a+n*ra),tuple(b+n*rb),tuple(b-n*rb),tuple(a-n*ra)],fill=fill)
    for c,r in ((a,ra),(b,rb)): d.ellipse([c[0]-r,c[1]-r,c[0]+r,c[1]+r],fill=fill)

def background():
    g=np.linspace(0,1,H*S)[:,None,None]
    arr=(np.array(BG_TOP)*(1-g)+np.array(BG_BOT)*g).astype(np.uint8)
    return Image.fromarray(np.repeat(arr,W*S,axis=1),"RGB")

def draw_hand(d,h,scale=1.0):
    w0=h[0,:2].copy(); hh=h.copy(); hh[:,:2]=w0+(hh[:,:2]-w0)*scale
    Q=[P(p) for p in hh]; z=hh[:,2]
    hw=np.linalg.norm(Q[5]-Q[17])          # palm width in px
    base=max(hw*0.16,7*S)
    # palm
    palm=[Q[0]*0.85+Q[1]*0.15,Q[1],Q[2]*0.5+Q[1]*0.5,Q[5],Q[9],Q[13],Q[17],Q[0]*0.6+Q[17]*0.4]
    d.polygon([tuple(p) for p in palm],fill=LINE)
    c=np.mean(palm,axis=0); d.polygon([tuple(c+(p-c)*0.94) for p in palm],fill=SKIN)
    for p in (Q[5],Q[9],Q[13],Q[17]): d.ellipse([p[0]-base,p[1]-base,p[0]+base,p[1]+base],fill=SKIN)
    fingers=[(1,2,3,4),(5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20)]
    widths={1:1.18,5:1.0,9:1.04,13:0.96,17:0.84}
    order=sorted(fingers,key=lambda f:-np.mean(z[list(f)]))   # far first
    for f in order:
        k=widths[f[0]]; rs=[base*k*1.05,base*k*0.92,base*k*0.82,base*k*0.70]
        if f[0]==1: rs=[base*1.35,base*1.18,base*1.0,base*0.86]
        for i in range(3): tapered(d,Q[f[i]],Q[f[i+1]],rs[i]+1.6*S,rs[i+1]+1.6*S,LINE)
        for i in range(3): tapered(d,Q[f[i]],Q[f[i+1]],rs[i],rs[i+1],SKIN)
        # subtle crease + nail
        for i in (1,2):
            p=Q[f[i]]; v=Q[f[i+1]]-Q[f[i-1]]; L=np.hypot(*v)+1e-6; n=np.array([-v[1],v[0]])/L
            d.line([tuple(p+n*rs[i]*0.55),tuple(p-n*rs[i]*0.55)],fill=SKIN_SH,width=max(1,int(1.2*S)))
        tip,pre=Q[f[3]],Q[f[2]]; v=tip-pre; L=np.hypot(*v)+1e-6; u=v/L
        nc=tip-u*rs[3]*0.55; r=rs[3]*0.62
        d.ellipse([nc[0]-r,nc[1]-r,nc[0]+r,nc[1]+r],fill=NAIL)

def draw_arm(d,sh,el,wr,near):
    # long sleeve (panjabi): wide at shoulder, tapering, cuff
    sh=sh+np.array([0,-0.06*SC*S]); tapered(d,sh,el,30*S,25*S,SHIRT_LN); tapered(d,el,wr,25*S,19*S,SHIRT_LN)
    tapered(d,sh,el,27.5*S,22.5*S,SHIRT); tapered(d,el,wr,22.5*S,16.5*S,SHIRT)
    # soft fold at elbow
    d.ellipse([el[0]-8*S,el[1]-3*S,el[0]+8*S,el[1]+3*S],fill=SHIRT_SH)
    # wrist
    v=wr-el; u=v/(np.hypot(*v)+1e-6)
    tapered(d,wr-u*4*S,wr+u*12*S,13*S,12*S,SKIN_SH)

def frame(p,caption):
    img=background(); d=ImageDraw.Draw(img)
    ls,rs,le,re,nose=P(p[11]),P(p[12]),P(p[13]),P(p[14]),P(p[0])
    lear,rear=P(p[7]),P(p[8]); lh,rh=p[25:46],p[46:67]
    lw,rw=P(lh[0]),P(rh[0])
    mid=(ls+rs)/2; sw=abs(ls[0]-rs[0])
    # shadow behind body
    shd=Image.new("L",img.size,0); sd=ImageDraw.Draw(shd)
    sd.rounded_rectangle([mid[0]-sw*0.62,mid[1]-sw*0.05,mid[0]+sw*0.70,H*S],radius=int(sw*0.3),fill=70)
    img.paste((180,190,204),(int(8*S),int(10*S)),shd.filter(ImageFilter.GaussianBlur(18*S)))
    d=ImageDraw.Draw(img)
    # torso (panjabi)
    torso=[(mid[0]-sw*0.17,mid[1]-sw*0.24),(rs[0]+sw*0.06,rs[1]-sw*0.15),(rs[0]-sw*0.12,rs[1]+sw*0.12),
           (rs[0]-sw*0.06,H*S),(ls[0]+sw*0.06,H*S),(ls[0]+sw*0.12,ls[1]+sw*0.12),(ls[0]-sw*0.06,ls[1]-sw*0.15),
           (mid[0]+sw*0.17,mid[1]-sw*0.24)]
    d.polygon([(x,y) for x,y in torso],fill=SHIRT_LN)
    tc=np.array([mid[0],mid[1]+sw*0.6]); d.polygon([tuple(tc+(np.array(q)-tc)*0.985) for q in torso],fill=SHIRT)

    # side shading
    d.polygon([(ls[0]+sw*0.04,ls[1]+sw*0.25),(ls[0]+sw*0.05,H*S),(ls[0]-sw*0.08,H*S),(ls[0]-sw*0.06,ls[1]+sw*0.3)],fill=SHIRT_SH)
    # neck
    nk_top=np.array([mid[0]+(nose[0]-mid[0])*0.5,(P(p[9])[1]+P(p[10])[1])/2])
    tapered(d,np.array([mid[0],mid[1]-sw*0.22]),nk_top,sw*0.13,sw*0.125,SKIN_SH)
    # collar (mandarin) + placket
    d.chord([mid[0]-sw*0.19,mid[1]-sw*0.34,mid[0]+sw*0.19,mid[1]-sw*0.08],0,180,fill=SHIRT_SH)
    d.line([(mid[0],mid[1]-sw*0.14),(mid[0],mid[1]+sw*0.65)],fill=SHIRT_LN,width=int(2*S))
    for k in range(3):
        y=mid[1]+sw*(0.0+0.2*k); d.ellipse([mid[0]-4*S,y-4*S,mid[0]+4*S,y+4*S],fill=(180,160,120))
    # head, placed from the real face keypoints
    eL,eR,mL,mR=P(p[2]),P(p[5]),P(p[9]),P(p[10])
    hw=max(abs(lear[0]-rear[0])*1.08, sw*0.5); Hh=hw*1.32
    ey=(eL[1]+eR[1])/2; cxh=(lear[0]+rear[0])/2
    top,bot=ey-Hh*0.52,ey+Hh*0.50
    for e_,sgn in ((lear,1),(rear,-1)):
        ex=e_[0]; d.ellipse([ex-hw*0.09,ey-hw*0.08,ex+hw*0.09,ey+hw*0.2],fill=SKIN_SH)
    d.ellipse([cxh-hw*0.5,top,cxh+hw*0.5,bot],fill=SKIN)
    d.ellipse([cxh-hw*0.36,bot-Hh*0.32,cxh+hw*0.36,bot+Hh*0.02],fill=SKIN)
    d.chord([cxh-hw*0.54,top-Hh*0.06,cxh+hw*0.54,ey+Hh*0.02],180,360,fill=HAIR)
    d.ellipse([cxh-hw*0.53,top+Hh*0.12,cxh-hw*0.40,ey-Hh*0.05],fill=HAIR)
    d.ellipse([cxh+hw*0.40,top+Hh*0.12,cxh+hw*0.53,ey-Hh*0.05],fill=HAIR)
    hc=np.array([cxh,(top+bot)/2]); hr=Hh/2
    er=hw*0.075
    for e_,sgn in ((eR,-1),(eL,1)):
        d.ellipse([e_[0]-er*1.6,e_[1]-er*0.85,e_[0]+er*1.6,e_[1]+er*0.85],fill=EYE_W)
        d.ellipse([e_[0]-er*0.8,e_[1]-er*0.8,e_[0]+er*0.8,e_[1]+er*0.8],fill=IRIS)
        by=e_[1]-hw*0.13
        d.line([(e_[0]-er*2.0,by+sgn*0),(e_[0]+er*2.0,by-er*0.3)] if sgn>0 else [(e_[0]-er*2.0,by-er*0.3),(e_[0]+er*2.0,by)],fill=HAIR,width=max(2,int(hw*0.035)))
    d.line([((eL[0]+eR[0])/2,ey+er),(nose[0]-hw*0.03,nose[1]),(nose[0]+hw*0.04,nose[1]+hw*0.02)],fill=SKIN_SH,width=max(2,int(hw*0.025)))
    mx,my=(mL[0]+mR[0])/2,(mL[1]+mR[1])/2; mw=max(abs(mL[0]-mR[0])*0.6,hw*0.14)
    d.arc([mx-mw,my-hw*0.06,mx+mw,my+hw*0.06],20,160,fill=LIP,width=max(2,int(hw*0.03)))
    # light beard shadow
    bd=Image.new("L",img.size,0); bdd=ImageDraw.Draw(bd)
    bdd.chord([hc[0]-hr*0.45,hc[1]+hr*0.2,hc[0]+hr*0.45,hc[1]+hr*1.0],15,165,fill=18)
    img.paste((95,70,58),(0,0),bd.filter(ImageFilter.GaussianBlur(3*S)))
    d=ImageDraw.Draw(img)
    # arms + hands: lower (resting) arm first, active arm last; signer's right arm is on image left
    arms=[(ls,le,lw,lh),(rs,re,rw,rh)]
    arms.sort(key=lambda a:-a[2][1])
    for sh,el,wr,h in arms[:1]: draw_arm(d,sh,el,wr,False); draw_hand(d,h)
    for sh,el,wr,h in arms[1:]: draw_arm(d,sh,el,wr,True); draw_hand(d,h)
    img=img.resize((W,H),Image.LANCZOS); dd=ImageDraw.Draw(img)
    try:
        f=ImageFont.truetype("arialbd.ttf",32)
    except OSError:
        f=ImageFont.load_default()
    dd.rectangle([0,H-64,W,H],fill=(255,255,255)); tw=dd.textlength(caption,font=f)
    dd.text(((W-tw)/2,H-54),caption,fill=(30,40,55),font=f)
    return img
if ONLY:
    idx=[int(x) for x in ONLY.split(",")]; fr=[frame(seq[i],CAP) for i in idx]
    w,h=fr[0].size; sheet=Image.new("RGB",(w*len(fr),h),"white")
    for k,f_ in enumerate(fr): sheet.paste(f_,(k*w,0))
    sheet.save(OUT+"_test.png"); sys.exit()
Path(OUT).parent.mkdir(parents=True,exist_ok=True)
frames=[frame(q,CAP) for q in seq]
frames[0].save(OUT+".gif",save_all=True,append_images=frames[1::2][1:] if False else frames[1:],duration=17,loop=0)
import imageio.v3 as iio
iio.imwrite(OUT+".mp4",np.stack([np.asarray(f) for f in frames]),fps=60,codec="libx264")
idx=np.linspace(0,len(frames)-1,8).astype(int); w,h=frames[0].size
sheet=Image.new("RGB",(w*2,h),"white")
for k,i in enumerate(idx): sheet.paste(frames[i].resize((w//2,h//2)),((k%4)*w//2,(k//4)*h//2))
sheet.save(OUT+"_sheet.png"); print(len(frames))

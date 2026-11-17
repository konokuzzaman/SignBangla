"""Prototype "figure" animation style for Module 2 (reference for integrating into render.py).

Usage (from the project folder):
    python scripts/experiments/prototype_render.py work/style_previews/proto_ma W20 "ma" 23
    python scripts/experiments/prototype_render.py work/style_previews/proto_cha W45 "cha" 30
Arguments: output path prefix, word file (W20), caption text, number of frames of variant 0.
Writes <prefix>.gif (30 fps), <prefix>.mp4 (60 fps) and <prefix>_sheet.png.
Needs: numpy, pillow, scipy, imageio, imageio-ffmpeg.
"""
import numpy as np, sys, os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
ROOT=Path(__file__).resolve().parents[2]
U=str(ROOT/"work"/"sign_dictionary")+os.sep
OUT=sys.argv[1]; word=sys.argv[2]; cap=sys.argv[3]
rest=np.load(U+"rest_pose.npy").astype(np.float32)          # (67,3)
sig=np.load(U+f"{word}.npy")[:int(sys.argv[4])].astype(np.float32)
# fill absent hands from rest pose
for sl in (slice(25,46),slice(46,67)):
    for t in range(len(sig)):
        if np.abs(sig[t,sl]).sum()==0: sig[t,sl]=rest[sl]
# smooth sign (keep shape): light savgol
if len(sig)>=7: sig=savgol_filter(sig,7,2,axis=0)
def mj(n):  # minimum-jerk 0..1
    s=np.linspace(0,1,n); return 10*s**3-15*s**4+6*s**5
def trans(a,b,n): w=mj(n)[:,None,None]; return a[None]*(1-w)+b[None]*w
seq=np.concatenate([np.repeat(rest[None],8,0),trans(rest,sig[0],14)[1:],sig,
                    np.repeat(sig[-1:],7,0),trans(sig[-1],rest,14)[1:],np.repeat(rest[None],10,0)])
# upsample 30 -> 60 fps
t=np.arange(len(seq)); tt=np.linspace(0,len(seq)-1,len(seq)*2-1)
seq=CubicSpline(t,seq,axis=0)(tt).astype(np.float32)

W,H,S=640,720,3         # supersample
CX,CY,SC=W/2,H*0.30,175  # shoulder-mid at CX,CY; 1 shoulder width = SC px
def px(p): return (CX+p[0]*SC)*S,(CY+p[1]*SC)*S
BG=(238,242,247); SHIRT=(64,88,122); SHIRT_D=(38,55,79); SLEEVE=(92,124,166); EDGE=(30,42,60); SKIN=(222,170,130); SKIN_D=(170,118,82)
HEAD=(214,166,128); HAIR=(40,32,30); RH=(232,140,60); LH=(70,130,200)
def capsule(d,a,b,r,fill):
    d.line([a,b],fill=fill,width=int(2*r)); 
    for c in (a,b): d.ellipse([c[0]-r,c[1]-r,c[0]+r,c[1]+r],fill=fill)
def hand(d,h,col,scale=1.55):
    w0=h[0,:2]; h=h.copy(); h[:,:2]=w0+(h[:,:2]-w0)*scale
    P=[px(p) for p in h]
    out=tuple(int(c*0.6) for c in col)
    palm=[P[i] for i in (0,1,5,9,13,17)]
    d.polygon(palm,fill=out); 
    d.polygon([((x-P[9][0])*0.93+P[9][0],(y-P[9][1])*0.93+P[9][1]) for x,y in palm],fill=col)
    for f in ((1,2,3,4),(5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20)):
        rr=[5.2,4.6,4.1,3.7] if f[0]!=1 else [6,5.4,4.8,4.2]
        for k in range(3):
            a,b=P[f[k]],P[f[k+1]]
            capsule(d,a,b,(rr[k]+1.4)*S,out)
        for k in range(3):
            a,b=P[f[k]],P[f[k+1]]
            capsule(d,a,b,rr[k]*S,col)
def frame(p,caption):
    img=Image.new("RGB",(W*S,H*S),BG); d=ImageDraw.Draw(img)
    ls,rs,le,re=p[11],p[12],p[13],p[14]; lh,rh=p[25:46],p[46:67]
    lw,rw=lh[0],rh[0]
    neck=(ls+rs)/2; nose=p[0]
    # shadow
    sh=Image.new("L",img.size,0); sd=ImageDraw.Draw(sh)
    xa,xb=sorted([px(ls)[0],px(rs)[0]]); sd.rounded_rectangle([xa-10*S,px((0,0.05))[1],xb+40*S,H*S],radius=60*S,fill=60)
    img.paste((205,212,222),(0,0),sh.filter(ImageFilter.GaussianBlur(25*S)))
    # torso: rounded shirt shape
    x0,x1=sorted([px(ls)[0],px(rs)[0]]); y0=px(neck)[1]
    d.rounded_rectangle([x0-22*S,y0-16*S,x1+22*S,H*S+50*S],radius=64*S,fill=EDGE)
    d.rounded_rectangle([x0-18*S,y0-12*S,x1+18*S,H*S+50*S],radius=60*S,fill=SHIRT)
    # neck + head
    hc=np.array(px((nose[0],nose[1]-0.05)))
    capsule(d,px((neck[0],neck[1]-0.05)),(hc[0],hc[1]+40*S),24*S,SKIN_D)
    r=62*S
    d.ellipse([hc[0]-r*0.95,hc[1]-r*1.12,hc[0]+r*0.95,hc[1]+r*0.2],fill=HAIR)
    d.ellipse([hc[0]-r*0.82,hc[1]-r*0.92,hc[0]+r*0.82,hc[1]+r*1.0],fill=HEAD)
    d.chord([hc[0]-r*0.86,hc[1]-r*1.05,hc[0]+r*0.86,hc[1]-r*0.25],180,360,fill=HAIR)
    # collar
    d.polygon([px((neck[0]-0.12,neck[1]-0.03)),px((neck[0]+0.12,neck[1]-0.03)),px((neck[0],neck[1]+0.16))],fill=SHIRT_D)
    # arms: draw lower hand last
    arms=[(ls,le,lw,lh,LH),(rs,re,rw,rh,RH)]
    arms.sort(key=lambda a:-a[2][1])   # higher hand (smaller y) drawn last
    for s_,e,w,h,c in arms:
        capsule(d,px(s_),px(e),27*S,EDGE); capsule(d,px(e),px(w),23*S,EDGE)
        capsule(d,px(s_),px(e),23*S,SLEEVE); capsule(d,px(e),px(w),19*S,SLEEVE)
        cuff=np.array(px(w)); d.ellipse([cuff[0]-15*S,cuff[1]-15*S,cuff[0]+15*S,cuff[1]+15*S],fill=SKIN_D)
    for s_,e,w,h,c in arms: hand(d,h,SKIN if True else c)
    # thin colour ring on wrists so L/R readable
    for s_,e,w,h,c in arms:
        q=np.array(px(w)); pass
    img=img.resize((W,H),Image.LANCZOS)
    dd=ImageDraw.Draw(img)
    try:
        f=ImageFont.truetype("arialbd.ttf",34)
    except OSError:
        f=ImageFont.load_default()
    dd.rectangle([0,H-70,W,H],fill=(255,255,255)); tw=dd.textlength(caption,font=f)
    dd.text(((W-tw)/2,H-58),caption,fill=(30,40,55),font=f)
    return img
frames=[frame(p,cap) for p in seq]
Path(OUT).parent.mkdir(parents=True,exist_ok=True)
frames[0].save(OUT+".gif",save_all=True,append_images=frames[1:],duration=1000//60*2,loop=0)  # GIF at 30fps
import imageio.v3 as iio
iio.imwrite(OUT+".mp4",np.stack([np.asarray(f) for f in frames]),fps=60,codec="libx264")
idx=np.linspace(0,len(frames)-1,8).astype(int); w,h=frames[0].size
sheet=Image.new("RGB",(w*4//2,h*2//2),"white")
for k,i in enumerate(idx): sheet.paste(frames[i].resize((w//2,h//2)),((k%4)*w//2,(k//4)*h//2))
sheet.save(OUT+"_sheet.png"); print(len(frames),"frames")

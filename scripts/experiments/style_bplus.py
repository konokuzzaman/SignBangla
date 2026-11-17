"""Improved skeleton style (style B+) for SignBangla Module 2.
Same look as style B, with: smooth motion (Savitzky-Golay + minimum-jerk transitions + end hold, 60 fps),
forearm always attached to the HAND wrist, natural hands (smoothed palm, slim tapered fingers with
rounded tips, highlights and knuckle creases) drawn back-to-front, head placed from the real face
keypoints with faint eye/nose/mouth guides, active arm drawn in front.
Usage from the project folder:
    python scripts/experiments/style_bplus.py work/style_previews/bp_ma W20 "মা" 23
Args: output prefix, word file, caption, number of frames of variant 0 [, comma list of frames for a test sheet]
Needs numpy, pillow (with Bangla text shaping), scipy, imageio, imageio-ffmpeg.
"""
import numpy as np, sys, os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
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
BG=(240,243,246); BONE=(55,64,78); JOINT_IN=(255,255,255); TORSO=(226,230,235); TORSO_LN=(160,166,175)
HEAD=(226,230,235); HEAD_LN=(150,156,166); FACE=(150,156,166)
ORANGE=(250,140,40); ORANGE_D=(196,96,20); BLUE=(40,128,240); BLUE_D=(22,84,180)
def cap(d,a,b,r,fill):
    d.line([tuple(a),tuple(b)],fill=fill,width=int(2*r))
    for c in (a,b): d.ellipse([c[0]-r,c[1]-r,c[0]+r,c[1]+r],fill=fill)
def taper(d,a,b,ra,rb,fill):
    a=np.asarray(a,float); b=np.asarray(b,float); v=b-a; L=np.hypot(*v)
    if L<1e-6: d.ellipse([a[0]-ra,a[1]-ra,a[0]+ra,a[1]+ra],fill=fill); return
    n=np.array([-v[1],v[0]])/L
    d.polygon([tuple(a+n*ra),tuple(b+n*rb),tuple(b-n*rb),tuple(a-n*ra)],fill=fill)
    for c,r in ((a,ra),(b,rb)): d.ellipse([c[0]-r,c[1]-r,c[0]+r,c[1]+r],fill=fill)
FING=[(1,2,3,4),(5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20)]
def lighten(c,f): return tuple(int(x+(255-x)*f) for x in c)
def chaikin(pts,n=3):
    pts=np.asarray(pts,float)
    for _ in range(n):
        q=[]
        for i in range(len(pts)):
            a,b=pts[i],pts[(i+1)%len(pts)]; q+= [a*0.75+b*0.25, a*0.25+b*0.75]
        pts=np.array(q)
    return [tuple(p) for p in pts]
def hand(d,h,col,dark):
    Q=[P(p) for p in h]; z=h[:,2]
    pw=max(np.linalg.norm(Q[5]-Q[17]),np.linalg.norm(Q[9]-Q[0])*0.55,14*S)          # palm width in px
    ol=1.3*S                                          # outline thickness
    hi=lighten(col,0.35); crease=tuple(int(x*0.8) for x in col)
    # palm: smoothed convex outline of wrist, thumb base and knuckles, with a minimum thickness
    from scipy.spatial import ConvexHull
    hl=np.linalg.norm(Q[9]-Q[0])
    pr=max(pw*0.5,hl*0.26)*0.92
    pts=np.array([Q[0],Q[1],Q[2]*0.35+Q[1]*0.65,Q[5],Q[9],Q[13],Q[17]])
    try:
        hull=pts[ConvexHull(pts).vertices]; palm=chaikin(hull,3)
    except Exception:
        palm=None
    taper(d,Q[0],Q[9]*0.8+Q[0]*0.2,pr*0.75+ol,pr+ol,dark)
    if palm:
        pc=np.mean(np.array(palm),axis=0)
        d.polygon([tuple(pc+(np.array(p)-pc)*(1+ol/(np.linalg.norm(np.array(p)-pc)+1e-6))) for p in palm],fill=dark)
    taper(d,Q[0],Q[9]*0.8+Q[0]*0.2,pr*0.75,pr,col)
    if palm: d.polygon(palm,fill=col)
    # fingers: back (far) fingers first
    for f in sorted(FING,key=lambda f:-np.mean(z[list(f)])):
        if f[0]==1: r=[pw*0.15,pw*0.13,pw*0.115,pw*0.10]
        elif f[0]==17: r=[pw*0.095,pw*0.085,pw*0.075,pw*0.068]
        else: r=[pw*0.115,pw*0.10,pw*0.09,pw*0.08]
        pts=[Q[k] for k in f]
        # extend tip slightly so the rounded end reaches the real fingertip
        v=pts[3]-pts[2]; L=np.linalg.norm(v)+1e-6; tip=pts[3]-v/L*r[3]*0.3
        pts=pts[:3]+[tip]
        for i in range(3): taper(d,pts[i],pts[i+1],r[i]+ol,r[i+1]+ol,dark)
        for i in range(3): taper(d,pts[i],pts[i+1],r[i],r[i+1],col)
        # soft highlight along the finger for roundness
        for i in range(3):
            a,b=pts[i],pts[i+1]; v=b-a; L=np.linalg.norm(v)+1e-6; n=np.array([-v[1],v[0]])/L
            taper(d,a+n*r[i]*0.35,b+n*r[i+1]*0.35,r[i]*0.28,r[i+1]*0.28,hi)
        # knuckle creases at the two finger joints
        for i in (1,2):
            p=pts[i]; v=pts[i+1]-pts[i-1]; L=np.linalg.norm(v)+1e-6; n=np.array([-v[1],v[0]])/L
            d.line([tuple(p+n*r[i]*0.6),tuple(p-n*r[i]*0.6)],fill=crease,width=max(1,int(1.1*S)))
font=ImageFont.truetype(FONT,46)
def frame(p):
    img=Image.new("RGB",(W*S,H*S),BG)
    sh=Image.new("L",img.size,0); sd=ImageDraw.Draw(sh)
    d=ImageDraw.Draw(img)
    ls,rs,le,re=P(p[11]),P(p[12]),P(p[13]),P(p[14]); lh,rh=p[25:46],p[46:67]
    lw,rw=P(lh[0]),P(rh[0]); mid=(ls+rs)/2; sw=abs(ls[0]-rs[0])
    # head from real face keypoints
    eL,eR,nose,earL,earR,mL,mR=[P(p[i]) for i in (2,5,0,7,8,9,10)]
    hwid=max(abs(earL[0]-earR[0])*1.05,sw*0.42); hh=hwid*1.3
    eyey=(eL[1]+eR[1])/2; hcx=(earL[0]+earR[0])/2; mouthy=(mL[1]+mR[1])/2
    hcy=(eyey+mouthy)/2-hh*0.06
    hb=[hcx-hwid/2,hcy-hh/2,hcx+hwid/2,hcy+hh/2]
    torso=[(rs[0]+3*S,rs[1]),(ls[0]-3*S,ls[1]),(ls[0]-sw*0.13,mid[1]+sw*1.25),(rs[0]+sw*0.13,mid[1]+sw*1.25)]
    # soft shadow of everything static
    sd.polygon(torso,fill=90); sd.ellipse(hb,fill=90)
    for a,b in ((ls,le),(le,lw),(rs,re),(re,rw)): sd.line([tuple(a),tuple(b)],fill=90,width=int(30*S))
    img.paste((214,220,228),(int(6*S),int(8*S)),sh.filter(ImageFilter.GaussianBlur(14*S)))
    d=ImageDraw.Draw(img)
    d.polygon(torso,fill=TORSO_LN); tc=np.mean(np.array(torso),axis=0)
    d.polygon([tuple(tc+(np.array(q)-tc)*0.985) for q in torso],fill=TORSO)
    # neck + head
    cap(d,mid,np.array([hcx,hb[3]-2*S]),7.5*S,BONE)
    d.ellipse([hb[0]-2.4*S,hb[1]-2.4*S,hb[2]+2.4*S,hb[3]+2.4*S],fill=HEAD_LN); d.ellipse(hb,fill=HEAD)
    # faint face guides (eyes, nose, mouth) so face contact points are readable
    for e in (eL,eR): d.ellipse([e[0]-5*S,e[1]-5*S,e[0]+5*S,e[1]+5*S],fill=FACE)
    d.line([tuple(nose+np.array([0,-6*S])),tuple(nose+np.array([0,5*S]))],fill=FACE,width=int(3*S))
    d.line([(mL[0]-2*S,mouthy),(mR[0]+2*S,mouthy)],fill=FACE,width=int(3*S))
    # shoulder line
    cap(d,ls,rs,7.5*S,BONE)
    arms=[(ls,le,lw,lh,BLUE,BLUE_D),(rs,re,rw,rh,ORANGE,ORANGE_D)]
    arms.sort(key=lambda a:-a[2][1])           # raised (active) arm drawn last, in front
    for sh_,el,wr,h,col,dk in arms:
        cap(d,sh_,el,8.5*S,BONE); cap(d,el,wr,8*S,BONE)
        for j in (sh_,el):
            d.ellipse([j[0]-9.5*S,j[1]-9.5*S,j[0]+9.5*S,j[1]+9.5*S],fill=BONE)
            d.ellipse([j[0]-5.5*S,j[1]-5.5*S,j[0]+5.5*S,j[1]+5.5*S],fill=JOINT_IN)
        hand(d,h,col,dk)
    img=img.resize((W,H),Image.LANCZOS); dd=ImageDraw.Draw(img)
    dd.rectangle([0,H-CAPH,W,H],fill=(248,249,251)); dd.line([(0,H-CAPH),(W,H-CAPH)],fill=(225,229,234),width=1)
    tw=dd.textlength(CAP,font=font); dd.text(((W-tw)/2,H-CAPH+12),CAP,fill=(28,32,40),font=font)
    return img
Path(OUT).parent.mkdir(parents=True,exist_ok=True)
if ONLY:
    idx=[int(x) for x in ONLY.split(",")]; fr=[frame(seq[i]) for i in idx]; w,h=fr[0].size
    sheet=Image.new("RGB",(w*len(fr),h),"white")
    for k,f_ in enumerate(fr): sheet.paste(f_,(k*w,0))
    sheet.save(OUT+"_test.png"); sys.exit()
Path(OUT).parent.mkdir(parents=True,exist_ok=True)
frames=[frame(q) for q in seq]
frames[0].save(OUT+".gif",save_all=True,append_images=frames[1:],duration=17,loop=0)
import imageio.v3 as iio
iio.imwrite(OUT+".mp4",np.stack([np.asarray(f) for f in frames]),fps=60,codec="libx264")
print(len(frames))

"""
enhance_single.py — EAIM-Net v5
Enhance one image. Produces a 2-panel figure showing:
  Left  — Input image + EPE condition cards
  Right — Enhanced image + filter weights + diff map + metrics

Usage:
  python scripts/enhance_single.py \
      --input      image.jpg \
      --output     enhanced.jpg \
      --checkpoint best_model.pth \
      --gt         ground_truth.jpg   # optional
"""
import argparse, os, sys, time
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as GS
import matplotlib.patches as mpatches
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from complete_model import AdaptiveEnhancementModel
from config import CONFIG

FN  = ["Low-light","Dehazing","Rain removal","Illum. norm","Glare red."]
FC  = ["#378ADD","#1D9E75","#EF9F27","#7F77DD","#D85A30"]
WN  = ["Clear","Rain","Fog","Light Snow","Glare"]
TN  = ["Dawn","Day","Dusk","Night"]
IN  = ["Low","Medium","High"]
WDESC={"Clear":"No atmospheric degradation detected",
       "Rain":"Rain streaks detected — rain removal active",
       "Fog":"Haze / fog — dehazing active",
       "Light Snow":"Light snow particles detected",
       "Glare":"Specular glare / hotspots detected"}
TDESC={"Dawn":"Low ambient light, warm colour cast","Day":"Daylight conditions",
       "Dusk":"Fading light, warm/orange tint","Night":"Very low light, severe underexposure"}
IDESC={"Low":"Low illumination — strong brightening needed",
       "Medium":"Normal illumination","High":"High illumination — possible overexposure"}
CTX={"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
     "font.size":10,"axes.titlesize":10,"savefig.dpi":400}

def load_model(p,dev):
    m=AdaptiveEnhancementModel(
        backbone=CONFIG["model"]["backbone"],
        num_weather_classes=CONFIG["model"]["num_weather_classes"],
        num_time_classes=CONFIG["model"]["num_time_classes"],
        num_illum_classes=CONFIG["model"]["num_illum_classes"],
        feature_dim=CONFIG["model"]["feature_dim"],
        num_filters=CONFIG["model"]["num_filters"],pretrained=False).to(dev)
    ck=torch.load(p,map_location=dev)
    try: m.load_state_dict(ck["model_state_dict"])
    except: m.load_state_dict(ck["model_state_dict"],strict=False)
    m.eval(); print(f"  Model: {os.path.basename(p)}  ({sum(p.numel() for p in m.parameters())/1e6:.1f}M)"); return m

def run(model,pil,dev):
    IW,IH=pil.size; PW=((IW+3)//4)*4; PH=((IH+3)//4)*4
    pad=Image.new("RGB",(PW,PH)); pad.paste(pil,(0,0))
    t=transforms.ToTensor()(pad).unsqueeze(0).to(dev)
    with torch.no_grad(): out=model(t)
    enh=out["enhanced"][0].cpu().numpy().transpose(1,2,0)[:IH,:IW]
    fw=out["filter_weights"][0].cpu().numpy()
    H=out["weight_entropy"][0].item(); tau=model.ess.tau.item()
    wl=out["weather_logits"][0].argmax().item()
    tl=out["time_logits"][0].argmax().item()
    il=out["illum_logits"][0].argmax().item()
    return np.clip(enh,0,1),fw,H,tau,wl,tl,il

def mets(pred,gt):
    from skimage.metrics import peak_signal_noise_ratio as pf,structural_similarity as sf
    return pf(gt,pred,data_range=1.0),sf(gt,pred,data_range=1.0,channel_axis=2)

def make_report(inp_path,inp_np,enh_np,gt_np,fw,H,tau,wl,tl,il,stem):
    has_gt=gt_np is not None
    fname=os.path.basename(inp_path); IH,IW=inp_np.shape[:2]
    wname=WN[wl]; tname=TN[tl]; iname=IN[il]; dom=int(np.argmax(fw)); nact=int(np.sum(fw>0.05))
    if has_gt:
        pi,si=mets(inp_np,gt_np); pe,se=mets(enh_np,gt_np); dp=pe-pi; dc="#1a6b1a" if dp>=0 else "#b31b1b"

    with matplotlib.rc_context(CTX):
        fig=plt.figure(figsize=(16,11),facecolor="white")
        # Top two image panels
        gi=GS.GridSpec(1,2,figure=fig,left=0.04,right=0.97,top=0.91,bottom=0.50,wspace=0.05)
        # Bottom four info panels
        gb=GS.GridSpec(1,4,figure=fig,left=0.04,right=0.97,top=0.46,bottom=0.04,wspace=0.22)

        # ── Input ──────────────────────────────────────────────────────────
        ai=fig.add_subplot(gi[0,0]); ai.imshow(np.clip(inp_np,0,1),interpolation="lanczos"); ai.axis("off")
        it=f"INPUT — {fname}   {IW}×{IH} px"
        if has_gt: it+=f"\nPSNR {pi:.2f} dB   SSIM {si:.4f}"
        ai.set_title(it,fontsize=10,pad=6,color="#333")
        ai.text(0.01,0.99,"INPUT",transform=ai.transAxes,va="top",ha="left",fontsize=9,fontweight="bold",color="white",
                bbox=dict(fc="#1B3A6B",ec="none",pad=3,alpha=0.88))

        # ── Enhanced ────────────────────────────────────────────────────────
        ae=fig.add_subplot(gi[0,1]); ae.imshow(np.clip(enh_np,0,1),interpolation="lanczos"); ae.axis("off")
        et=f"ENHANCED — {wname} · {tname} · Illum={iname}\nH={H:.4f}   dominant: {FN[dom]}   {nact}/5 filters active"
        if has_gt: et+=f"\nPSNR {pe:.2f} dB   SSIM {se:.4f}   dPSNR {dp:+.2f} dB"; ecol=dc
        else: ecol="#1a6b1a"
        ae.set_title(et,fontsize=10,pad=6,color=ecol)
        ae.text(0.01,0.99,"ENHANCED",transform=ae.transAxes,va="top",ha="left",fontsize=9,fontweight="bold",color="white",
                bbox=dict(fc="#0D6E6E",ec="none",pad=3,alpha=0.88))

        # ── INFO 0: EPE condition cards ────────────────────────────────────
        a0=fig.add_subplot(gb[0,0]); a0.axis("off"); a0.set_xlim(0,1); a0.set_ylim(0,1)
        a0.text(0.5,0.97,"EPE Condition Analysis",ha="center",va="top",fontsize=10,fontweight="bold",color="#1B3A6B",transform=a0.transAxes)
        cards=[("Weather",wname,WDESC[wname],"#378ADD"),
               ("Time of day",tname,TDESC[tname],"#1D9E75"),
               ("Illumination",iname,IDESC[iname],"#7F77DD")]
        for k,(lbl,val,desc,col) in enumerate(cards):
            y=0.82-k*0.30
            a0.add_patch(mpatches.FancyBboxPatch((0.02,y-0.10),0.96,0.26,
                boxstyle="round,pad=0.02",fc=col+"18",ec=col,lw=1.2,transform=a0.transAxes))
            a0.text(0.50,y+0.12,lbl,ha="center",va="center",fontsize=8,color=col,fontweight="bold",transform=a0.transAxes)
            a0.text(0.50,y+0.02,val,ha="center",va="center",fontsize=11,color="#111",transform=a0.transAxes)
            a0.text(0.50,y-0.06,desc,ha="center",va="center",fontsize=7.5,color="#555",transform=a0.transAxes)
        hcol="#1a6b1a" if H>0.20 else "#b31b1b"
        a0.text(0.5,0.02,f"H={H:.4f}  tau={tau:.4f}  {'Genuine blending' if H>0.20 else 'Low blending'}",
                ha="center",va="bottom",fontsize=8,color=hcol,transform=a0.transAxes,
                bbox=dict(fc=hcol+"12",ec=hcol,pad=2.5,lw=0.8))

        # ── INFO 1: Filter weights ─────────────────────────────────────────
        a1=fig.add_subplot(gb[0,1])
        bars=a1.barh(range(5),fw[::-1],color=FC[::-1],height=0.50,edgecolor="white")
        a1.set_yticks(range(5)); a1.set_yticklabels(FN[::-1],fontsize=9)
        a1.set_xlim(0,1.05); a1.set_xlabel("Blending weight",fontsize=9)
        a1.set_title(f"Filter blending weights\n{nact}/5 filters active (w > 5%)",
                     fontsize=10,fontweight="bold",pad=5,color="#1B3A6B")
        a1.axvline(0.20,color="#888",lw=0.7,ls=":",alpha=0.6)
        a1.spines[["top","right"]].set_visible(False); a1.grid(axis="x",alpha=0.20,lw=0.4); a1.set_axisbelow(True)
        for bar,v,fn2,fc2 in zip(bars,fw[::-1],FN[::-1],FC[::-1]):
            a1.text(v+0.01,bar.get_y()+bar.get_height()/2,f"{v*100:.1f}%",va="center",fontsize=9,color="#333")
            if abs(v-fw[dom])<1e-6: bar.set_edgecolor(fc2); bar.set_linewidth(2)
        a1.text(0.98,0.02,f"Dominant: {FN[dom]} ({fw[dom]*100:.1f}%)",ha="right",va="bottom",fontsize=8.5,
                color=FC[dom],transform=a1.transAxes,bbox=dict(fc="white",ec=FC[dom],pad=2.5,lw=0.8))

        # ── INFO 2: Difference map ─────────────────────────────────────────
        a2=fig.add_subplot(gb[0,2])
        diff=np.clip(np.abs(enh_np-inp_np)*5,0,1)
        im=a2.imshow(diff,cmap="hot",interpolation="lanczos"); a2.axis("off")
        a2.set_title("Difference map (×5 amplified)\nBright = most changed regions",
                     fontsize=10,fontweight="bold",pad=5,color="#1B3A6B")
        cb=plt.colorbar(im,ax=a2,fraction=0.046,pad=0.04); cb.ax.tick_params(labelsize=8)
        cb.set_label("Change magnitude",fontsize=8)

        # ── INFO 3: Summary text ──────────────────────────────────────────
        a3=fig.add_subplot(gb[0,3]); a3.axis("off"); a3.set_xlim(0,1); a3.set_ylim(0,1)
        a3.text(0.0,0.97,"Enhancement Summary",fontsize=10,fontweight="bold",va="top",color="#1B3A6B",transform=a3.transAxes)
        lines=[f"Image: {IW} x {IH} px","",
               "EPE prediction:",f"  Weather     : {wname}",
               f"  Time        : {tname}",f"  Illumination: {iname}","",
               "ESS blending:",f"  Entropy H   : {H:.4f}",
               f"  Temperature : {tau:.4f}",f"  Active filters: {nact}/5","",
               "Filter weights:"]
        for fn2,fv in zip(FN,fw):
            bl="\u2588"*int(fv*18)+"\u2591"*(18-int(fv*18))
            lines.append(f"  {fn2:<14} {bl} {fv*100:5.1f}%")
        if has_gt:
            lines+=[f"","Metrics vs GT:",f"  Input  PSNR: {pi:.2f} dB  SSIM: {si:.4f}",
                    f"  Enh.   PSNR: {pe:.2f} dB  SSIM: {se:.4f}",f"  dPSNR:  {dp:+.2f} dB"]
        for k,line in enumerate(lines):
            a3.text(0.02,0.91-k*0.058,line,va="top",fontsize=7.8,color="#222",
                    fontfamily="monospace",transform=a3.transAxes)

        fig.suptitle("EAIM-Net v5 — Single Image Enhancement Analysis",
                     fontsize=13,fontweight="bold",y=0.99,color="#1B3A6B")

        for ext2,kw in [(".png",{}),(".jpg",{"pil_kwargs":{"quality":96,"subsampling":0}})]:
            fig.savefig(stem+"_report"+ext2,dpi=400,bbox_inches="tight",facecolor="white",**kw)
        plt.close()
        print(f"  Report : {stem}_report.jpg / .png")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--gt",default=None)
    p.add_argument("--quality",type=int,default=95)
    p.add_argument("--device",default="auto",choices=["auto","cuda","cpu"])
    args=p.parse_args()

    dev=(torch.device("cuda" if torch.cuda.is_available() else "cpu")
         if args.device=="auto" else torch.device(args.device))
    print(f"\nEAIM-Net v5 — Single Image Enhancement\n{'='*50}")
    print(f"  Input  : {args.input}\n  Output : {args.output}\n  Device : {dev}")
    assert os.path.exists(args.input), f"Not found: {args.input}"
    assert os.path.exists(args.checkpoint), f"Not found: {args.checkpoint}"
    os.makedirs(os.path.dirname(os.path.abspath(args.output)),exist_ok=True)

    model=load_model(args.checkpoint,dev)
    inp_pil=Image.open(args.input).convert("RGB")
    inp_np=np.array(inp_pil).astype(np.float32)/255.0

    t0=time.time()
    enh_np,fw,H,tau,wl,tl,il=run(model,inp_pil,dev)
    print(f"  Time   : {time.time()-t0:.2f}s")

    enh_pil=Image.fromarray((enh_np*255).astype(np.uint8))
    enh_pil.save(args.output,quality=args.quality) if args.output.lower().endswith((".jpg",".jpeg")) else enh_pil.save(args.output)
    print(f"  Saved  : {args.output}")

    gt_np=None
    if args.gt and os.path.exists(args.gt):
        gp=Image.open(args.gt).convert("RGB")
        if gp.size!=inp_pil.size: gp=gp.resize(inp_pil.size,Image.LANCZOS)
        gt_np=np.array(gp).astype(np.float32)/255.0

    make_report(args.input,inp_np,enh_np,gt_np,fw,H,tau,wl,tl,il,
                os.path.splitext(args.output)[0])

    print(f"\n{'='*50}")
    print(f"  EPE: {WN[wl]} · {TN[tl]} · Illum={IN[il]}")
    print(f"  H={H:.4f}  tau={tau:.4f}  active={int(np.sum(fw>0.05))}/5")
    for fn2,fv in zip(FN,fw):
        print(f"    {fn2:<22} {chr(9608)*int(fv*30):<30} {fv*100:5.1f}%")
    if gt_np is not None:
        pi,si=mets(inp_np,gt_np); pe,se=mets(enh_np,gt_np)
        print(f"  PSNR: {pi:.2f}->{pe:.2f} dB  SSIM: {si:.4f}->{se:.4f}")

if __name__=="__main__":
    main()

"""
enhance_folder.py — EAIM-Net v5
Batch enhance all images in a folder.
Produces: enhanced images + CSV results + visual grid comparison.

Usage:
  python scripts/enhance_folder.py \
      --input_dir   images/ \
      --output_dir  enhanced/ \
      --checkpoint  best_model.pth \
      --gt_dir      ground_truth/   # optional
      --grid_cols   4               # default 4
      --max_images  0               # 0 = all
"""
import argparse, os, sys, time, csv
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as GS
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from complete_model import AdaptiveEnhancementModel
from config import CONFIG

FN  = ["Low-light","Dehazing","Rain removal","Illum. norm","Glare red."]
FC  = ["#378ADD","#1D9E75","#EF9F27","#7F77DD","#D85A30"]
WN  = ["Clear","Rain","Fog","Light Snow","Glare"]
TN  = ["Dawn","Day","Dusk","Night"]
IN  = ["Low","Medium","High"]
EXTS= {".jpg",".jpeg",".png",".bmp",".webp",".tiff"}
CTX = {"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
       "font.size":10,"axes.titlesize":9,"savefig.dpi":400}

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
    m.eval(); return m

def run(model,pil,dev):
    IW,IH=pil.size; PW=((IW+3)//4)*4; PH=((IH+3)//4)*4
    pad=Image.new("RGB",(PW,PH)); pad.paste(pil,(0,0))
    t=transforms.ToTensor()(pad).unsqueeze(0).to(dev)
    with torch.no_grad(): out=model(t)
    enh=out["enhanced"][0].cpu().numpy().transpose(1,2,0)[:IH,:IW]
    fw=out["filter_weights"][0].cpu().numpy()
    H=out["weight_entropy"][0].item()
    wl=out["weather_logits"][0].argmax().item()
    tl=out["time_logits"][0].argmax().item()
    il=out["illum_logits"][0].argmax().item()
    return np.clip(enh,0,1),fw,H,wl,tl,il

def psnr_ssim(pred,gt):
    from skimage.metrics import peak_signal_noise_ratio as pf,structural_similarity as sf
    return pf(gt,pred,data_range=1.0),sf(gt,pred,data_range=1.0,channel_axis=2)

def make_grid(results,out_dir,cols=4):
    n=len(results)
    if n==0: return
    rows=(n+cols-1)//cols
    fw2=cols*3.2; fh2=rows*5.8
    with matplotlib.rc_context(CTX):
        fig,axes=plt.subplots(rows*2,cols,figsize=(fw2,fh2),facecolor="white")
        if rows*2==1: axes=[axes]
        flat=[ax for row in axes for ax in (row if hasattr(row,"__iter__") else [row])]
        for ax in flat: ax.axis("off")

        for idx,r in enumerate(results):
            ci=idx%cols; rb=(idx//cols)*2
            ai=flat[rb*cols+ci]; ae=flat[(rb+1)*cols+ci]
            # Input
            ai.imshow(np.clip(r["inp_np"],0,1),interpolation="lanczos"); ai.axis("off")
            t=r["fname"][:18]
            if r.get("pi") is not None: t+=f"\n{r['pi']:.1f} dB / {r['si']:.3f}"
            ai.set_title(t,fontsize=8,pad=3,color="#555")
            if ci==0: ai.text(-0.05,0.5,"Input",transform=ai.transAxes,
                               rotation=90,ha="right",va="center",fontsize=8,color="#1B3A6B",fontweight="bold")
            # Enhanced
            ae.imshow(np.clip(r["enh_np"],0,1),interpolation="lanczos"); ae.axis("off")
            dom=FN[int(np.argmax(r["fw"]))]
            t2=f"{WN[r['wl']]}  H={r['H']:.2f}\n{dom}"
            col="#1a6b1a"
            if r.get("pe") is not None:
                dp=r["pe"]-r["pi"]; col="#1a6b1a" if dp>=0 else "#b31b1b"
                t2+=f"  dPSNR {dp:+.1f}"
            ae.set_title(t2,fontsize=8,pad=3,color=col)
            if ci==0: ae.text(-0.05,0.5,"Enhanced",transform=ae.transAxes,
                               rotation=90,ha="right",va="center",fontsize=8,color="#0D6E6E",fontweight="bold")

        fig.suptitle("EAIM-Net v5 — Batch Enhancement Results",
                     fontsize=12,fontweight="bold",y=1.002,color="#1B3A6B")
        plt.tight_layout(pad=0.4)
        gp=os.path.join(out_dir,"_grid_comparison.jpg")
        fig.savefig(gp,dpi=200,bbox_inches="tight",facecolor="white",
                    pil_kwargs={"quality":92})
        plt.close(); print(f"  Grid: {gp}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input_dir",required=True)
    p.add_argument("--output_dir",required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--gt_dir",default=None)
    p.add_argument("--quality",type=int,default=92)
    p.add_argument("--grid_cols",type=int,default=4)
    p.add_argument("--max_images",type=int,default=0)
    p.add_argument("--device",default="auto",choices=["auto","cuda","cpu"])
    args=p.parse_args()

    dev=(torch.device("cuda" if torch.cuda.is_available() else "cpu")
         if args.device=="auto" else torch.device(args.device))
    os.makedirs(args.output_dir,exist_ok=True)

    files=sorted([f for f in os.listdir(args.input_dir)
                  if os.path.splitext(f)[1].lower() in EXTS])
    if args.max_images>0: files=files[:args.max_images]

    print(f"\nEAIM-Net v5 — Batch Enhancement\n{'='*50}")
    print(f"  Input dir : {args.input_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Images    : {len(files)}")
    print(f"  Device    : {dev}")

    model=load_model(args.checkpoint,dev)
    results=[]; csv_rows=[]; t_total=0

    for fname in tqdm(files,desc="Enhancing"):
        ip=os.path.join(args.input_dir,fname)
        op=os.path.join(args.output_dir,fname)
        try:
            pil=Image.open(ip).convert("RGB")
            inp_np=np.array(pil).astype(np.float32)/255.0
            t0=time.time()
            enh_np,fw,H,wl,tl,il=run(model,pil,dev); elapsed=time.time()-t0; t_total+=elapsed
            enh_pil=Image.fromarray((enh_np*255).astype(np.uint8))
            enh_pil.save(op,quality=args.quality) if op.lower().endswith((".jpg",".jpeg")) else enh_pil.save(op)
            gt_np=pi=si=pe=se=None
            if args.gt_dir:
                gp2=os.path.join(args.gt_dir,fname)
                if os.path.exists(gp2):
                    gpi=Image.open(gp2).convert("RGB")
                    if gpi.size!=pil.size: gpi=gpi.resize(pil.size,Image.LANCZOS)
                    gt_np=np.array(gpi).astype(np.float32)/255.0
                    pi,si=psnr_ssim(inp_np,gt_np); pe,se=psnr_ssim(enh_np,gt_np)
            r=dict(fname=fname,inp_np=inp_np,enh_np=enh_np,gt_np=gt_np,
                   fw=fw,H=H,wl=wl,tl=tl,il=il,pi=pi,si=si,pe=pe,se=se,elapsed=elapsed)
            results.append(r)
            dom=FN[int(np.argmax(fw))]
            row=dict(filename=fname,weather=WN[wl],time=TN[tl],illum=IN[il],
                     H=f"{H:.4f}",dominant=dom,elapsed_s=f"{elapsed:.2f}",
                     fw=",".join(f"{v:.4f}" for v in fw))
            if gt_np is not None:
                row.update(psnr_in=f"{pi:.2f}",ssim_in=f"{si:.4f}",
                           psnr_enh=f"{pe:.2f}",ssim_enh=f"{se:.4f}",
                           dpsnr=f"{pe-pi:+.2f}")
            csv_rows.append(row)
        except Exception as e:
            tqdm.write(f"  [skip] {fname}: {e}")

    # CSV
    if csv_rows:
        cp=os.path.join(args.output_dir,"_results.csv")
        with open(cp,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=csv_rows[0].keys()); w.writeheader(); w.writerows(csv_rows)
        print(f"\n  CSV: {cp}")

    make_grid(results,args.output_dir,cols=args.grid_cols)

    n=len(results)
    print(f"\n{'='*50}\n  BATCH COMPLETE\n{'='*50}")
    print(f"  Processed : {n}/{len(files)}")
    if n: print(f"  Avg time  : {t_total/n:.2f}s/image")
    valid=[r for r in results if r.get("pe") is not None]
    if valid:
        print(f"  Mean dPSNR: {np.mean([r['pe']-r['pi'] for r in valid]):+.2f} dB")
        print(f"  Mean SSIM : {np.mean([r['se'] for r in valid]):.4f}")
    print(f"{'='*50}")

if __name__=="__main__":
    main()

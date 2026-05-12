"""
evaluate.py — EAIM-Net v5
==========================
Full quantitative evaluation: PSNR / SSIM / LPIPS on standard benchmarks.

Usage:
    python scripts/evaluate.py --dataset lol      --data_root /path/to/LOL/eval    --checkpoint checkpoints/best_model.pth
    python scripts/evaluate.py --dataset reside   --data_root /path/to/RESIDE_SOTS --checkpoint checkpoints/best_model.pth
    python scripts/evaluate.py --dataset rain100l --data_root /path/to/Rain100L    --checkpoint checkpoints/best_model.pth
    python scripts/evaluate.py --dataset rain100h --data_root /path/to/Rain100H    --checkpoint checkpoints/best_model.pth
    python scripts/evaluate.py --dataset didmdn   --data_root /path/to/DID-MDN     --checkpoint checkpoints/best_model.pth
    python scripts/evaluate.py --dataset sd1      --data_root /path/to/SD1         --checkpoint checkpoints/best_model.pth
    python scripts/evaluate.py --dataset wtt      --data_root /path/to/WTT         --checkpoint checkpoints/best_model.pth
    python scripts/evaluate.py --dataset all      --data_root /path/to/datasets/   --checkpoint checkpoints/best_model.pth
"""

import argparse, os, sys, csv, json, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as GS
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from complete_model import AdaptiveEnhancementModel
from config import CONFIG

try:
    import lpips as lpips_lib
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("[warn] lpips not installed — LPIPS will be skipped. pip install lpips")

from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity  as ssim_fn

FILTER_NAMES  = ["Low-light", "Dehazing", "Rain removal", "Illum. norm", "Glare red."]
FILTER_COLORS = ["#378ADD",   "#1D9E75",  "#EF9F27",      "#7F77DD",     "#D85A30"]
WEATHER_NAMES = ["Clear", "Rain", "Fog", "Light Snow", "Glare"]
TIME_NAMES    = ["Dawn", "Day", "Dusk", "Night"]
ILLUM_NAMES   = ["Low", "Medium", "High"]

FIG_CTX = {
    "font.family":    "serif",
    "font.serif":     ["Times New Roman", "DejaVu Serif"],
    "font.size":      11,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "savefig.dpi":    400,
}
IMG_EXTS = {".jpg",".jpeg",".png",".bmp"}


# ── Model ────────────────────────────────────────────────────────────────────
def load_model(ckpt_path, device):
    m = AdaptiveEnhancementModel(
        backbone=CONFIG["model"]["backbone"],
        num_weather_classes=CONFIG["model"]["num_weather_classes"],
        num_time_classes=CONFIG["model"]["num_time_classes"],
        num_illum_classes=CONFIG["model"]["num_illum_classes"],
        feature_dim=CONFIG["model"]["feature_dim"],
        num_filters=CONFIG["model"]["num_filters"],
        pretrained=False).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    try:    m.load_state_dict(ck["model_state_dict"])
    except: m.load_state_dict(ck["model_state_dict"], strict=False)
    m.eval()
    return m


def run_model(model, inp_pil, device):
    IW, IH = inp_pil.size
    PW = ((IW+3)//4)*4; PH = ((IH+3)//4)*4
    pad = Image.new("RGB",(PW,PH)); pad.paste(inp_pil,(0,0))
    t = transforms.ToTensor()(pad).unsqueeze(0).to(device)
    with torch.no_grad(): out = model(t)
    enh = out["enhanced"][0].cpu().numpy().transpose(1,2,0)[:IH,:IW]
    fw  = out["filter_weights"][0].cpu().numpy()
    H   = out["weight_entropy"][0].item()
    wl  = out["weather_logits"][0].argmax().item()
    return np.clip(enh,0,1), fw, H, wl


# ── Pair finders ─────────────────────────────────────────────────────────────
def find_lol_pairs(root):
    pairs = []
    for r,dirs,_ in os.walk(root):
        dc = {d.lower():d for d in dirs}
        if "low" in dc and "high" in dc:
            ld=os.path.join(r,dc["low"]); hd=os.path.join(r,dc["high"])
            hlut={os.path.splitext(f)[0]:os.path.join(hd,f)
                  for f in os.listdir(hd) if os.path.splitext(f)[1].lower() in IMG_EXTS}
            for f in sorted(os.listdir(ld)):
                if os.path.splitext(f)[1].lower() not in IMG_EXTS: continue
                st=os.path.splitext(f)[0]
                if st in hlut: pairs.append((os.path.join(ld,f),hlut[st]))
    return pairs


def find_reside_pairs(root):
    pairs = []
    for r,dirs,_ in os.walk(root):
        dc={d.lower():d for d in dirs}
        if "hazy" in dc and "clear" in dc:
            hd=os.path.join(r,dc["hazy"]); cd=os.path.join(r,dc["clear"])
            clut={os.path.splitext(f)[0]:os.path.join(cd,f)
                  for f in os.listdir(cd) if os.path.splitext(f)[1].lower() in IMG_EXTS}
            for f in sorted(os.listdir(hd)):
                if os.path.splitext(f)[1].lower() not in IMG_EXTS: continue
                st=os.path.splitext(f)[0].split("_")[0]
                if st in clut: pairs.append((os.path.join(hd,f),clut[st]))
            break
    return pairs


def find_rain_pairs(root):
    import re
    pairs = []
    for r,dirs,_ in os.walk(root):
        if r[len(root):].count(os.sep)>3: continue
        dc={d.lower():d for d in dirs}
        if "rain" in dc:
            for cn in ["norain","clean","clear","gt"]:
                if cn in dc:
                    rd=os.path.join(r,dc["rain"]); cd=os.path.join(r,dc[cn])
                    clut={os.path.splitext(f)[0]:os.path.join(cd,f)
                          for f in os.listdir(cd) if os.path.splitext(f)[1].lower() in IMG_EXTS}
                    for f in sorted(os.listdir(rd)):
                        if os.path.splitext(f)[1].lower() not in IMG_EXTS: continue
                        st=os.path.splitext(f)[0]
                        gt=(clut.get(st) or clut.get(st[:-2] if st.endswith("x2") else "")
                            or clut.get(re.sub(r"x[0-9]+$","",st)))
                        if gt: pairs.append((os.path.join(rd,f),gt))
                    return pairs
    return pairs


def find_didmdn_pairs(root):
    for sp in ["test","Test"]:
        rd=os.path.join(root,sp,"rain")
        for cn in ["clean","clear"]:
            cd=os.path.join(root,sp,cn)
            if os.path.isdir(rd) and os.path.isdir(cd):
                return find_rain_pairs_from_dirs(rd,cd)
    return []


def find_rain_pairs_from_dirs(rain_d, clean_d):
    import re
    clut={os.path.splitext(f)[0]:os.path.join(clean_d,f)
          for f in os.listdir(clean_d) if os.path.splitext(f)[1].lower() in IMG_EXTS}
    pairs=[]
    for f in sorted(os.listdir(rain_d)):
        if os.path.splitext(f)[1].lower() not in IMG_EXTS: continue
        st=os.path.splitext(f)[0]
        gt=(clut.get(st) or clut.get(st[:-2] if st.endswith("x2") else "")
            or clut.get(re.sub(r"x[0-9]+$","",st)))
        if gt: pairs.append((os.path.join(rain_d,f),gt))
    return pairs


def find_sd1_pairs(root):
    pairs=[]
    exts=tuple(IMG_EXTS)
    for sub in ["test","Test","val","Val",""]:
        d=os.path.join(root,sub) if sub else root
        if not os.path.isdir(d): continue
        gts=[f for f in os.listdir(d) if "_gt." in f.lower() and f.lower().endswith(exts)]
        for gf in gts:
            stem=gf.lower().split("_gt")[0]
            for ext in exts:
                lf=os.path.join(d,stem+"_light"+ext)
                if os.path.exists(lf):
                    pairs.append((lf,os.path.join(d,gf))); break
        if pairs: break
    return pairs


def find_wtt_pairs(root):
    pairs=[]
    td=os.path.join(root,"test","images"); cd=os.path.join(root,"test","clean_images")
    if os.path.isdir(td) and os.path.isdir(cd):
        for f in sorted(os.listdir(td)):
            if os.path.splitext(f)[1].lower() not in IMG_EXTS: continue
            cp=os.path.join(cd,f)
            if os.path.exists(cp): pairs.append((os.path.join(td,f),cp))
    return pairs


DATASET_FINDERS = {
    "lol":     find_lol_pairs,
    "reside":  find_reside_pairs,
    "rain100l":find_rain_pairs,
    "rain100h":find_rain_pairs,
    "didmdn":  find_didmdn_pairs,
    "sd1":     find_sd1_pairs,
    "wtt":     find_wtt_pairs,
}
DATASET_LABELS = {
    "lol":"LOL eval15","reside":"RESIDE-ITS",
    "rain100l":"Rain100L","rain100h":"Rain100H",
    "didmdn":"DID-MDN","sd1":"SD1 glare","wtt":"WTT Synthetic",
}


# ── Evaluation core ──────────────────────────────────────────────────────────
def evaluate_dataset(model, pairs, label, device, lpips_fn=None,
                     max_n=200, n_vis=4, out_dir=None):
    if not pairs:
        print(f"  {label}: no pairs found"); return None

    import random; random.shuffle(pairs)
    if max_n: pairs=pairs[:max_n]

    psnrs_i=[]; ssims_i=[]
    psnrs_e=[]; ssims_e=[]; lpips_e=[]
    fw_all=[]; H_all=[]; samples=[]
    n_err=0

    print(f"\n  Evaluating {label}: {len(pairs)} pairs")
    for idx,(ip,gp) in enumerate(tqdm(pairs,desc=f"  {label}",leave=False)):
        try:
            inp_pil=Image.open(ip).convert("RGB")
            gt_pil =Image.open(gp).convert("RGB")
            if inp_pil.size!=gt_pil.size:
                gt_pil=gt_pil.resize(inp_pil.size,Image.LANCZOS)
            inp_np=np.array(inp_pil).astype(np.float32)/255.0
            gt_np =np.array(gt_pil ).astype(np.float32)/255.0

            enh_np,fw,H,wl=run_model(model,inp_pil,device)

            pi=psnr_fn(gt_np,inp_np,data_range=1.0)
            si=ssim_fn(gt_np,inp_np,data_range=1.0,channel_axis=2)
            pe=psnr_fn(gt_np,enh_np,data_range=1.0)
            se=ssim_fn(gt_np,enh_np,data_range=1.0,channel_axis=2)

            lp=float("nan")
            if lpips_fn is not None:
                pt=torch.from_numpy(enh_np.copy()).permute(2,0,1).unsqueeze(0).float().to(device)
                gt_t=torch.from_numpy(gt_np.copy()).permute(2,0,1).unsqueeze(0).float().to(device)
                with torch.no_grad(): lp=lpips_fn(pt*2-1,gt_t*2-1).item()

            psnrs_i.append(pi); ssims_i.append(si)
            psnrs_e.append(pe); ssims_e.append(se)
            if not np.isnan(lp): lpips_e.append(lp)
            fw_all.append(fw); H_all.append(H)

            if idx<n_vis:
                samples.append(dict(inp_np=inp_np,enh_np=enh_np,gt_np=gt_np,
                                    fname=os.path.basename(ip),
                                    pi=pi,si=si,pe=pe,se=se,lp=lp,
                                    fw=fw,H=H,wl=wl))
        except Exception as e:
            n_err+=1
            if n_err<=3: tqdm.write(f"  [skip {idx}] {os.path.basename(ip)}: {e}")

    if not psnrs_e:
        print(f"  {label}: 0 pairs succeeded"); return None

    mean_fw=np.mean(fw_all,axis=0)
    result=dict(
        label=label, n=len(psnrs_e),
        psnr_inp=round(float(np.mean(psnrs_i)),2),
        psnr_enh=round(float(np.mean(psnrs_e)),2),
        ssim_inp=round(float(np.mean(ssims_i)),4),
        ssim_enh=round(float(np.mean(ssims_e)),4),
        lpips=round(float(np.mean(lpips_e)),4) if lpips_e else float("nan"),
        dpsnr=round(float(np.mean(psnrs_e))-float(np.mean(psnrs_i)),2),
        mean_fw=mean_fw.tolist(),
        mean_H=round(float(np.mean(H_all)),4),
        dom_filter=FILTER_NAMES[int(np.argmax(mean_fw))],
        samples=samples, n_err=n_err,
    )

    print(f"  {label} ({result['n']} pairs):")
    print(f"    PSNR  : {result['psnr_inp']:.2f} -> {result['psnr_enh']:.2f} dB"
          f"  (dPSNR {result['dpsnr']:+.2f})")
    print(f"    SSIM  : {result['ssim_inp']:.4f} -> {result['ssim_enh']:.4f}")
    if not np.isnan(result["lpips"]):
        print(f"    LPIPS :                    {result['lpips']:.4f}")
    print(f"    H     : {result['mean_H']:.4f}   dominant: {result['dom_filter']}")

    if out_dir:
        _save_eval_figure(result, out_dir)

    return result


# ── Publication figure ────────────────────────────────────────────────────────
def _save_eval_figure(result, out_dir):
    samples=result["samples"]
    n=len(samples)
    if n==0: return
    label=result["label"]

    with matplotlib.rc_context(FIG_CTX):
        col_w=2.5; fig_w=n*col_w+0.9; fig_h=fig_w*0.86
        fig=plt.figure(figsize=(fig_w,fig_h),facecolor="white")
        gs=GS.GridSpec(3,1,figure=fig,height_ratios=[4,4,2.0],
                       hspace=0.14,left=0.09,right=0.97,top=0.93,bottom=0.01)
        gs_top=GS.GridSpecFromSubplotSpec(1,n,subplot_spec=gs[0],wspace=0.04)
        gs_bot=GS.GridSpecFromSubplotSpec(1,n,subplot_spec=gs[1],wspace=0.04)

        for k,s in enumerate(samples):
            dp=s["pe"]-s["pi"]; col="#1a6b1a" if dp>=0 else "#b31b1b"
            ax_i=fig.add_subplot(gs_top[0,k])
            ax_i.imshow(np.clip(s["inp_np"],0,1),interpolation="lanczos"); ax_i.axis("off")
            ax_i.set_title(f"PSNR {s['pi']:.2f} dB,  SSIM {s['si']:.4f}",fontsize=9,pad=4,color="#222")
            if k==0:
                ax_i.text(-0.08,0.5,"Input",transform=ax_i.transAxes,
                          ha="right",va="center",rotation=90,
                          fontsize=11,fontweight="bold",color="#1B3A6B")
            ax_e=fig.add_subplot(gs_bot[0,k])
            ax_e.imshow(np.clip(s["enh_np"],0,1),interpolation="lanczos"); ax_e.axis("off")
            ax_e.set_title(f"PSNR {s['pe']:.2f} dB,  SSIM {s['se']:.4f}\n"
                           f"dPSNR {dp:+.2f} dB,  LPIPS {s['lp']:.4f}",
                           fontsize=9,pad=4,color=col)
            if k==0:
                ax_e.text(-0.08,0.5,"Enhanced",transform=ax_e.transAxes,
                          ha="right",va="center",rotation=90,
                          fontsize=11,fontweight="bold",color="#0D6E6E")
            fn=(s["fname"][:18]+"..") if len(s["fname"])>18 else s["fname"]
            ax_e.text(0.5,-0.04,fn,transform=ax_e.transAxes,ha="center",fontsize=8,color="#aaa")

        ax_f=fig.add_subplot(gs[2]); ax_f.set_xlim(0,1); ax_f.set_ylim(0,1); ax_f.axis("off")
        txt="\n".join([
            f"Dataset: {label}     N = {result['n']}",
            f"PSNR:  {result['psnr_inp']:.2f} dB  ->  {result['psnr_enh']:.2f} dB   (dPSNR {result['dpsnr']:+.2f} dB)",
            f"SSIM:  {result['ssim_inp']:.4f}  ->  {result['ssim_enh']:.4f}",
            f"LPIPS: {result['lpips']:.4f}     H: {result['mean_H']:.4f}     Dominant: {result['dom_filter']}",
        ])
        ax_f.text(0.01,0.97,txt,transform=ax_f.transAxes,va="top",ha="left",
                  fontsize=9,color="#111",linespacing=1.6,
                  bbox=dict(boxstyle="round,pad=0.45",fc="#f8f8f8",ec="#ccc",lw=0.7))
        fw=result["mean_fw"]; x0=0.60; xmax=0.97; ys=np.linspace(0.88,0.10,5); bh=0.095
        ax_f.text(x0,0.97,"Mean ESS blending weights",transform=ax_f.transAxes,
                  va="top",ha="left",fontsize=9,fontweight="bold",color="#333")
        for k,(fn,w,fc) in enumerate(zip(FILTER_NAMES,fw,FILTER_COLORS)):
            bw=w*(xmax-x0-0.12)
            ax_f.add_patch(plt.Rectangle((x0+0.12,ys[k]-bh/2),xmax-x0-0.12,bh,
                           transform=ax_f.transAxes,fc="#eeeeee",ec="none",zorder=1))
            ax_f.add_patch(plt.Rectangle((x0+0.12,ys[k]-bh/2),bw,bh,
                           transform=ax_f.transAxes,fc=fc,ec="none",alpha=0.88,zorder=2))
            ax_f.text(x0+0.11,ys[k],fn,transform=ax_f.transAxes,va="center",ha="right",fontsize=8.5,color="#333")
            ax_f.text(x0+0.12+bw+0.012,ys[k],f"{w*100:.1f}%",transform=ax_f.transAxes,
                      va="center",ha="left",fontsize=8.5,color="#333")

        fig.text(0.5,0.975,f"Figure: EAIM-Net v5 — {label} Enhancement Results",
                 ha="center",va="top",fontsize=12,fontweight="bold",color="#1B3A6B")

        safe=label.replace(" ","_").replace("/","_")
        out_png=os.path.join(out_dir,f"Fig_{safe}.png")
        out_jpg=os.path.join(out_dir,f"Fig_{safe}.jpg")
        fig.savefig(out_png,dpi=400,bbox_inches="tight",facecolor="white",format="png")
        fig.savefig(out_jpg,dpi=400,bbox_inches="tight",facecolor="white",format="jpeg",
                    pil_kwargs={"quality":97,"subsampling":0})
        plt.close()
        print(f"  Fig: {out_png}")


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(results):
    if not results: return
    print(f"\n{'='*92}")
    print(f"  EAIM-Net v5 — Evaluation Summary")
    print(f"{'='*92}")
    print(f"  {'Dataset':<22} {'N':>6} {'PSNR_in':>8} {'PSNR_enh':>9} {'dPSNR':>7}"
          f" {'SSIM_in':>8} {'SSIM_enh':>9} {'LPIPS':>7}")
    print(f"  {'-'*90}")
    for r in results:
        lp=f"{r['lpips']:.4f}" if not np.isnan(r["lpips"]) else "   -"
        print(f"  {r['label']:<22} {r['n']:>6} {r['psnr_inp']:>8.2f} "
              f"{r['psnr_enh']:>9.2f} {r['dpsnr']:>+7.2f} "
              f"{r['ssim_inp']:>8.4f} {r['ssim_enh']:>9.4f} {lp:>7}")
    print(f"{'='*92}")


def save_csv(results, out_path):
    if not results: return
    keys=["label","n","psnr_inp","psnr_enh","dpsnr","ssim_inp","ssim_enh","lpips","mean_H","dom_filter"]
    with open(out_path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader()
        for r in results:
            w.writerow({k:r.get(k,"") for k in keys})
    print(f"  CSV: {out_path}")


def save_latex(results, out_path):
    lines=["\\begin{table}[h]","\\centering",
           "\\caption{EAIM-Net v5 quantitative evaluation results.}",
           "\\label{tab:quant}","\\begin{tabular}{lrrrrrr}","\\hline",
           "Dataset & N & $\\mathrm{PSNR_{in}}$ & $\\mathrm{PSNR_{enh}}$ & "
           "$\\Delta$PSNR & $\\mathrm{SSIM_{enh}}$ & LPIPS \\\\","\\hline"]
    for r in results:
        lp=f"{r['lpips']:.4f}" if not np.isnan(r["lpips"]) else "--"
        lines.append(f"{r['label']} & {r['n']} & {r['psnr_inp']:.2f} & "
                     f"{r['psnr_enh']:.2f} & {r['dpsnr']:+.2f} & "
                     f"{r['ssim_enh']:.4f} & {lp} \\\\")
    lines+=["\\hline","\\end{tabular}","\\end{table}"]
    with open(out_path,"w") as f: f.write("\n".join(lines))
    print(f"  LaTeX: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p=argparse.ArgumentParser(description="EAIM-Net v5 evaluation")
    p.add_argument("--dataset",    required=True,
                   help="lol|reside|rain100l|rain100h|didmdn|sd1|wtt|all")
    p.add_argument("--data_root",  required=True,
                   help="Root of the dataset (or parent folder for --dataset all)")
    p.add_argument("--checkpoint", required=True,
                   help="Path to best_model.pth")
    p.add_argument("--out_dir",    default="results/eval",
                   help="Output directory for figures and CSV")
    p.add_argument("--max_n",      type=int, default=200,
                   help="Max images per dataset (0=all)")
    p.add_argument("--n_vis",      type=int, default=4,
                   help="Samples per figure")
    p.add_argument("--device",     default="auto",
                   choices=["auto","cuda","cpu"])
    p.add_argument("--no_lpips",   action="store_true",
                   help="Skip LPIPS (faster)")
    args=p.parse_args()

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device=="auto" else torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nEAIM-Net v5 — Evaluation")
    print(f"{'='*50}")
    print(f"  Dataset    : {args.dataset}")
    print(f"  Data root  : {args.data_root}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Device     : {device}")

    model=load_model(args.checkpoint, device)

    lpips_fn=None
    if HAS_LPIPS and not args.no_lpips:
        lpips_fn=lpips_lib.LPIPS(net="alex").to(device)

    # Which datasets to run
    if args.dataset=="all":
        ds_list=list(DATASET_FINDERS.keys())
    else:
        ds_list=[args.dataset]

    results=[]
    for ds in ds_list:
        if ds not in DATASET_FINDERS:
            print(f"  Unknown dataset: {ds}"); continue

        # Resolve data root
        if args.dataset=="all":
            root=os.path.join(args.data_root, ds)
            if not os.path.isdir(root):
                print(f"  {ds}: folder not found ({root})"); continue
        else:
            root=args.data_root

        pairs=DATASET_FINDERS[ds](root)
        max_n=args.max_n if args.max_n>0 else None
        r=evaluate_dataset(model, pairs, DATASET_LABELS[ds],
                           device, lpips_fn,
                           max_n=max_n, n_vis=args.n_vis,
                           out_dir=args.out_dir)
        if r: results.append(r)

    print_summary(results)
    if results:
        save_csv(results,    os.path.join(args.out_dir,"metrics.csv"))
        save_latex(results,  os.path.join(args.out_dir,"metrics_table.tex"))
        with open(os.path.join(args.out_dir,"metrics.json"),"w") as f:
            json.dump([{k:v for k,v in r.items() if k!="samples"}
                       for r in results], f, indent=2)
        print(f"\n  All outputs saved to: {args.out_dir}")


if __name__=="__main__":
    main()

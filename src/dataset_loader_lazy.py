"""
dataset_loader_lazy.py — EAIM-Net v5 Final

Smart on-demand dataset loader.
Each training cell extracts only the datasets it needs,
trains the filter, then optionally deletes extracted files
to free local disk space before the next cell runs.

At any point only ONE dataset group is on local disk.
Maximum disk usage:
  LOL          ~  0.3 GB
  RESIDE_SOTS  ~  0.8 GB
  Rain (all 3) ~  1.5 GB
  SD1          ~ 10.0 GB   (largest - runs alone)
  WTT          ~  0.2 GB

Usage in each filter cell:
    from dataset_loader_lazy import load_for_filter, cleanup_after_filter
    paths = load_for_filter("rain", WEATHER_DIR)
    # ... train ...
    cleanup_after_filter("rain")   # optional: free disk before next cell
"""

import os, glob, zipfile, shutil, time
from typing import Dict, List, Optional

# ── Where things extract to (local Colab disk, free 100 GB) ──
LOCAL_BASE = "/content/datasets"

# ── Extraction registry ───────────────────────────────────────
# filter_name -> list of (zip_name_variants, extract_subdir, validation_subdir)
FILTER_ZIPS = {
    "lowlight": [
        (["LOL.zip","lol.zip","LOL_dataset.zip"], "LOL", ["our485","eval15","train"]),
    ],
    "dehazing": [
        (["RESIDE_SOTS.zip","RESIDE.zip","reside_sots.zip","RESIDE_ITS.zip"],
         "RESIDE_SOTS", ["hazy","clear","indoor"]),
    ],
    "rain": [
        (["Rain100L.zip","rain100l.zip"],   "Rain100L", ["train","test","rain"]),
        (["Rain100H.zip","rain100h.zip"],   "Rain100H", ["train","test","rain"]),
        (["DID-MDN.zip","did-mdn.zip","DID_MDN.zip","DIDMDN.zip"],
         "DID-MDN",   ["medium_dataset","train"]),
    ],
    "glare": [
        (["SD1.zip","sd1.zip","SD1_dataset.zip"], "SD1", ["train","Train","test","Test"]),
    ],
    "illumnorm": [
        (["weather_time_data.zip","WTT.zip","wtt.zip","synthetic_dataset.zip"],
         "WTT", ["train","val","test"]),
    ],
    "finetune": [
        # Fine-tuning needs LOL + Rain100L for combined loader
        (["LOL.zip","lol.zip"],           "LOL",      ["our485","eval15"]),
        (["Rain100L.zip","rain100l.zip"], "Rain100L", ["train","test"]),
        (["weather_time_data.zip","WTT.zip","wtt.zip"],
         "WTT", ["train","val"]),
    ],
}

# Track what is currently extracted
_EXTRACTED: Dict[str, str] = {}   # subdir_name -> full local path


def _count_imgs(d):
    if not d or not os.path.isdir(d): return 0
    n = 0
    for _,_,fs in os.walk(d):
        n += sum(1 for f in fs if f.lower().endswith((".jpg",".jpeg",".png",".bmp")))
    return n


def _find_zip(weather_dir, candidates):
    if not os.path.isdir(weather_dir): return None
    existing = {f.lower(): os.path.join(weather_dir, f)
                for f in os.listdir(weather_dir) if f.lower().endswith(".zip")}
    for c in candidates:
        if c.lower() in existing:
            return existing[c.lower()]
    return None


def _has_content(local_path, valid_subdirs):
    if not os.path.isdir(local_path): return False
    contents = {d.lower() for d in os.listdir(local_path)}
    return any(v.lower() in contents for v in valid_subdirs)


def _extract(zip_path, target_dir, label=""):
    os.makedirs(target_dir, exist_ok=True)
    tmp = target_dir + "_tmp"
    if os.path.exists(tmp): shutil.rmtree(tmp)
    os.makedirs(tmp)

    t0 = time.time()
    sz_gb = os.path.getsize(zip_path)/1e9
    print(f"  Extracting {os.path.basename(zip_path)} ({sz_gb:.1f} GB)...", end="", flush=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)
        items = os.listdir(tmp)
        # Single top-level folder → lift its contents
        if len(items) == 1 and os.path.isdir(os.path.join(tmp, items[0])):
            inner = os.path.join(tmp, items[0])
            for item in os.listdir(inner):
                dst = os.path.join(target_dir, item)
                if not os.path.exists(dst):
                    shutil.move(os.path.join(inner, item), dst)
        else:
            for item in items:
                dst = os.path.join(target_dir, item)
                if not os.path.exists(dst):
                    shutil.move(os.path.join(tmp, item), dst)
        shutil.rmtree(tmp, ignore_errors=True)
        print(f" done ({time.time()-t0:.0f}s, {_count_imgs(target_dir)} images)")
        return True
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f" FAILED: {e}")
        return False


def _resolve_didmdn(base):
    """DID-MDN: find folder containing medium_dataset/"""
    for root, dirs, _ in os.walk(base):
        if root[len(base):].count(os.sep) > 3: continue
        if "medium_dataset" in [d.lower() for d in dirs]:
            return os.path.join(root, "medium_dataset")
    return base


def load_for_filter(
    filter_name:  str,
    weather_dir:  str,
    sd1_source:   Optional[str] = None,   # alternate path for SD1 (second drive)
    local_base:   str = LOCAL_BASE,
    force:        bool = False,
) -> Dict[str, Optional[str]]:
    """
    Extract only the datasets needed for this filter.

    Parameters
    ----------
    filter_name : "lowlight", "dehazing", "rain", "glare", "illumnorm", "finetune"
    weather_dir : folder containing zip files
    sd1_source  : alternate path for SD1.zip (second drive / shared folder)
    local_base  : where to extract (default /content/datasets)
    force       : re-extract even if already present

    Returns
    -------
    dict with keys matching the filter:
      lowlight  -> {"lol": path}
      dehazing  -> {"reside": path}
      rain      -> {"rain100l": path, "rain100h": path, "didmdn": path}
      glare     -> {"sd1": path}
      illumnorm -> {"wtt": path}
      finetune  -> {"lol": path, "rain100l": path, "wtt": path}
    """
    global _EXTRACTED
    os.makedirs(local_base, exist_ok=True)

    filter_name = filter_name.lower()
    if filter_name not in FILTER_ZIPS:
        raise ValueError(f"Unknown filter: {filter_name}. "
                         f"Choose from: {list(FILTER_ZIPS.keys())}")

    specs = FILTER_ZIPS[filter_name]
    result = {}

    print(f"\n{'='*60}")
    print(f"  Loading datasets for [{filter_name}]")
    print(f"{'='*60}")

    for candidates, subdir, valid_subs in specs:
        local_path = os.path.join(local_base, subdir)
        n_existing = _count_imgs(local_path)
        has_content = _has_content(local_path, valid_subs)

        if has_content and n_existing > 0 and not force:
            print(f"  OK (cached)  {subdir:<18} {n_existing} images")
            key = _subdir_to_key(subdir)
            path = _resolve_path(subdir, local_path)
            result[key] = path
            _EXTRACTED[subdir] = local_path
            continue

        # Find the zip
        zip_path = None
        if subdir == "SD1" and sd1_source:
            # SD1 from alternate source
            if os.path.isfile(sd1_source):   zip_path = sd1_source
            elif os.path.isdir(sd1_source):
                # Already a folder
                print(f"  OK (dir)     {subdir:<18} {_count_imgs(sd1_source)} images")
                result["sd1"] = sd1_source; _EXTRACTED[subdir] = sd1_source; continue
        if zip_path is None:
            zip_path = _find_zip(weather_dir, candidates)

        if zip_path is None:
            print(f"  MISSING      {subdir:<18} no zip found  (tried: {candidates[0]}...)")
            result[_subdir_to_key(subdir)] = None
            continue

        if force and os.path.isdir(local_path):
            shutil.rmtree(local_path)

        ok = _extract(zip_path, local_path, subdir)
        if ok:
            key  = _subdir_to_key(subdir)
            path = _resolve_path(subdir, local_path)
            result[key] = path
            _EXTRACTED[subdir] = local_path
        else:
            result[_subdir_to_key(subdir)] = None

    print(f"{'='*60}")
    print(f"  Ready: {[k for k,v in result.items() if v]}")
    return result


def cleanup_after_filter(filter_name: str, local_base: str = LOCAL_BASE):
    """
    Delete extracted dataset files after a filter finishes training.
    Frees local disk before the next (large) dataset is extracted.

    Call at the END of each pre-training cell:
        cleanup_after_filter("rain")   # frees Rain100L + Rain100H + DID-MDN
    """
    global _EXTRACTED
    specs = FILTER_ZIPS.get(filter_name.lower(), [])
    freed = 0
    print(f"\nCleaning up [{filter_name}] datasets from local disk...")
    for candidates, subdir, _ in specs:
        local_path = os.path.join(local_base, subdir)
        if os.path.isdir(local_path):
            size_gb = sum(
                os.path.getsize(os.path.join(r,f))
                for r,_,fs in os.walk(local_path) for f in fs
            ) / 1e9
            shutil.rmtree(local_path)
            _EXTRACTED.pop(subdir, None)
            print(f"  Deleted {subdir}  ({size_gb:.1f} GB freed)")
            freed += size_gb
        else:
            print(f"  {subdir} not on local disk — skipped")
    print(f"  Total freed: {freed:.1f} GB")


def disk_usage(local_base: str = LOCAL_BASE):
    """Print current local disk usage by dataset."""
    print(f"\nLocal disk usage ({local_base}):")
    if not os.path.isdir(local_base):
        print("  (empty)")
        return
    total = 0
    for d in sorted(os.listdir(local_base)):
        full = os.path.join(local_base, d)
        if not os.path.isdir(full): continue
        sz = sum(os.path.getsize(os.path.join(r,f))
                 for r,_,fs in os.walk(full) for f in fs) / 1e9
        n  = _count_imgs(full)
        print(f"  {d:<20} {sz:>6.1f} GB  {n:>6} images")
        total += sz
    print(f"  {'TOTAL':<20} {total:>6.1f} GB")


def _subdir_to_key(subdir):
    mapping = {
        "LOL": "lol", "RESIDE_SOTS": "reside",
        "Rain100L": "rain100l", "Rain100H": "rain100h",
        "DID-MDN": "didmdn", "SD1": "sd1", "WTT": "wtt",
    }
    return mapping.get(subdir, subdir.lower())


def _resolve_path(subdir, local_path):
    """Handle special cases like DID-MDN needing medium_dataset subfolder."""
    if subdir == "DID-MDN":
        return _resolve_didmdn(local_path)
    return local_path

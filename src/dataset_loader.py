"""
dataset_loader.py — EAIM-Net v5 Final

Auto-detects and extracts dataset zip files from a single Weather folder.

Usage (Colab):
    from dataset_loader import load_all_datasets, PATHS
    load_all_datasets(weather_dir="/content/drive/MyDrive/Weather")
    # PATHS is then populated with all dataset roots
    # Use PATHS["lol"], PATHS["reside"], PATHS["rain100l"], etc.

Expected zip files in Weather/:
    LOL.zip           → extracts to Weather/LOL/
    RESIDE_SOTS.zip   → extracts to Weather/RESIDE_SOTS/
    Rain100L.zip      → extracts to Weather/Rain100L/
    Rain100H.zip      → extracts to Weather/Rain100H/
    DID-MDN.zip       → extracts to Weather/DID-MDN/
    SD1.zip           → extracts to Weather/SD1/
    weather_time_data.zip → extracts to /content/weather_time_data/

Zip files can have any capitalisation — matching is case-insensitive.
Already-extracted folders are skipped (no re-extraction).
Prints a clear status table showing what was found, extracted, and skipped.
"""

import os
import zipfile
import glob
import shutil
import time
from typing import Dict, Optional

# ── Canonical paths populated by load_all_datasets() ─────────────────────────
PATHS: Dict[str, Optional[str]] = {
    "lol":      None,
    "reside":   None,
    "rain100l": None,
    "rain100h": None,
    "didmdn":   None,
    "sd1":      None,
    "wtt":      None,
}

# ── Dataset definitions ────────────────────────────────────────────────────────
# Each entry: (PATHS key, zip name variants, expected extracted folder name,
#              validation subfolder that must exist after extraction)
_DATASETS = [
    (
        "lol",
        ["LOL.zip", "lol.zip", "LOL_dataset.zip", "lol_dataset.zip"],
        "LOL",
        ["our485", "eval15", "train", "test"],   # at least one must exist
    ),
    (
        "reside",
        ["RESIDE_SOTS.zip", "RESIDE.zip", "reside_sots.zip", "RESIDE_ITS.zip", "reside.zip"],
        "RESIDE_SOTS",
        ["hazy", "clear", "indoor", "outdoor"],
    ),
    (
        "rain100l",
        ["Rain100L.zip", "rain100l.zip", "Rain100L_dataset.zip"],
        "Rain100L",
        ["train", "test", "rain"],
    ),
    (
        "rain100h",
        ["Rain100H.zip", "rain100h.zip", "Rain100H_dataset.zip"],
        "Rain100H",
        ["train", "test", "rain"],
    ),
    (
        "didmdn",
        ["DID-MDN.zip", "did-mdn.zip", "DID_MDN.zip", "DIDMDN.zip", "didmdn.zip"],
        "DID-MDN",
        ["medium_dataset", "train"],
    ),
    (
        "sd1",
        ["SD1.zip", "sd1.zip", "SD1_dataset.zip"],
        "SD1",
        ["train", "Train", "test", "Test", "val"],
    ),
]

_WTT_ZIPS = [
    "weather_time_data.zip",
    "Weather_Time_Data.zip",
    "WTT.zip",
    "wtt.zip",
    "synthetic_dataset.zip",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _count_images(d: str) -> int:
    if not d or not os.path.isdir(d):
        return 0
    n = 0
    for _, _, fs in os.walk(d):
        n += sum(1 for f in fs if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
    return n


def _find_zip(search_dir: str, candidates: list) -> Optional[str]:
    """Case-insensitive zip file search."""
    existing = {f.lower(): f for f in os.listdir(search_dir)
                if f.lower().endswith(".zip")}
    for c in candidates:
        if c.lower() in existing:
            return os.path.join(search_dir, existing[c.lower()])
    return None


def _has_subdir(base: str, names: list) -> bool:
    """Return True if any of the listed subdirectory names exist under base."""
    if not os.path.isdir(base):
        return False
    contents = {d.lower() for d in os.listdir(base)}
    return any(n.lower() in contents for n in names)


def _extract_zip(zip_path: str, target_dir: str, label: str) -> bool:
    """
    Extract zip_path to target_dir.
    Handles two layouts:
      (a) Zip contains a single top-level folder → move its contents
      (b) Zip contains files directly → extract to target_dir
    Returns True on success.
    """
    os.makedirs(target_dir, exist_ok=True)
    tmp_dir = target_dir + "_tmp_extract"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        print(f"  Extracting {os.path.basename(zip_path)} ...", end="", flush=True)
        t0 = time.time()
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)
        elapsed = time.time() - t0

        # Check if there is a single top-level folder
        tmp_contents = os.listdir(tmp_dir)
        if len(tmp_contents) == 1 and os.path.isdir(os.path.join(tmp_dir, tmp_contents[0])):
            inner = os.path.join(tmp_dir, tmp_contents[0])
            # If the folder name matches target exactly, move contents up
            for item in os.listdir(inner):
                src = os.path.join(inner, item)
                dst = os.path.join(target_dir, item)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
        else:
            # Multiple items — move all to target_dir
            for item in tmp_contents:
                src = os.path.join(tmp_dir, item)
                dst = os.path.join(target_dir, item)
                if not os.path.exists(dst):
                    shutil.move(src, dst)

        shutil.rmtree(tmp_dir, ignore_errors=True)
        n = _count_images(target_dir)
        print(f" done ({elapsed:.0f}s, {n} images)")
        return True

    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f" FAILED: {e}")
        return False


def _deep_search(base: str, target_name: str) -> Optional[str]:
    """
    If extraction created a nested folder (e.g. LOL/LOL/our485),
    search up to 3 levels deep for the correct root.
    """
    if not os.path.isdir(base):
        return None
    # Level 0: base itself
    if os.path.basename(base).lower() == target_name.lower():
        return base
    # Level 1-3
    for root, dirs, _ in os.walk(base):
        depth = root[len(base):].count(os.sep)
        if depth > 3:
            continue
        for d in dirs:
            if d.lower() == target_name.lower():
                return os.path.join(root, d)
    return base  # fallback


def _resolve_didmdn(base: str) -> str:
    """DID-MDN may extract as DID-MDN/ or contain medium_dataset/ at various depths."""
    for root, dirs, _ in os.walk(base):
        depth = root[len(base):].count(os.sep)
        if depth > 3:
            continue
        if "medium_dataset" in [d.lower() for d in dirs]:
            # Return the parent of medium_dataset
            return root
    return base


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def load_all_datasets(
    weather_dir: str = "/content/drive/MyDrive/Weather",
    wtt_extract_to: str = "/content/weather_time_data",
    force_reextract: bool = False,
    skip_if_extracted: bool = True,
) -> Dict[str, Optional[str]]:
    """
    Auto-detect, extract, and register all dataset zip files.

    Parameters
    ----------
    weather_dir      : folder containing all .zip files
    wtt_extract_to   : where to extract the WTT synthetic dataset
                       (local /content is faster than Drive for training)
    force_reextract  : re-extract even if folder already exists
    skip_if_extracted: skip zip extraction if images already present

    Returns
    -------
    PATHS dict with keys: lol, reside, rain100l, rain100h, didmdn, sd1, wtt
    Each value is the path string or None if not found.
    """
    global PATHS

    if not os.path.isdir(weather_dir):
        print(f"ERROR: Weather folder not found: {weather_dir}")
        print("Create the folder and upload your zip files to it.")
        return PATHS

    print("\n" + "=" * 68)
    print("  EAIM-Net v5 — Dataset Auto-Loader")
    print(f"  Weather folder: {weather_dir}")
    print("=" * 68)

    # List available zips
    available_zips = [f for f in os.listdir(weather_dir) if f.lower().endswith(".zip")]
    if available_zips:
        print(f"\n  Zip files found ({len(available_zips)}):")
        for z in sorted(available_zips):
            sz_mb = os.path.getsize(os.path.join(weather_dir, z)) / 1e6
            print(f"    {z:<35} {sz_mb:>8.1f} MB")
    else:
        print("  No zip files found in Weather folder.")

    print()

    # ── Process each dataset zip ───────────────────────────────────────────────
    for key, zip_candidates, folder_name, validation_subdirs in _DATASETS:
        target_dir = os.path.join(weather_dir, folder_name)
        zip_path   = _find_zip(weather_dir, zip_candidates)
        n_existing = _count_images(target_dir)
        has_valid  = _has_subdir(target_dir, validation_subdirs)

        if has_valid and n_existing > 0 and not force_reextract:
            # Already extracted — just register
            resolved = _deep_search(target_dir, folder_name)
            if key == "didmdn":
                resolved = _resolve_didmdn(target_dir)
                # DID-MDN expects root to be the folder containing medium_dataset/
                if os.path.isdir(os.path.join(resolved, "medium_dataset")):
                    resolved = os.path.join(resolved, "medium_dataset")
            PATHS[key] = resolved
            print(f"  {'OK':<6} {folder_name:<18} {n_existing:>6} images  (already extracted)")

        elif zip_path:
            if force_reextract and os.path.isdir(target_dir):
                shutil.rmtree(target_dir)
            success = _extract_zip(zip_path, target_dir, folder_name)
            if success:
                resolved = _deep_search(target_dir, folder_name)
                if key == "didmdn":
                    resolved = _resolve_didmdn(target_dir)
                    if os.path.isdir(os.path.join(resolved, "medium_dataset")):
                        resolved = os.path.join(resolved, "medium_dataset")
                n_now = _count_images(resolved or target_dir)
                PATHS[key] = resolved or target_dir
                print(f"  {'OK':<6} {folder_name:<18} {n_now:>6} images  (extracted from zip)")
            else:
                PATHS[key] = None
                print(f"  {'FAIL':<6} {folder_name:<18}        extraction failed")

        else:
            PATHS[key] = None
            status = "folder empty" if os.path.isdir(target_dir) else "no zip found"
            print(f"  {'MISS':<6} {folder_name:<18}        {status}")

    # ── WTT synthetic dataset ──────────────────────────────────────────────────
    wtt_zip = _find_zip(weather_dir, _WTT_ZIPS)

    # Also check if WTT is already extracted to /content
    wtt_local_ok = (
        os.path.isdir(wtt_extract_to) and
        os.path.isdir(os.path.join(wtt_extract_to, "train")) and
        _count_images(os.path.join(wtt_extract_to, "train")) > 0
    )

    if wtt_local_ok and not force_reextract:
        n_wtt = _count_images(wtt_extract_to)
        PATHS["wtt"] = wtt_extract_to
        print(f"  {'OK':<6} {'WTT Synthetic':<18} {n_wtt:>6} images  (already at {wtt_extract_to})")

    elif wtt_zip:
        print(f"  Extracting WTT to {wtt_extract_to} (local disk = faster training)...")
        os.makedirs(wtt_extract_to, exist_ok=True)
        success = _extract_zip(wtt_zip, wtt_extract_to, "weather_time_data")
        if success:
            n_wtt = _count_images(wtt_extract_to)
            PATHS["wtt"] = wtt_extract_to
            print(f"  {'OK':<6} {'WTT Synthetic':<18} {n_wtt:>6} images  (extracted to {wtt_extract_to})")
        else:
            PATHS["wtt"] = None

    else:
        PATHS["wtt"] = None
        print(f"  {'MISS':<6} {'WTT Synthetic':<18}        no zip found  (tried: {', '.join(_WTT_ZIPS[:2])}...)")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  PATH SUMMARY")
    print("=" * 68)
    path_labels = {
        "lol":      "LOL (low-light)",
        "reside":   "RESIDE_SOTS (haze)",
        "rain100l": "Rain100L (light rain)",
        "rain100h": "Rain100H (heavy rain)",
        "didmdn":   "DID-MDN (medium rain)",
        "sd1":      "SD1 (glare)",
        "wtt":      "WTT Synthetic (all)",
    }
    n_ready = 0
    for key, label in path_labels.items():
        p = PATHS[key]
        if p and os.path.isdir(p):
            n = _count_images(p)
            print(f"  {'OK':<4}  {label:<26}  {n:>6} images  {p}")
            n_ready += 1
        else:
            print(f"  MISS  {label:<26}  (not available)")
    print("=" * 68)
    print(f"  {n_ready}/{len(PATHS)} datasets ready")
    if n_ready < len(PATHS):
        missing = [path_labels[k] for k, p in PATHS.items() if not p or not os.path.isdir(p)]
        print(f"  Missing: {', '.join(missing)}")
        print("  Upload the corresponding .zip files to the Weather folder and re-run.")
    print("=" * 68)

    return PATHS


def get_loader_kwargs(paths: Dict = None) -> Dict:
    """
    Return keyword arguments for create_combined_dataloaders().
    Call after load_all_datasets().

    Usage:
        kwargs = get_loader_kwargs()
        loaders = create_combined_dataloaders(**kwargs, batch_size=4, image_size=512)
    """
    p = paths or PATHS
    return {
        "synthetic_data_root": p.get("wtt"),
        "lol_root":            p.get("lol"),
        "rain100l_root":       p.get("rain100l"),
        "rain100h_root":       p.get("rain100h"),
    }


def print_paths():
    """Print current PATHS dict."""
    print("\nCurrent dataset paths:")
    for k, v in PATHS.items():
        print(f"  PATHS['{k}'] = {v!r}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weather_dir",    default="/content/drive/MyDrive/Weather")
    parser.add_argument("--wtt_extract_to", default="/content/weather_time_data")
    parser.add_argument("--force",          action="store_true")
    args = parser.parse_args()
    load_all_datasets(args.weather_dir, args.wtt_extract_to, force_reextract=args.force)


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-DRIVE SUPPORT
# Use when SD1 (or any large dataset) is on a different Google Drive account.
# ══════════════════════════════════════════════════════════════════════════════

def load_sd1_from_second_drive(
    sd1_zip_or_dir: str,
    extract_to: str = "/content/SD1",
    force_reextract: bool = False,
) -> Optional[str]:
    """
    Load SD1 from a second Google Drive account or any alternate path.

    Three use cases:
      A. Second Drive mounted at /content/drive2:
            load_sd1_from_second_drive("/content/drive2/MyDrive/SD1.zip")

      B. Shared Drive shortcut already in main Drive:
            load_sd1_from_second_drive("/content/drive/MyDrive/Weather/SD1.zip")
            (same as normal loading — just call load_all_datasets normally)

      C. Already a folder (no zip needed):
            load_sd1_from_second_drive("/content/drive2/MyDrive/SD1")

    Parameters
    ----------
    sd1_zip_or_dir : path to SD1.zip OR path to already-extracted SD1 folder
    extract_to     : where to extract (default: /content/SD1 — local disk,
                     free 100 GB, much faster than Drive for training)
    force_reextract: re-extract even if folder already exists

    Returns
    -------
    Path to SD1 root directory, or None on failure.
    Updates PATHS["sd1"] automatically.
    """
    global PATHS

    print("\n" + "="*60)
    print("  SD1 Dataset Loader (alternate drive / path)")
    print("="*60)

    # Case C: already a directory
    if os.path.isdir(sd1_zip_or_dir):
        n = _count_images(sd1_zip_or_dir)
        print(f"  SD1 directory found: {sd1_zip_or_dir}  ({n} images)")
        PATHS["sd1"] = sd1_zip_or_dir
        return sd1_zip_or_dir

    # Case A/B: zip file
    if not os.path.exists(sd1_zip_or_dir):
        print(f"  ERROR: Not found: {sd1_zip_or_dir}")
        print("  Check the path and that the Drive is mounted.")
        return None

    # Check if already extracted to local disk
    n_existing = _count_images(extract_to)
    has_content = any(
        os.path.isdir(os.path.join(extract_to, sub))
        for sub in ["train", "Train", "test", "Test", "val"]
        if os.path.isdir(extract_to)
    )

    if has_content and n_existing > 0 and not force_reextract:
        print(f"  SD1 already extracted: {extract_to}  ({n_existing} images)")
        PATHS["sd1"] = extract_to
        return extract_to

    zip_size_gb = os.path.getsize(sd1_zip_or_dir) / 1e9
    print(f"  Zip: {sd1_zip_or_dir}  ({zip_size_gb:.1f} GB)")
    print(f"  Extracting to local disk: {extract_to}")
    print(f"  NOTE: This is /content/ (free 100GB local, not Drive)")
    print(f"  Will be lost when Colab session ends — re-run this cell next session")

    success = _extract_zip(sd1_zip_or_dir, extract_to, "SD1")
    if success:
        n = _count_images(extract_to)
        PATHS["sd1"] = extract_to
        print(f"  SD1 ready: {extract_to}  ({n} images)")
        return extract_to
    return None


def mount_second_drive(mount_point: str = "/content/drive2") -> bool:
    """
    Mount a second Google Drive account in Colab.
    When prompted, sign in with your second Google account.

    Usage:
        mount_second_drive()
        load_sd1_from_second_drive("/content/drive2/MyDrive/SD1.zip")

    Parameters
    ----------
    mount_point : where to mount (default: /content/drive2)

    Returns
    -------
    True if mounted successfully.
    """
    try:
        from google.colab import drive as _drive
        print(f"Mounting second Drive at {mount_point} ...")
        print("Sign in with your SECOND Google account when prompted.")
        _drive.mount(mount_point, force_remount=True)
        # Verify
        if os.path.isdir(os.path.join(mount_point, "MyDrive")):
            print(f"  Mounted successfully: {mount_point}/MyDrive")
            return True
        else:
            print(f"  Mount point exists but MyDrive not found. Check authentication.")
            return False
    except Exception as e:
        print(f"  Failed to mount: {e}")
        return False


def load_all_datasets_multi_drive(
    weather_dir:      str,
    sd1_source:       str,
    wtt_extract_to:   str = "/content/weather_time_data",
    sd1_extract_to:   str = "/content/SD1",
    force_reextract:  bool = False,
) -> Dict[str, Optional[str]]:
    """
    Load all datasets where SD1 comes from a different location.

    Parameters
    ----------
    weather_dir    : main Weather folder (has LOL.zip, RESIDE_SOTS.zip, etc.)
    sd1_source     : path to SD1.zip or SD1 folder (can be on second Drive,
                     shared folder, or anywhere Colab can access)
    wtt_extract_to : extract WTT here (local /content)
    sd1_extract_to : extract SD1 here (local /content — saves Drive space)

    Usage example:
        # Option A: SD1 on second Drive account
        mount_second_drive()
        load_all_datasets_multi_drive(
            weather_dir  = "/content/drive/MyDrive/Weather",
            sd1_source   = "/content/drive2/MyDrive/SD1.zip",
        )

        # Option B: SD1 as shared folder shortcut in main Drive
        load_all_datasets_multi_drive(
            weather_dir  = "/content/drive/MyDrive/Weather",
            sd1_source   = "/content/drive/MyDrive/SD1_shared/SD1.zip",
        )

        # Option C: SD1 zip already on local disk
        load_all_datasets_multi_drive(
            weather_dir  = "/content/drive/MyDrive/Weather",
            sd1_source   = "/content/SD1.zip",
        )
    """
    # Load everything except SD1 from main weather dir
    load_all_datasets(
        weather_dir    = weather_dir,
        wtt_extract_to = wtt_extract_to,
        force_reextract= force_reextract,
    )

    # Load SD1 separately
    print("\n--- SD1 (separate source) ---")
    load_sd1_from_second_drive(
        sd1_zip_or_dir = sd1_source,
        extract_to     = sd1_extract_to,
        force_reextract= force_reextract,
    )

    return PATHS

import argparse
import csv
import multiprocessing as mp
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FADE_ROOT = Path("/data/users/gaoyin/2024_LYB/evaluation")
FADE_PACKAGE = FADE_ROOT / "FADE_release_python"
_FADE_FN = None
_MAX_SIDE = 512


def parse_args():
    parser = argparse.ArgumentParser(description="Compute FADE scores and move low-fog images to a backup folder.")
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--csv", type=Path, default=Path("fade_scores_hazy_single.csv"))
    parser.add_argument("--move_dir", type=Path, default=None)
    parser.add_argument("--low_percent", type=float, default=25.0, help="Move the lowest FADE percentile.")
    parser.add_argument("--threshold", type=float, default=None, help="Move images with FADE <= threshold.")
    parser.add_argument("--max_side", type=int, default=512, help="Resize longest side before FADE; set 0 to disable.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes.")
    parser.add_argument("--dry_run", action="store_true", help="Only compute and report; do not move files.")
    return parser.parse_args()


def load_fade():
    if not FADE_PACKAGE.exists():
        raise FileNotFoundError(f"FADE implementation not found: {FADE_PACKAGE}")
    sys.path.insert(0, str(FADE_PACKAGE))
    old_cwd = Path.cwd()
    os.chdir(str(FADE_ROOT))
    try:
        from FADE import FADE
    finally:
        os.chdir(str(old_cwd))
    return FADE


def collect_images(image_dir: Path):
    if not image_dir.is_dir():
        raise NotADirectoryError(image_dir)
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def compute_fade(fade_fn, image_path: Path, max_side: int):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("cv2.imread returned None")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = resize_for_fade(rgb, max_side)
    return compute_fade_rgb(fade_fn, rgb)


def resize_for_fade(rgb: np.ndarray, max_side: int):
    if max_side <= 0:
        return rgb
    h, w = rgb.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return rgb
    scale = max_side / long_side
    new_w = max(8, int(round(w * scale)))
    new_h = max(8, int(round(h * scale)))
    new_w -= new_w % 8
    new_h -= new_h % 8
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def compute_fade_rgb(fade_fn, rgb: np.ndarray):

    old_cwd = Path.cwd()
    os.chdir(str(FADE_ROOT))
    try:
        score = fade_fn(rgb)
    finally:
        os.chdir(str(old_cwd))
    if isinstance(score, (tuple, list)):
        score = score[0]
    return float(np.asarray(score).reshape(-1)[0])


def init_worker(max_side: int):
    global _FADE_FN, _MAX_SIDE
    _FADE_FN = load_fade()
    _MAX_SIDE = max_side


def score_worker(image_path_str: str):
    image_path = Path(image_path_str)
    try:
        score = compute_fade(_FADE_FN, image_path, _MAX_SIDE)
    except Exception as exc:
        return str(image_path), None, f"{type(exc).__name__}: {exc}"
    return str(image_path), score, ""


def write_csv(csv_path: Path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "fade", "selected_low_fog", "status", "error"])
        writer.writerows(rows)


def main():
    args = parse_args()
    images = collect_images(args.image_dir)
    if not images:
        raise RuntimeError(f"No images found in {args.image_dir}")

    scored = []
    rows = []
    image_strings = [str(path) for path in images]
    if args.workers > 1:
        with mp.Pool(processes=args.workers, initializer=init_worker, initargs=(args.max_side,)) as pool:
            iterator = pool.imap_unordered(score_worker, image_strings)
            for image_path_str, score, error in tqdm(iterator, total=len(image_strings), desc="Computing FADE"):
                if error:
                    rows.append([image_path_str, "", False, "failed", error])
                else:
                    scored.append((Path(image_path_str), score))
    else:
        fade_fn = load_fade()
        for image_path in tqdm(images, desc="Computing FADE"):
            try:
                score = compute_fade(fade_fn, image_path, args.max_side)
            except Exception as exc:
                rows.append([str(image_path), "", False, "failed", f"{type(exc).__name__}: {exc}"])
                continue
            scored.append((image_path, score))

    if not scored:
        write_csv(args.csv, rows)
        raise RuntimeError("No valid FADE scores were computed.")

    scores = np.array([score for _, score in scored], dtype=np.float64)
    threshold = args.threshold
    if threshold is None:
        threshold = float(np.percentile(scores, args.low_percent))

    selected = {path for path, score in scored if score <= threshold}
    for image_path, score in scored:
        rows.append([str(image_path), f"{score:.8f}", image_path in selected, "ok", ""])
    write_csv(args.csv, rows)

    print(f"Images: {len(images)}")
    print(f"Valid scores: {len(scored)}")
    print(f"FADE min/mean/max: {scores.min():.6f} / {scores.mean():.6f} / {scores.max():.6f}")
    print(f"Low-fog threshold: FADE <= {threshold:.6f}")
    print(f"Selected low-fog images: {len(selected)}")
    print(f"CSV: {args.csv}")

    if args.dry_run:
        print("Dry run enabled; no files were moved.")
        return

    move_dir = args.move_dir or args.image_dir.parent / f"{args.image_dir.name}_low_fade_removed"
    move_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for image_path, _ in scored:
        if image_path not in selected:
            continue
        target = move_dir / image_path.name
        if target.exists():
            target = move_dir / f"{image_path.stem}__dup{image_path.suffix}"
        shutil.move(str(image_path), str(target))
        moved += 1
    print(f"Moved {moved} images to: {move_dir}")


if __name__ == "__main__":
    main()

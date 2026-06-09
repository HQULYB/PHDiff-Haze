import argparse
import csv
import multiprocessing as mp
import re
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from filter_low_fade import IMAGE_SUFFIXES, init_worker, load_fade, compute_fade, score_worker


ROOT = Path("/data/users/gaoyin/datasets/dehaze")
OUT_ROOT = ROOT / "Haze_Gen" / "Stage1" / "Train"
HAZE4K_MAX_OUTDOOR_ID = 1501


def list_images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def pair_key(path: Path) -> str:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 3:
        try:
            float(parts[-1])
            float(parts[-2])
            return "_".join(parts[:-2])
        except ValueError:
            pass
    return stem


def norm_ntire_key(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"_(GT|hazy)$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_outdoor_(GT|hazy)$", "", stem, flags=re.IGNORECASE)
    return stem


def haze4k_id(path: Path) -> int | None:
    key = pair_key(path)
    try:
        return int(key)
    except ValueError:
        return None


def is_haze4k_outdoor(path: Path) -> bool:
    image_id = haze4k_id(path)
    return image_id is not None and image_id <= HAZE4K_MAX_OUTDOOR_ID


def score_images(images: list[Path], max_side: int, workers: int):
    if workers > 1:
        scored = []
        failed = []
        with mp.Pool(processes=workers, initializer=init_worker, initargs=(max_side,)) as pool:
            iterator = pool.imap_unordered(score_worker, [str(p) for p in images])
            for image_path_str, score, error in tqdm(iterator, total=len(images), desc="Computing Haze4K FADE"):
                image_path = Path(image_path_str)
                if error:
                    failed.append((image_path, error))
                else:
                    scored.append((image_path, score))
    else:
        fade_fn = load_fade()
        scored = []
        failed = []
        for image_path in tqdm(images, desc="Computing Haze4K FADE"):
            try:
                scored.append((image_path, compute_fade(fade_fn, image_path, max_side)))
            except Exception as exc:
                failed.append((image_path, f"{type(exc).__name__}: {exc}"))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored, failed


def copy_pair(clean: Path, hazy: Path, dataset: str, key: str, out_gt: Path, out_hazy: Path):
    clean_target = out_gt / f"{dataset}_{key}{clean.suffix.lower()}"
    hazy_target = out_hazy / f"{dataset}_{key}{hazy.suffix.lower()}"
    shutil.copy2(clean, clean_target)
    shutil.copy2(hazy, hazy_target)
    return clean_target, hazy_target


def add_ots_pairs(rows, out_gt: Path, out_hazy: Path):
    clean_dir = ROOT / "OTS" / "train" / "clear"
    hazy_dir = ROOT / "OTS" / "train" / "hazy_single_top800_fade"
    clean_by_key = {p.stem: p for p in list_images(clean_dir)}
    count = 0
    for hazy in tqdm(list_images(hazy_dir), desc="Copying OTS top800"):
        clean = clean_by_key.get(hazy.stem)
        if clean is None:
            rows.append(["OTS_top800_fade", "", str(hazy), "", "", "missing_clean"])
            continue
        clean_target, hazy_target = copy_pair(clean, hazy, "ots", hazy.stem, out_gt, out_hazy)
        rows.append(["OTS_top800_fade", hazy.stem, str(clean), str(hazy), str(clean_target), str(hazy_target), "ok"])
        count += 1
    return count


def add_haze4k_pairs(rows, out_gt: Path, out_hazy: Path, top_k: int, max_side: int, workers: int, fade_csv: Path):
    clean_dir = ROOT / "Haze4K" / "train" / "gt"
    hazy_dir = ROOT / "Haze4K" / "train" / "haze"
    clean_by_key = {p.stem: p for p in list_images(clean_dir)}
    hazy_images = [p for p in list_images(hazy_dir) if is_haze4k_outdoor(p)]
    scored, failed = score_images(hazy_images, max_side, workers)
    selected = scored[:top_k]
    fade_csv.parent.mkdir(parents=True, exist_ok=True)
    selected_set = {p for p, _ in selected}
    with fade_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "image", "fade", "selected_top_fog", "status", "error"])
        for rank, (image_path, score) in enumerate(scored, start=1):
            writer.writerow([rank, str(image_path), f"{score:.8f}", image_path in selected_set, "ok", ""])
        for image_path, error in failed:
            writer.writerow(["", str(image_path), "", False, "failed", error])

    count = 0
    for hazy, score in tqdm(selected, desc=f"Copying Haze4K top{top_k}"):
        key = pair_key(hazy)
        clean = clean_by_key.get(key)
        if clean is None:
            rows.append(["Haze4K_top_fade", key, "", str(hazy), "", "", f"missing_clean_fade={score:.8f}"])
            continue
        clean_target, hazy_target = copy_pair(clean, hazy, "haze4k", key, out_gt, out_hazy)
        rows.append(["Haze4K_top_fade", key, str(clean), str(hazy), str(clean_target), str(hazy_target), f"ok_fade={score:.8f}"])
        count += 1

    if selected:
        scores = np.array([score for _, score in scored], dtype=np.float64)
        print(f"Haze4K valid scores: {len(scored)}, failed: {len(failed)}")
        print(f"Haze4K FADE min/mean/max: {scores.min():.6f} / {scores.mean():.6f} / {scores.max():.6f}")
        print(f"Haze4K top-{top_k} cutoff: {selected[-1][1]:.6f}")
    return count


def add_ntire_pairs(rows, out_gt: Path, out_hazy: Path, dataset: str, clean_dir: Path, hazy_dir: Path, prefix: str):
    clean_by_key = {norm_ntire_key(p): p for p in list_images(clean_dir)}
    count = 0
    for hazy in tqdm(list_images(hazy_dir), desc=f"Copying {dataset}"):
        key = norm_ntire_key(hazy)
        clean = clean_by_key.get(key)
        if clean is None:
            rows.append([dataset, key, "", str(hazy), "", "", "missing_clean"])
            continue
        clean_target, hazy_target = copy_pair(clean, hazy, prefix, key, out_gt, out_hazy)
        rows.append([dataset, key, str(clean), str(hazy), str(clean_target), str(hazy_target), "ok"])
        count += 1
    return count


def parse_args():
    parser = argparse.ArgumentParser(description="Build Stage1 HazeGen paired train set.")
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    parser.add_argument("--haze4k_top_k", type=int, default=1500)
    parser.add_argument("--max_side", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out_gt = args.out_root / "gt"
    out_hazy = args.out_root / "hazy"
    manifest = args.out_root / "manifest_stage1_train.csv"
    haze4k_fade_csv = args.out_root / "haze4k_fade_scores_top1500.csv"

    if args.overwrite:
        for directory in (out_gt, out_hazy):
            if directory.exists():
                for path in directory.iterdir():
                    if path.is_file():
                        path.unlink()
    out_gt.mkdir(parents=True, exist_ok=True)
    out_hazy.mkdir(parents=True, exist_ok=True)

    rows = []
    counts = {}
    counts["OTS_top800_fade"] = add_ots_pairs(rows, out_gt, out_hazy)
    counts["Haze4K_top_fade"] = add_haze4k_pairs(
        rows, out_gt, out_hazy, args.haze4k_top_k, args.max_side, args.workers, haze4k_fade_csv
    )
    counts["Dense_Haze_NTIRE19"] = add_ntire_pairs(
        rows,
        out_gt,
        out_hazy,
        "Dense_Haze_NTIRE19",
        ROOT / "Dense_Haze_NTIRE19" / "GT",
        ROOT / "Dense_Haze_NTIRE19" / "hazy",
        "dense",
    )
    counts["NH-HAZE"] = add_ntire_pairs(
        rows,
        out_gt,
        out_hazy,
        "NH-HAZE",
        ROOT / "NH-HAZE" / "GT",
        ROOT / "NH-HAZE" / "hazy",
        "nhhaze",
    )
    counts["O-HAZE"] = add_ntire_pairs(
        rows,
        out_gt,
        out_hazy,
        "O-HAZE",
        ROOT / "O-HAZE" / "O-HAZY NTIRE 2018" / "GT",
        ROOT / "O-HAZE" / "O-HAZY NTIRE 2018" / "hazy",
        "ohaze",
    )

    with manifest.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "pair_key", "source_gt", "source_hazy", "target_gt", "target_hazy", "status"])
        writer.writerows(rows)

    print("Counts:")
    for name, count in counts.items():
        print(f"  {name}: {count}")
    print(f"Total copied pairs: {sum(counts.values())}")
    print(f"GT files: {len(list_images(out_gt))}")
    print(f"Hazy files: {len(list_images(out_hazy))}")
    print(f"Output: {args.out_root}")
    print(f"Manifest: {manifest}")
    print(f"Haze4K FADE CSV: {haze4k_fade_csv}")


if __name__ == "__main__":
    main()

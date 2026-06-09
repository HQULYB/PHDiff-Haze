import argparse
import csv
import multiprocessing as mp
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from filter_low_fade import IMAGE_SUFFIXES, compute_fade, init_worker, load_fade, score_worker


def parse_args():
    parser = argparse.ArgumentParser(description="Select images with the highest FADE scores.")
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=Path("fade_scores_top.csv"))
    parser.add_argument("--top_k", type=int, default=800)
    parser.add_argument("--max_side", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--move", action="store_true", help="Move instead of copy selected images.")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def collect_images(image_dir: Path):
    if not image_dir.is_dir():
        raise NotADirectoryError(image_dir)
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def unique_target(output_dir: Path, source: Path):
    target = output_dir / output_name(source)
    if not target.exists():
        return target
    index = 1
    while True:
        candidate = output_dir / f"{source.stem}__dup{index}{source.suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def output_name(source: Path) -> str:
    parts = source.stem.split("_")
    if len(parts) >= 3:
        try:
            float(parts[-1])
            float(parts[-2])
        except ValueError:
            return source.name
        return f"{'_'.join(parts[:-2])}{source.suffix}"
    return source.name


def write_csv(csv_path: Path, scored, failed, selected):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    selected_paths = {path for path, _ in selected}
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "image", "fade", "selected_top_fog", "status", "error"])
        for rank, (image_path, score) in enumerate(scored, start=1):
            writer.writerow(
                [
                    rank,
                    str(image_path),
                    f"{score:.8f}",
                    image_path in selected_paths,
                    "ok",
                    "",
                ]
            )
        for image_path, error in failed:
            writer.writerow(["", str(image_path), "", False, "failed", error])


def main():
    args = parse_args()
    images = collect_images(args.image_dir)
    if not images:
        raise RuntimeError(f"No images found in {args.image_dir}")

    scored = []
    failed = []
    image_strings = [str(path) for path in images]
    if args.workers > 1:
        with mp.Pool(processes=args.workers, initializer=init_worker, initargs=(args.max_side,)) as pool:
            iterator = pool.imap_unordered(score_worker, image_strings)
            for image_path_str, score, error in tqdm(iterator, total=len(image_strings), desc="Computing FADE"):
                image_path = Path(image_path_str)
                if error:
                    failed.append((image_path, error))
                else:
                    scored.append((image_path, score))
    else:
        fade_fn = load_fade()
        for image_path in tqdm(images, desc="Computing FADE"):
            try:
                score = compute_fade(fade_fn, image_path, args.max_side)
            except Exception as exc:
                failed.append((image_path, f"{type(exc).__name__}: {exc}"))
                continue
            scored.append((image_path, score))

    if len(scored) < args.top_k:
        raise RuntimeError(f"Only {len(scored)} valid FADE scores; cannot select top {args.top_k}")

    scored.sort(key=lambda item: item[1], reverse=True)
    selected = scored[: args.top_k]
    write_csv(args.csv, scored, failed, selected)

    scores = np.array([score for _, score in scored], dtype=np.float64)
    cutoff = selected[-1][1]
    print(f"Images: {len(images)}")
    print(f"Valid scores: {len(scored)}")
    print(f"Failed: {len(failed)}")
    print(f"FADE min/mean/max: {scores.min():.6f} / {scores.mean():.6f} / {scores.max():.6f}")
    print(f"Top-{args.top_k} cutoff: FADE >= {cutoff:.6f}")
    print(f"CSV: {args.csv}")

    if args.dry_run:
        print("Dry run enabled; no files were copied or moved.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for image_path, _ in tqdm(selected, desc="Saving selected images"):
        target = unique_target(args.output_dir, image_path)
        if args.move:
            shutil.move(str(image_path), str(target))
        else:
            shutil.copy2(str(image_path), str(target))
    print(f"Saved {len(selected)} images to: {args.output_dir}")


if __name__ == "__main__":
    main()

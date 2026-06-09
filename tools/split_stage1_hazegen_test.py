import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


STAGE1_ROOT = Path("/data/users/gaoyin/datasets/dehaze/Haze_Gen/Stage1")
DEFAULT_COUNTS = {
    "OTS_top800_fade": 80,
    "Haze4K_top_fade": 150,
    "Dense_Haze_NTIRE19": 5,
    "NH-HAZE": 5,
    "O-HAZE": 5,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Split Stage1 HazeGen train pairs into test.")
    parser.add_argument("--stage1_root", type=Path, default=STAGE1_ROOT)
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path):
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if fieldnames is None:
        raise RuntimeError(f"empty manifest: {path}")
    return fieldnames, rows


def write_manifest(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    train_root = args.stage1_root / "Train"
    test_root = args.stage1_root / "Test"
    train_manifest = train_root / "manifest_stage1_train.csv"
    test_manifest = test_root / "manifest_stage1_test.csv"

    fieldnames, rows = read_manifest(train_manifest)
    ok_rows = [row for row in rows if row.get("status", "").startswith("ok")]

    by_dataset = defaultdict(list)
    for row in ok_rows:
        by_dataset[row["dataset"]].append(row)

    rng = random.Random(args.seed)
    selected_ids = set()
    selected_rows = []
    summary = {}
    for dataset, count in DEFAULT_COUNTS.items():
        candidates = by_dataset.get(dataset, [])
        if len(candidates) < count:
            raise RuntimeError(f"{dataset} only has {len(candidates)} rows, cannot split {count}")
        picked = rng.sample(candidates, count)
        summary[dataset] = len(picked)
        for row in picked:
            selected_ids.add((row["dataset"], row["pair_key"], row["target_gt"], row["target_hazy"]))
            selected_rows.append(row)

    test_gt_dir = test_root / "gt"
    test_hazy_dir = test_root / "hazy"
    train_gt_dir = train_root / "gt"
    train_hazy_dir = train_root / "hazy"
    if not args.dry_run:
        test_gt_dir.mkdir(parents=True, exist_ok=True)
        test_hazy_dir.mkdir(parents=True, exist_ok=True)

    updated_test_rows = []
    for row in selected_rows:
        src_gt = Path(row["target_gt"])
        src_hazy = Path(row["target_hazy"])
        if not src_gt.exists() or not src_hazy.exists():
            raise FileNotFoundError(f"missing pair files: {src_gt}, {src_hazy}")

        dst_gt = test_gt_dir / src_gt.name
        dst_hazy = test_hazy_dir / src_hazy.name
        if dst_gt.exists() or dst_hazy.exists():
            raise FileExistsError(f"test target already exists: {dst_gt} or {dst_hazy}")

        test_row = dict(row)
        test_row["target_gt"] = str(dst_gt)
        test_row["target_hazy"] = str(dst_hazy)
        updated_test_rows.append(test_row)

        if not args.dry_run:
            shutil.move(str(src_gt), str(dst_gt))
            shutil.move(str(src_hazy), str(dst_hazy))

    remaining_train_rows = []
    for row in rows:
        row_id = (row["dataset"], row["pair_key"], row["target_gt"], row["target_hazy"])
        if row_id not in selected_ids:
            remaining_train_rows.append(row)

    existing_test_rows = []
    if test_manifest.exists():
        _, existing_test_rows = read_manifest(test_manifest)

    if not args.dry_run:
        write_manifest(train_manifest, fieldnames, remaining_train_rows)
        write_manifest(test_manifest, fieldnames, existing_test_rows + updated_test_rows)

    print("Split summary:")
    for dataset, count in summary.items():
        print(f"  {dataset}: {count}")
    print(f"Total moved pairs: {sum(summary.values())}")
    print(f"Train manifest rows after split: {len(remaining_train_rows)}")
    print(f"Test manifest rows after split: {len(existing_test_rows) + len(updated_test_rows)}")
    print(f"Train GT files: {len(list(train_gt_dir.glob('*')))}")
    print(f"Train hazy files: {len(list(train_hazy_dir.glob('*')))}")
    print(f"Test GT files: {len(list(test_gt_dir.glob('*')))}")
    print(f"Test hazy files: {len(list(test_hazy_dir.glob('*')))}")
    if args.dry_run:
        print("Dry run enabled; no files were moved.")


if __name__ == "__main__":
    main()

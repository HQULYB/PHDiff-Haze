"""Sync existing TensorBoard loss scalars to SwanLab.

Example:
    python tools/sync_tensorboard_losses_to_swanlab.py \
        --logdir experiment \
        --project Learning-Hazing-to-Dehazing \
        --run_name tb-loss-backfill
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader


def scalar_value(summary_value):
    if summary_value.HasField("simple_value"):
        return float(summary_value.simple_value)
    if summary_value.HasField("tensor") and summary_value.tensor.float_val:
        return float(summary_value.tensor.float_val[0])
    return None


def collect_loss_scalars(logdir: Path, tag_prefix: str) -> dict[int, dict[str, float]]:
    by_step: dict[int, dict[str, float]] = defaultdict(dict)
    event_files = sorted(logdir.rglob("events.out.tfevents*"))
    if not event_files:
        raise FileNotFoundError(f"no TensorBoard event files found under: {logdir}")

    for event_file in event_files:
        run_name = event_file.parent.relative_to(logdir).as_posix().replace("/", "_")
        for event in EventFileLoader(str(event_file)).Load():
            for value in event.summary.value:
                val = scalar_value(value)
                if val is None:
                    continue
                if not (value.tag.startswith("loss/") or value.tag.startswith("loss_weighted/")):
                    continue
                tag = value.tag.replace("/", "_")
                by_step[int(event.step)][f"{tag_prefix}/{run_name}/{tag}"] = val
    return by_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=Path, default=Path("experiment"))
    parser.add_argument("--project", type=str, default="Learning-Hazing-to-Dehazing")
    parser.add_argument("--run_name", type=str, default="tensorboard-loss-backfill")
    parser.add_argument("--tag_prefix", type=str, default="tb")
    parser.add_argument("--api_key", type=str, default=os.environ.get("SWANLAB_API_KEY"))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    scalars = collect_loss_scalars(args.logdir, args.tag_prefix)
    total_points = sum(len(values) for values in scalars.values())
    print(f"found {total_points} scalar points across {len(scalars)} steps")

    if args.dry_run:
        for step in sorted(scalars)[:5]:
            print(step, scalars[step])
        return

    import swanlab

    if args.api_key:
        swanlab.login(api_key=args.api_key, save=False)
    swanlab.init(
        project=args.project,
        experiment_name=args.run_name,
        config={
            "source_logdir": str(args.logdir),
            "tag_prefix": args.tag_prefix,
            "total_scalar_points": total_points,
        },
    )
    try:
        for step in sorted(scalars):
            swanlab.log(scalars[step], step=step)
    finally:
        swanlab.finish()


if __name__ == "__main__":
    main()

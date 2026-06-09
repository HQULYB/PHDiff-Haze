#!/usr/bin/env python
"""Analyze Fourier spectra of haze residuals.

This script compares clean images and generated/real hazy images:

    residual = hazy - clean

It saves:
  - average residual FFT heatmap
  - radial power spectrum
  - CSV metrics per image

The goal is to diagnose whether the residual looks like a smooth haze layer or
contains high-frequency noise / local color artifacts.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def build_clean_index(clean_dir: Path) -> dict[str, Path]:
    clean_paths = list_images(clean_dir)
    index: dict[str, Path] = {}
    for path in clean_paths:
        rel_key = str(path.relative_to(clean_dir).with_suffix(""))
        index.setdefault(rel_key, path)
        index.setdefault(path.stem, path)
    return index


def find_pairs(clean_dir: Path, hazy_dir: Path, max_images: int) -> list[tuple[Path, Path]]:
    clean_index = build_clean_index(clean_dir)
    pairs: list[tuple[Path, Path]] = []
    for hazy_path in list_images(hazy_dir):
        rel_key = str(hazy_path.relative_to(hazy_dir).with_suffix("")) if hazy_dir.is_dir() else hazy_path.stem
        clean_path = clean_index.get(rel_key) or clean_index.get(hazy_path.stem)
        if clean_path is not None:
            pairs.append((clean_path, hazy_path))
        if max_images > 0 and len(pairs) >= max_images:
            break
    return pairs


def crop_or_resize_clean(clean: np.ndarray, hazy: np.ndarray) -> np.ndarray:
    if clean.shape[:2] == hazy.shape[:2]:
        return clean
    h, w = hazy.shape[:2]
    return cv2.resize(clean, (w, h), interpolation=cv2.INTER_CUBIC)


def resize_for_analysis(clean: np.ndarray, hazy: np.ndarray, analysis_size: int) -> tuple[np.ndarray, np.ndarray]:
    if analysis_size <= 0:
        return clean, hazy
    h, w = hazy.shape[:2]
    short = min(h, w)
    if short <= analysis_size:
        return clean, hazy
    scale = analysis_size / short
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    clean = cv2.resize(clean, (new_w, new_h), interpolation=cv2.INTER_AREA)
    hazy = cv2.resize(hazy, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return clean, hazy


def fft_power(residual: np.ndarray) -> np.ndarray:
    """Return shifted power spectrum with shape [H, W, 3]."""
    spectrum = np.fft.fftshift(np.fft.fft2(residual, axes=(0, 1)), axes=(0, 1))
    return np.abs(spectrum) ** 2


def normalized_radius(h: int, w: int) -> np.ndarray:
    yy, xx = np.indices((h, w))
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return radius / max(min(h, w) / 2.0, 1.0)


def radial_profile(power_2d: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    r = normalized_radius(*power_2d.shape)
    bin_ids = np.clip((r * bins).astype(np.int32), 0, bins - 1)
    sums = np.bincount(bin_ids.ravel(), weights=power_2d.ravel(), minlength=bins)
    counts = np.bincount(bin_ids.ravel(), minlength=bins).clip(min=1)
    centers = (np.arange(bins) + 0.5) / bins
    return centers, sums / counts


def spectral_metrics(residual: np.ndarray, high_freq_threshold: float) -> dict[str, float]:
    power = fft_power(residual).mean(axis=2)
    r = normalized_radius(*power.shape)
    total = float(power.sum() + 1e-12)
    high = float(power[r >= high_freq_threshold].sum() / total)
    low = float(power[r <= 0.15].sum() / total)

    residual_energy = float(np.mean(residual**2) + 1e-12)
    gray_residual = residual.mean(axis=2, keepdims=True)
    chroma_energy = float(np.mean((residual - gray_residual) ** 2) / residual_energy)
    negative_ratio = float((residual < 0).mean())

    return {
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "low_freq_ratio": low,
        "high_freq_ratio": high,
        "chroma_energy_ratio": chroma_energy,
        "negative_residual_ratio": negative_ratio,
    }


def save_heatmap(power: np.ndarray, output_path: Path, title: str) -> None:
    log_power = np.log1p(power)
    vmax = np.percentile(log_power, 99.5)
    plt.figure(figsize=(6, 5), dpi=160)
    plt.imshow(log_power, cmap="magma", vmin=0.0, vmax=vmax)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_radial_plot(radii: np.ndarray, profiles: list[np.ndarray], output_path: Path) -> None:
    stacked = np.stack(profiles, axis=0)
    mean_profile = stacked.mean(axis=0)
    p25 = np.percentile(stacked, 25, axis=0)
    p75 = np.percentile(stacked, 75, axis=0)

    plt.figure(figsize=(7, 4.5), dpi=160)
    plt.plot(radii, mean_profile, label="mean radial power")
    plt.fill_between(radii, p25, p75, alpha=0.25, label="25-75 percentile")
    plt.yscale("log")
    plt.xlabel("normalized frequency radius")
    plt.ylabel("power")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", type=Path, required=True)
    parser.add_argument("--hazy_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_images", type=int, default=200)
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--spectrum_size", type=int, default=512)
    parser.add_argument(
        "--analysis_size",
        type=int,
        default=512,
        help="Resize image shorter side to this size before FFT. Set <=0 to use original resolution.",
    )
    parser.add_argument("--high_freq_threshold", type=float, default=0.35)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = find_pairs(args.clean_dir, args.hazy_dir, args.max_images)
    if not pairs:
        raise RuntimeError("no clean/hazy image pairs found")

    rows = []
    radial_profiles = []
    avg_power = None
    radii = None

    for clean_path, hazy_path in pairs:
        clean = read_rgb(clean_path)
        hazy = read_rgb(hazy_path)
        clean = crop_or_resize_clean(clean, hazy)
        clean, hazy = resize_for_analysis(clean, hazy, args.analysis_size)
        residual = hazy - clean

        power = fft_power(residual).mean(axis=2)
        power_for_average = cv2.resize(
            power,
            (args.spectrum_size, args.spectrum_size),
            interpolation=cv2.INTER_AREA,
        )
        avg_power = power_for_average if avg_power is None else avg_power + power_for_average
        radii, profile = radial_profile(power, args.bins)
        radial_profiles.append(profile)

        row = {
            "clean": str(clean_path),
            "hazy": str(hazy_path),
            **spectral_metrics(residual, args.high_freq_threshold),
        }
        rows.append(row)

    avg_power = avg_power / len(pairs)
    save_heatmap(avg_power, args.output_dir / "average_residual_fft.png", "Average residual FFT")
    save_radial_plot(radii, radial_profiles, args.output_dir / "radial_power_spectrum.png")

    csv_path = args.output_dir / "residual_spectrum_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0].keys()
        if key not in {"clean", "hazy"}
    }
    summary_path = args.output_dir / "summary.txt"
    with summary_path.open("w") as f:
        f.write(f"pairs: {len(pairs)}\n")
        for key, value in summary.items():
            f.write(f"{key}: {value:.6f}\n")

    print(f"Analyzed {len(pairs)} pairs")
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()

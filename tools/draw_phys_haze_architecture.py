from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "phys_hazegen_architecture_cn_4k.png"
LEGACY_OUT = ROOT / "figures" / "hazegen_architecture_cn_4k.png"

FONT_REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_MED = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


F_TITLE = font(FONT_BOLD, 92)
F_SUB = font(FONT_MED, 38)
F_PANEL = font(FONT_BOLD, 44)
F_NODE = font(FONT_BOLD, 34)
F_TEXT = font(FONT_REG, 26)
F_SMALL = font(FONT_REG, 21)
F_EQ = font(FONT_BOLD, 46)

INK = "#172033"
MUTED = "#5e6b76"
BG = "#f5f7f8"
PAPER = "#ffffff"
LINE = "#2e3c45"
BLUE = "#2f70a0"
BLUE_L = "#edf5fb"
GREEN = "#2d7d62"
GREEN_L = "#eef8f3"
AMBER = "#a87523"
AMBER_L = "#fff6e6"
PURPLE = "#6b5aad"
PURPLE_L = "#f4f1fb"
GRAY = "#d8e0e7"


def rounded(draw: ImageDraw.ImageDraw, xy, fill, outline=None, width=2, r=26):
    draw.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=width)


def shadow_box(img: Image.Image, xy, fill=PAPER, outline=GRAY, r=28, shadow=True, width=2):
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = xy
    if shadow:
        sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(sh)
        sd.rounded_rectangle((x1 + 8, y1 + 12, x2 + 8, y2 + 12), radius=r, fill=(34, 46, 58, 30))
        sh = sh.filter(ImageFilter.GaussianBlur(14))
        img.alpha_composite(sh)
    rounded(draw, xy, fill, outline, width=width, r=r)


def text(draw, xy, s, fnt, fill=INK, anchor="la", align="left", spacing=6):
    draw.multiline_text(xy, s, font=fnt, fill=fill, anchor=anchor, align=align, spacing=spacing)


def center_text(draw, xy, s, fnt, fill=INK, spacing=6):
    x1, y1, x2, y2 = xy
    box = draw.multiline_textbbox((0, 0), s, font=fnt, spacing=spacing, align="center")
    tw, th = box[2] - box[0], box[3] - box[1]
    draw.multiline_text((x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - th) / 2), s, font=fnt, fill=fill, spacing=spacing, align="center")


def arrow(draw: ImageDraw.ImageDraw, p1, p2, color=LINE, w=8, dashed=False):
    if dashed:
        dashed_line(draw, p1, p2, color, w)
    else:
        draw.line((p1, p2), fill=color, width=w)
    x1, y1 = p1
    x2, y2 = p2
    if abs(x2 - x1) >= abs(y2 - y1):
        sign = 1 if x2 >= x1 else -1
        pts = [(x2, y2), (x2 - sign * 34, y2 - 19), (x2 - sign * 34, y2 + 19)]
    else:
        sign = 1 if y2 >= y1 else -1
        pts = [(x2, y2), (x2 - 19, y2 - sign * 34), (x2 + 19, y2 - sign * 34)]
    draw.polygon(pts, fill=color)


def dashed_line(draw: ImageDraw.ImageDraw, p1, p2, color, w=6, dash=30, gap=18):
    x1, y1 = p1
    x2, y2 = p2
    length = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    if length == 0:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    t = 0.0
    while t < length:
        t2 = min(length, t + dash)
        draw.line((x1 + dx * t, y1 + dy * t, x1 + dx * t2, y1 + dy * t2), fill=color, width=w)
        t += dash + gap


def elbow(draw: ImageDraw.ImageDraw, pts, color, w=7, dashed=True):
    for a, b in zip(pts[:-2], pts[1:-1]):
        if dashed:
            dashed_line(draw, a, b, color, w)
        else:
            draw.line((a, b), fill=color, width=w)
    arrow(draw, pts[-2], pts[-1], color=color, w=w, dashed=dashed)


def load_clean() -> Image.Image:
    paths = sorted((ROOT / "inputs").glob("*.png"))
    if paths:
        return Image.open(paths[0]).convert("RGB")
    h, w = 520, 720
    yy, xx = np.mgrid[0:h, 0:w]
    sky = np.dstack([
        0.66 + 0.08 * yy / h,
        0.76 + 0.05 * yy / h,
        0.88 - 0.10 * yy / h,
    ])
    hill = yy > (0.58 * h - 0.18 * h * np.sin(xx / w * np.pi))
    sky[hill] = [0.45, 0.58, 0.38]
    return Image.fromarray((sky.clip(0, 1) * 255).astype(np.uint8))


def depth_img(size=(560, 360)) -> Image.Image:
    w, h = size
    x = np.linspace(0, 1, w)[None, :]
    y = np.linspace(0, 1, h)[:, None]
    d = 0.16 + 0.78 * (0.70 * y + 0.30 * x)
    yy, xx = np.mgrid[0:h, 0:w]
    d -= 0.42 * np.exp(-(((xx - w * 0.32) / (w * 0.19)) ** 2 + ((yy - h * 0.64) / (h * 0.24)) ** 2))
    d = np.clip(d, 0, 1)
    return Image.fromarray((d * 255).astype(np.uint8), "L").convert("RGB")


def density_img(size=(640, 420)) -> Image.Image:
    w, h = size
    x = np.linspace(-1, 1, w)[None, :]
    y = np.linspace(-1, 1, h)[:, None]
    d = 0.16 + 0.60 * ((y + 1) / 2) + np.zeros((h, w))
    d += 0.24 * np.exp(-((x - 0.36) ** 2 / 0.22 + (y + 0.18) ** 2 / 0.20))
    d += 0.15 * np.exp(-((x + 0.55) ** 2 / 0.16 + (y - 0.45) ** 2 / 0.18))
    d = np.clip(d, 0, 1)
    r = 45 + 200 * d
    g = 88 + 118 * d
    b = 170 - 55 * d
    return Image.fromarray(np.dstack([r, g, b]).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))


def hazy_from(clean: Image.Image, dimg: Image.Image, air=(224, 229, 216)) -> Image.Image:
    d = np.asarray(dimg.convert("L").resize(clean.size, Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    d = np.asarray(Image.fromarray((d * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(7)), dtype=np.float32) / 255.0
    c = np.asarray(clean, dtype=np.float32)
    a = np.array(air, dtype=np.float32).reshape(1, 1, 3)
    out = c * (1 - 0.72 * d[..., None]) + a * (0.72 * d[..., None])
    out = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    return ImageEnhance.Contrast(out).enhance(0.88)


def paste_cover(canvas: Image.Image, im: Image.Image, box, r=18):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    im = im.convert("RGB")
    scale = max(bw / im.width, bh / im.height)
    im = im.resize((int(im.width * scale), int(im.height * scale)), Image.Resampling.LANCZOS)
    left = (im.width - bw) // 2
    top = (im.height - bh) // 2
    im = im.crop((left, top, left + bw, top + bh))
    mask = Image.new("L", (bw, bh), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle((0, 0, bw, bh), radius=r, fill=255)
    canvas.paste(im, (x1, y1), mask)


def image_card(canvas: Image.Image, xy, im: Image.Image, label: str):
    draw = ImageDraw.Draw(canvas)
    shadow_box(canvas, xy, fill=PAPER, outline=GRAY, r=24, shadow=True, width=2)
    x1, y1, x2, y2 = xy
    paste_cover(canvas, im, (x1 + 18, y1 + 18, x2 - 18, y2 - 62), r=18)
    center_text(draw, (x1, y2 - 48, x2, y2 - 8), label, F_SMALL, MUTED)


def node(canvas: Image.Image, xy, title, subtitle, color, fill):
    draw = ImageDraw.Draw(canvas)
    shadow_box(canvas, xy, fill=fill, outline=color, r=30, shadow=True, width=4)
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle((x1, y1 + 34, x1 + 22, y2 - 34), radius=9, fill=color)
    is_multiline = "\n" in title
    text(draw, (x1 + 54, y1 + (58 if is_multiline else 66)), title, F_NODE, INK, spacing=4)
    if subtitle:
        text(draw, (x1 + 54, y1 + (150 if is_multiline else 126)), subtitle, F_TEXT, MUTED)


def small_node(canvas: Image.Image, xy, title, subtitle, color):
    draw = ImageDraw.Draw(canvas)
    shadow_box(canvas, xy, fill=PAPER, outline=color, r=22, shadow=False, width=3)
    x1, y1, x2, y2 = xy
    center_text(draw, (x1, y1 + 20, x2, y1 + 70), title, F_TEXT, INK)
    if subtitle:
        center_text(draw, (x1, y1 + 72, x2, y2 - 12), subtitle, F_SMALL, MUTED)


def airlight_node(canvas: Image.Image, xy):
    draw = ImageDraw.Draw(canvas)
    node(canvas, xy, "Atmospheric light A", "head / bank / fixed", AMBER, AMBER_L)
    x1, y1, _, _ = xy
    cols = [(224, 229, 216), (236, 226, 203), (211, 224, 235)]
    for i, col in enumerate(cols):
        draw.rounded_rectangle((x1 + 58 + i * 110, y1 + 170, x1 + 138 + i * 110, y1 + 250), radius=18, fill=col, outline="#9c8d70", width=2)


def main() -> None:
    W, H = 3840, 2160
    canvas = Image.new("RGBA", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    # Paper-like main canvas.
    shadow_box(canvas, (90, 70, 3750, 2080), fill=PAPER, outline="#edf1f4", r=42, shadow=False, width=2)

    text(draw, (180, 150), "PhysHazeDiffusion", F_TITLE, INK)
    text(draw, (184, 250), "physical haze synthesis with density diffusion and low-dimensional airlight", F_SUB, MUTED)
    shadow_box(canvas, (2490, 130, 3590, 250), fill="#fbfdff", outline="#c2cfd9", r=34, shadow=False, width=3)
    center_text(draw, (2490, 130, 3590, 250), "I = J · (1 - d) + A · d", F_EQ, "#17364d")

    clean = load_clean()
    depth = depth_img()
    density = density_img()
    hazy = hazy_from(clean.resize((720, 470), Image.Resampling.LANCZOS), density)

    # Panel A.
    text(draw, (180, 390), "(a) Inference architecture", F_PANEL, INK)
    draw.line((180, 450, 3660, 450), fill="#dce3e9", width=3)

    image_card(canvas, (185, 570, 545, 875), clean, "Clean image J")
    image_card(canvas, (185, 940, 545, 1245), depth, "Depth map D")
    node(canvas, (720, 760, 1160, 1030), "Condition\nEncoder", "VAE(J), D, prompt", BLUE, BLUE_L)
    node(canvas, (1390, 660, 1990, 1125), "Density\nDiffusion", "generate density carrier", BLUE, BLUE_L)
    image_card(canvas, (2210, 650, 2680, 1125), density, "Predicted density d(x)")
    airlight_node(canvas, (2835, 560, 3325, 850))
    node(canvas, (2890, 990, 3450, 1245), "Physical Renderer", "J, d(x), A  →  I", GREEN, GREEN_L)
    image_card(canvas, (2960, 1340, 3450, 1650), hazy, "Hazy image I")

    arrow(draw, (545, 718), (720, 850))
    arrow(draw, (545, 1092), (720, 900))
    arrow(draw, (1160, 895), (1390, 895))
    arrow(draw, (1990, 895), (2210, 895))
    arrow(draw, (2680, 895), (2890, 1085))
    arrow(draw, (3080, 850), (3090, 990), color=AMBER)
    arrow(draw, (3170, 1245), (3180, 1340), color=GREEN)

    # Denoising strip.
    for i, col in enumerate(["#e9f3fc", "#d8e8f7", "#c1d8f2", "#9ebee7"]):
        draw.rounded_rectangle((1480 + i * 115, 875, 1565 + i * 115, 940), radius=18, fill=col, outline="#8fb5d6", width=2)
    center_text(draw, (1475, 952, 1930, 985), "zT  →  z0", F_SMALL, MUTED)

    # Panel B.
    text(draw, (180, 1720), "(b) Training supervision", F_PANEL, INK)
    draw.line((180, 1780, 3660, 1780), fill="#dce3e9", width=3)

    shadow_box(canvas, (210, 1845, 1660, 2035), fill=GREEN_L, outline="#acd2c0", r=28, shadow=False, width=3)
    text(draw, (255, 1883), "Stage 1: paired synthetic supervision", F_TEXT, GREEN)
    image_card(canvas, (270, 1930, 420, 2020), clean, "J")
    image_card(canvas, (460, 1930, 610, 2020), hazy, "I_syn")
    image_card(canvas, (650, 1930, 800, 2020), depth, "D")
    small_node(canvas, (980, 1912, 1350, 2022), "Estimator Ψ", "d*, A*", GREEN)
    arrow(draw, (820, 1975), (980, 1975), color=GREEN, w=6, dashed=True)
    elbow(draw, [(1350, 1975), (1510, 1975), (1510, 1160), (1705, 1160)], GREEN, w=5, dashed=True)

    shadow_box(canvas, (1880, 1845, 3600, 2035), fill=AMBER_L, outline="#d7b26d", r=28, shadow=False, width=3)
    text(draw, (1925, 1883), "Stage 2: real-domain adaptation", F_TEXT, AMBER)
    small_node(canvas, (1950, 1912, 2220, 2022), "Teacher", "pseudo d", AMBER)
    small_node(canvas, (2315, 1912, 2605, 2022), "Real hazy", "A bank", AMBER)
    small_node(canvas, (2700, 1912, 2990, 2022), "CLIP dir", "haze-clear", PURPLE)
    small_node(canvas, (3095, 1912, 3375, 2022), "Student", "adapt", BLUE)
    arrow(draw, (2220, 1975), (2315, 1975), color=AMBER, w=5, dashed=True)
    arrow(draw, (2605, 1975), (2700, 1975), color=AMBER, w=5, dashed=True)
    arrow(draw, (2990, 1975), (3095, 1975), color=PURPLE, w=5, dashed=True)
    elbow(draw, [(2085, 1912), (2085, 1720), (1645, 1720), (1645, 1160), (1725, 1160)], AMBER, w=5, dashed=True)
    elbow(draw, [(2845, 1912), (2845, 1690), (3180, 1690), (3180, 1650)], PURPLE, w=5, dashed=True)

    # Legend.
    draw.line((230, 2110, 430, 2110), fill=LINE, width=7)
    arrow(draw, (430, 2110), (500, 2110), w=7)
    text(draw, (550, 2110), "inference", F_SMALL, MUTED, anchor="lm")
    dashed_line(draw, (810, 2110), (1020, 2110), GREEN, w=6)
    arrow(draw, (1020, 2110), (1090, 2110), color=GREEN, w=6, dashed=True)
    text(draw, (1140, 2110), "training signal", F_SMALL, MUTED, anchor="lm")
    text(draw, (1510, 2110), "Diffusion models density d(x); A and the renderer control color.", F_SMALL, INK, anchor="lm")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rgb = canvas.convert("RGB")
    rgb.save(OUT, quality=96)
    rgb.save(LEGACY_OUT, quality=96)
    print(f"saved {OUT}")
    print(f"saved {LEGACY_OUT}")


if __name__ == "__main__":
    main()

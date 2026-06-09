from __future__ import annotations

from pathlib import Path
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "two_stage_phys_haze_framework.png"

FONT_SANS = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_SANS_MED = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
FONT_SANS_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

W, H = 3200, 1900

BG = "#f7fafc"
INK = "#172331"
MUTED = "#66778a"
LINE = "#c7d3df"
BLUE = "#1f77b4"
BLUE_L = "#e8f3fb"
TEAL = "#139c8f"
TEAL_L = "#e7f7f4"
AMBER = "#d99321"
AMBER_L = "#fff4db"
GREEN = "#3b8f4a"
GREEN_L = "#eaf7ec"
PURPLE = "#7457c8"
PURPLE_L = "#f0ecff"
RED = "#c84d4d"
RED_L = "#fff0f0"
GRAY_L = "#f0f4f8"


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


F_TITLE = font(FONT_SANS_BOLD, 78)
F_SUB = font(FONT_SANS_MED, 34)
F_PANEL = font(FONT_SANS_BOLD, 40)
F_NODE = font(FONT_SANS_BOLD, 31)
F_BODY = font(FONT_SANS_MED, 24)
F_SMALL = font(FONT_SANS, 20)
F_TINY = font(FONT_SANS, 17)
F_EQ = font(FONT_SANS_BOLD, 38)


def bbox(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont):
    return draw.multiline_textbbox((0, 0), text, font=fnt, spacing=6)


def center_text(draw: ImageDraw.ImageDraw, rect, text: str, fnt, fill=INK, spacing=6):
    x1, y1, x2, y2 = rect
    b = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=spacing, align="center")
    tw, th = b[2] - b[0], b[3] - b[1]
    draw.multiline_text(
        (x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - th) / 2),
        text,
        font=fnt,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def text(draw, xy, s, fnt, fill=INK, anchor="la", spacing=5):
    draw.multiline_text(xy, s, font=fnt, fill=fill, anchor=anchor, spacing=spacing)


def rounded(draw, rect, fill, outline=LINE, width=2, radius=22):
    draw.rounded_rectangle(rect, radius=radius, fill=fill, outline=outline, width=width)


def shadowed_box(canvas: Image.Image, rect, fill, outline=LINE, radius=24):
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    x1, y1, x2, y2 = rect
    sd.rounded_rectangle((x1 + 10, y1 + 12, x2 + 10, y2 + 12), radius=radius, fill=(26, 48, 74, 22))
    shadow = shadow.filter(ImageFilter.GaussianBlur(14))
    canvas.alpha_composite(shadow)
    draw = ImageDraw.Draw(canvas)
    rounded(draw, rect, fill, outline=outline, width=2, radius=radius)


def arrow(draw, start, end, color="#40566d", width=4, head=18):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    ang = math.atan2(y2 - y1, x2 - x1)
    pts = [
        (x2, y2),
        (x2 - head * math.cos(ang - math.pi / 6), y2 - head * math.sin(ang - math.pi / 6)),
        (x2 - head * math.cos(ang + math.pi / 6), y2 - head * math.sin(ang + math.pi / 6)),
    ]
    draw.polygon(pts, fill=color)


def node(canvas, rect, title, subtitle="", fill=GRAY_L, edge=LINE, accent=None):
    shadowed_box(canvas, rect, fill, edge, radius=20)
    draw = ImageDraw.Draw(canvas)
    x1, y1, x2, y2 = rect
    if accent:
        draw.rounded_rectangle((x1, y1, x1 + 18, y2), radius=18, fill=accent)
    center_text(draw, (x1 + 30, y1 + 12, x2 - 30, y1 + 70), title, F_NODE, INK)
    if subtitle:
        center_text(draw, (x1 + 28, y1 + 76, x2 - 28, y2 - 12), subtitle, F_SMALL, MUTED)


def label_pill(draw, xy, text_value, fill, edge, fg=INK):
    x, y = xy
    b = draw.textbbox((0, 0), text_value, font=F_SMALL)
    w = b[2] - b[0] + 34
    h = b[3] - b[1] + 18
    draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=fill, outline=edge, width=2)
    draw.text((x + 17, y + 9), text_value, font=F_SMALL, fill=fg)


def clean_thumb(size=(300, 190)):
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        sky = np.array([120 + y * 0.18, 178 + y * 0.10, 220 + y * 0.05])
        ground = np.array([90 + y * 0.35, 116 + y * 0.25, 98 + y * 0.18])
        if y < h * 0.58:
            arr[y, :, :] = np.clip(sky, 0, 255)
        else:
            arr[y, :, :] = np.clip(ground, 0, 255)
    img = Image.fromarray(arr, "RGB")
    d = ImageDraw.Draw(img)
    d.polygon([(80, 108), (135, 45), (205, 108)], fill="#617a8d")
    d.polygon([(170, 112), (235, 55), (306, 112)], fill="#6f8796")
    d.rectangle((38, 105, 82, 170), fill="#445b64")
    d.rectangle((95, 118, 135, 170), fill="#536c70")
    d.polygon([(145, 190), (182, 116), (220, 190)], fill="#596064")
    d.line((182, 116, 182, 190), fill="#f7fafc", width=3)
    return img


def density_thumb(size=(300, 190), seed=7):
    rng = np.random.default_rng(seed)
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    base = 0.15 + 0.65 * (yy / max(1, h - 1)) ** 1.6
    for cx, cy, amp, sx, sy in [
        (rng.uniform(0.15, 0.9) * w, rng.uniform(0.1, 0.8) * h, 0.25, 80, 38),
        (rng.uniform(0.1, 0.8) * w, rng.uniform(0.0, 0.7) * h, 0.18, 60, 50),
        (rng.uniform(0.3, 0.95) * w, rng.uniform(0.3, 0.9) * h, -0.14, 90, 70),
    ]:
        base += amp * np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
    base = np.clip(base, 0, 1)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = (48 + 170 * base).astype(np.uint8)
    rgb[..., 1] = (88 + 135 * base).astype(np.uint8)
    rgb[..., 2] = (128 + 80 * base).astype(np.uint8)
    return Image.fromarray(rgb, "RGB").filter(ImageFilter.GaussianBlur(1.2))


def hazy_thumb(clean: Image.Image, density: Image.Image, air=(224, 230, 229)):
    j = np.asarray(clean).astype(np.float32) / 255.0
    d = np.asarray(density.convert("L")).astype(np.float32)[..., None] / 255.0
    a = np.array(air, dtype=np.float32).reshape(1, 1, 3) / 255.0
    out = j * (1.0 - 0.72 * d) + a * (0.72 * d)
    return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8), "RGB")


def image_card(canvas, xy, img, label, edge=LINE):
    draw = ImageDraw.Draw(canvas)
    x, y = xy
    w, h = img.size
    shadowed_box(canvas, (x, y, x + w + 26, y + h + 62), "#ffffff", edge, radius=18)
    canvas.alpha_composite(img.convert("RGBA"), (x + 13, y + 13))
    center_text(draw, (x + 10, y + h + 22, x + w + 16, y + h + 58), label, F_TINY, MUTED)


def main():
    canvas = Image.new("RGBA", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    # Header
    text(draw, (135, 92), "Two-stage Physical Prior Haze Generation Framework", F_TITLE, INK)
    text(draw, (140, 180), "Density diffusion learns spatial haze distribution; atmospheric light controls color; physical renderer composes realistic haze.", F_SUB, MUTED)
    label_pill(draw, (2530, 102), "I(x) = J(x)(1-d(x)) + A d(x)", "#ffffff", BLUE, BLUE)
    label_pill(draw, (2530, 168), "d(x) = 1 - t(x)", "#ffffff", TEAL, TEAL)

    # Main panels
    stage1 = (120, 285, 1535, 1210)
    stage2 = (1665, 285, 3080, 1210)
    shadowed_box(canvas, stage1, "#ffffff", "#b8d8ef", radius=30)
    shadowed_box(canvas, stage2, "#ffffff", "#d2c6f2", radius=30)
    text(draw, (165, 340), "Stage 1  合成配对物理监督", F_PANEL, BLUE)
    text(draw, (1710, 340), "Stage 2  真实雾域自适应", F_PANEL, PURPLE)

    # Stage 1 thumbnails
    clean = clean_thumb()
    dens = density_thumb()
    hazy = hazy_thumb(clean, dens)
    image_card(canvas, (180, 430), clean, "Clean J")
    image_card(canvas, (180, 700), hazy, "Synthetic Hazy I")

    node(canvas, (600, 445, 990, 610), "Estimate A*", "top bright pixels\nfrom synthetic hazy image", AMBER_L, "#f0cd8b", AMBER)
    node(canvas, (600, 700, 990, 900), "Invert Density d*", "d = (I - J) / (A* - J)\nchannel mean + Gaussian blur", TEAL_L, "#9bd7cf", TEAL)
    node(canvas, (1080, 520, 1455, 720), "Density Carrier", "repeat(d*, 3) * 2 - 1\nVAE encode -> z0", BLUE_L, "#9cc8e6", BLUE)
    node(canvas, (1080, 815, 1455, 1015), "Train Networks", "ControlNet learns density latent\nAirlightHead learns A*", GREEN_L, "#a9d8b1", GREEN)

    arrow(draw, (506, 525), (598, 525), BLUE)
    arrow(draw, (506, 795), (598, 795), TEAL)
    arrow(draw, (990, 800), (1080, 645), TEAL)
    arrow(draw, (1268, 720), (1268, 812), GREEN)
    arrow(draw, (780, 610), (780, 698), AMBER)
    text(draw, (165, 1125), "Goal: obtain physically interpretable pseudo labels d* and A*, then train diffusion on density rather than RGB residual.", F_SMALL, MUTED)

    # Stage 2
    image_card(canvas, (1725, 430), clean_thumb(), "Real Clean J")
    node(canvas, (2140, 415, 2545, 595), "Frozen Teacher", "Stage1 checkpoint\nsample pseudo density latent", BLUE_L, "#9cc8e6", BLUE)
    node(canvas, (2640, 415, 3005, 595), "Student Diffusion", "learn pseudo density\nLpseudo", GREEN_L, "#a9d8b1", GREEN)
    arrow(draw, (2048, 525), (2138, 525), BLUE)
    arrow(draw, (2545, 525), (2638, 525), GREEN)

    node(canvas, (1735, 715, 2115, 925), "Real Hazy Set", "unpaired real haze\nestimate A distribution", AMBER_L, "#f0cd8b", AMBER)
    node(canvas, (2230, 715, 2605, 925), "Airlight Bank", "sample A_real\nmatch AirlightHead", AMBER_L, "#f0cd8b", AMBER)
    node(canvas, (2705, 715, 3025, 925), "CLIP Direction", "CLIP(I_hat)-CLIP(J)\n≈ haze-clear direction", PURPLE_L, "#c8bced", PURPLE)
    arrow(draw, (2115, 820), (2228, 820), AMBER)
    arrow(draw, (2605, 820), (2703, 820), PURPLE)
    arrow(draw, (2820, 715), (2820, 598), PURPLE)

    # A bank swatches
    for i, c in enumerate(["#d8ddd8", "#e8e5d8", "#cfd9dd", "#eee7d7", "#d7dce6"]):
        x = 2310 + i * 48
        draw.rounded_rectangle((x, 865, x + 34, 900), radius=7, fill=c, outline="#aab4bd")

    text(draw, (1710, 1125), "Goal: keep density structure with teacher, adapt haze color/brightness with real A-bank, guide clean-to-haze semantics with CLIP direction.", F_SMALL, MUTED)

    # Bottom inference pipeline
    bottom = (120, 1310, 3080, 1780)
    shadowed_box(canvas, bottom, "#ffffff", "#cfd8e2", radius=30)
    text(draw, (165, 1360), "Inference / Physical Rendering", F_PANEL, GREEN)

    image_card(canvas, (205, 1440), clean_thumb((250, 158)), "Input Clean")
    node(canvas, (580, 1445, 925, 1618), "Condition", "clean latent + depth hint\n+ haze prompt", GRAY_L, "#cfd8e2", MUTED)
    node(canvas, (1045, 1445, 1390, 1618), "Density Diffusion", "sample z -> VAE decode\ncarrier -> d(x)", BLUE_L, "#9cc8e6", BLUE)
    image_card(canvas, (1510, 1435), density_thumb((250, 158), seed=12), "Predicted d(x)")
    node(canvas, (1885, 1445, 2230, 1618), "Airlight A", "head / bank / fixed\nlow-dimensional RGB", AMBER_L, "#f0cd8b", AMBER)
    node(canvas, (2350, 1445, 2685, 1618), "Physical Renderer", "I = J(1-d) + A d", GREEN_L, "#a9d8b1", GREEN)
    image_card(canvas, (2810, 1435), hazy_thumb(clean_thumb((250, 158)), density_thumb((250, 158), seed=12)), "Output Hazy")

    arrow(draw, (482, 1523), (578, 1523), MUTED)
    arrow(draw, (925, 1523), (1043, 1523), BLUE)
    arrow(draw, (1390, 1523), (1508, 1523), BLUE)
    arrow(draw, (1776, 1523), (1883, 1523), AMBER)
    arrow(draw, (2230, 1523), (2348, 1523), GREEN)
    arrow(draw, (2685, 1523), (2808, 1523), GREEN)

    label_pill(draw, (620, 1662), "No free RGB residual", "#fff7e8", AMBER, AMBER)
    label_pill(draw, (1025, 1662), "model p(d | J, D, prompt)", "#eef6ff", BLUE, BLUE)
    label_pill(draw, (1530, 1662), "non-uniform haze field", "#eaf7f4", TEAL, TEAL)
    label_pill(draw, (2020, 1662), "real-domain A distribution", "#fff4db", AMBER, AMBER)
    label_pill(draw, (2485, 1662), "physics-constrained color", "#eaf7ec", GREEN, GREEN)

    # Footer
    text(draw, (135, 1832), "Key idea: diffusion generates haze density, not RGB haze; atmospheric light and the physical renderer provide controllable, interpretable image formation.", F_SMALL, MUTED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(OUT, quality=96)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Banner animado: retrato en puntos que se desintegra en siluetas.

Uso (desde la raíz del repo):
    python scripts/banner.py                      # usa assets/source/photo.jpg
    python scripts/banner.py --photo otra.jpg

Necesita (no se publican; assets/source/ está en .gitignore):
    assets/source/photo.jpg   tu foto (rostro de frente, bien iluminado)
    assets/source/hat.png     foto de un sombrero sobre fondo blanco
Dependencias: pip install pillow numpy scipy

Técnica: tramado Floyd-Steinberg -> nube de puntos; cada silueta se muestrea
con k-means (puntos bien repartidos); el algoritmo húngaro empareja cada
punto con su destino más cercano y SMIL anima el trazado. Sale un SVG por
tema (claro/oscuro) con fondo transparente.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

W, H = 1200, 420
BOX = (40.0, 20.0, 380.0)  # x, y, lado del cuadrado del retrato
N = 1500
SEED = 20260514
STAGGER = 0.07  # s entre grupos de puntos

# Un solo tema: fondo negro y colores de terminal vieja / Norton Commander.
THEMES = {
    "retro": {
        "bg": "#000000",
        "frame": "#FFB000",
        "text": "#FFE55C",
        "muted": "#D98F00",
        "dots": ["#FFB000", "#FF8A1F", "#FFE55C"],
        "r": 1.9,
    },
}

# (etiqueta, forma). La primera etiqueta es la del retrato.
LABELS = [
    ("Analizo la falla. Construyo la solución.", "portrait"),
    ("Blue Team · defender y detectar", "shield"),
    ("Black Hat · pensar como el atacante", "hat"),
    ("Linux · Bash · VPS", "terminal"),
    ("APIs · bots · Python", "code"),
    ("Agentes de IA · skills", "agent"),
]

HOLD_PORTRAIT = 3.6
HOLD_SHAPE = 1.7
MORPH = 1.4


# ---------------------------------------------------------------- retrato
def floyd_steinberg(ink: np.ndarray, gain: float) -> np.ndarray:
    a = np.clip(ink * gain, 0, 1).astype(np.float64)
    h, w = a.shape
    out = np.zeros_like(a, dtype=bool)
    for y in range(h):
        for x in range(w):
            old = a[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = new > 0
            err = old - new
            if x + 1 < w:
                a[y, x + 1] += err * 7 / 16
            if y + 1 < h:
                if x > 0:
                    a[y + 1, x - 1] += err * 3 / 16
                a[y + 1, x] += err * 5 / 16
                if x + 1 < w:
                    a[y + 1, x + 1] += err * 1 / 16
    return out


def ink_map(photo: Path, theme: str, res: int = 190) -> np.ndarray:
    """Mapa de 'tinta' 0..1. Divide por el brillo local (retinex) para resaltar
    rasgos (ojos, cejas, nariz, barba) y aplica una máscara elíptica suave.
    Ajustá la caja de recorte y la elipse si cambiás de foto."""
    img = ImageOps.exif_transpose(Image.open(photo)).convert("L")
    crop = img.crop((170, 70, 1050, 950)).resize((res, res), Image.LANCZOS)
    a = np.asarray(crop, dtype=np.float64) / 255
    blur = np.asarray(crop.filter(ImageFilter.GaussianBlur(res / 9)), dtype=np.float64) / 255
    ink = np.clip(1.05 - 0.95 * a / (blur + 0.06), 0, 1) ** 1.5
    ink = np.clip(ink + 0.35 * (1 - a) ** 2, 0, 1)
    yy, xx = np.mgrid[0:res, 0:res] / (res - 1.0)
    d = np.sqrt(((xx - 0.50) / 0.35) ** 2 + ((yy - 0.50) / 0.49) ** 2)
    return ink * np.clip((1.0 - d) / 0.2, 0, 1)  # misma tinta en ambos temas


def portrait_points(photo: Path, theme: str, rng: np.random.Generator) -> np.ndarray:
    res = 190
    ink = ink_map(photo, theme, res)
    lo, hi = 0.2, 8.0
    for _ in range(14):
        gain = (lo + hi) / 2
        if int(floyd_steinberg(ink, gain).sum()) > N:
            hi = gain
        else:
            lo = gain
    ys, xs = np.nonzero(floyd_steinberg(ink, lo))
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    if len(pts) > N:
        pts = pts[rng.choice(len(pts), N, replace=False)]
    while len(pts) < N:  # relleno improbable
        pts = np.vstack([pts, pts[rng.integers(len(pts))] + rng.normal(0, 0.4, 2)])
    pts += rng.uniform(-0.4, 0.4, pts.shape)
    return pts / res * 400.0


# ---------------------------------------------------------------- siluetas
def shape_mask(name: str) -> Image.Image:
    S = 400
    im = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(im)
    if name == "shield":
        outer = [(200, 30), (330, 75), (330, 200), (300, 280), (200, 372), (100, 280), (70, 200), (70, 75)]
        inner = [(200, 62), (302, 98), (302, 198), (278, 262), (200, 338), (122, 262), (98, 198), (98, 98)]
        d.polygon(outer, fill=255)
        d.polygon(inner, fill=0)
        d.rounded_rectangle((152, 196, 248, 276), 10, fill=255)  # cuerpo del candado
        d.arc((166, 140, 234, 226), 180, 360, fill=255, width=14)  # argolla
        d.ellipse((190, 224, 210, 244), fill=0)
        d.rectangle((196, 238, 204, 260), fill=0)
    elif name == "agent":  # agente con skills: nodo central y herramientas
        cx, cy, R = 200, 200, 135
        sats = [(cx + R * math.cos(math.radians(30 + 60 * k)), cy + R * math.sin(math.radians(30 + 60 * k))) for k in range(6)]
        for i, (x, y) in enumerate(sats):
            d.line((cx, cy, x, y), fill=255, width=9)
            nx, ny = sats[(i + 1) % 6]
            d.line((x, y, nx, ny), fill=255, width=5)
        for x, y in sats:
            d.ellipse((x - 27, y - 27, x + 27, y + 27), fill=255)
            d.ellipse((x - 10, y - 10, x + 10, y + 10), fill=0)
        d.ellipse((cx - 56, cy - 56, cx + 56, cy + 56), fill=255)
        d.ellipse((cx - 22, cy - 22, cx + 22, cy + 22), fill=0)
    elif name == "terminal":
        d.rounded_rectangle((40, 70, 360, 330), 22, fill=255)
        d.rounded_rectangle((56, 108, 344, 314), 12, fill=0)
        d.ellipse((62, 82, 80, 100), fill=0)
        d.ellipse((92, 82, 110, 100), fill=0)
        d.ellipse((122, 82, 140, 100), fill=0)
        d.line((92, 156, 142, 206, 92, 256), fill=255, width=22, joint="curve")
        d.rectangle((166, 252, 252, 270), fill=255)
    elif name == "code":
        d.line((130, 110, 50, 200, 130, 290), fill=255, width=36, joint="curve")
        d.line((270, 110, 350, 200, 270, 290), fill=255, width=36, joint="curve")
        d.line((228, 85, 172, 315), fill=255, width=30)
    else:
        raise ValueError(name)
    return im


HAT_SRC = ASSETS / "source/hat.png"


def hat_ink(res: int = 200) -> np.ndarray:
    """Sombrero a partir de una foto de producto sobre fondo blanco: la máscara
    sale de lo que no es blanco y la densidad de puntos sigue la luz del
    sombrero (cinta, pliegues y reflejos), con un contorno que remarca el ala."""
    from scipy import ndimage as ndi

    img = Image.open(HAT_SRC)
    if img.mode == "RGBA":  # PNG con transparencia: aplanar sobre blanco
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    g = np.asarray(img.convert("L"), dtype=np.float64) / 255

    mask = ndi.binary_fill_holes(ndi.binary_closing(ndi.gaussian_filter(g, 1.0) < 0.93, iterations=3))
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    g, mask = g[y0:y1, x0:x1], mask[y0:y1, x0:x1]

    # ecualización por rango dentro del sombrero: realza cinta, pliegues y reflejos
    vals = ndi.gaussian_filter(g, 0.8)
    order = np.argsort(vals[mask])
    rank = np.empty(order.size)
    rank[order] = np.linspace(0, 1, order.size)
    t = np.zeros_like(g)
    t[mask] = rank
    ink = 0.10 + 0.90 * t ** 1.8
    ring = ndi.binary_dilation(mask, iterations=1) & ~ndi.binary_erosion(mask, iterations=3)
    ink[ring] = np.maximum(ink[ring], 0.75)
    ink = ink * ndi.binary_dilation(mask, iterations=1)

    h, w = ink.shape
    scale = (res * 0.95) / max(h, w)
    small = np.asarray(
        Image.fromarray((ink * 255).astype("uint8")).resize((round(w * scale), round(h * scale)), Image.LANCZOS),
        dtype=np.float64) / 255
    out = np.zeros((res, res))
    oy, ox = (res - small.shape[0]) // 2, (res - small.shape[1]) // 2
    out[oy:oy + small.shape[0], ox:ox + small.shape[1]] = small
    return out


def dither_points(ink: np.ndarray, rng: np.random.Generator, scale: float) -> np.ndarray:
    """N puntos por tramado de `ink` (se ajusta la ganancia hasta llegar a N)."""
    lo, hi = 0.1, 12.0
    for _ in range(14):
        gain = (lo + hi) / 2
        if int(floyd_steinberg(ink, gain).sum()) > N:
            hi = gain
        else:
            lo = gain
    ys, xs = np.nonzero(floyd_steinberg(ink, lo))
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    if len(pts) > N:
        pts = pts[rng.choice(len(pts), N, replace=False)]
    while len(pts) < N:
        pts = np.vstack([pts, pts[rng.integers(len(pts))] + rng.normal(0, 0.4, 2)])
    return (pts + rng.uniform(-0.4, 0.4, pts.shape)) * scale


def shape_points(name: str, rng: np.random.Generator) -> np.ndarray:
    if name == "hat":
        return dither_points(hat_ink(200), rng, 2.0)  # 200 px -> lienzo de 400
    m = np.asarray(shape_mask(name)) > 127
    ys, xs = np.nonzero(m)
    cand = np.stack([xs, ys], axis=1).astype(np.float64)
    cand = cand[rng.choice(len(cand), min(len(cand), 14000), replace=False)]
    centers, _ = kmeans2(cand, N, minit="points", seed=int(rng.integers(1 << 30)), iter=12)
    centers = np.nan_to_num(centers)
    return centers


# ---------------------------------------------------------------- animación
def assign_cyclic(states: list[np.ndarray], sweeps: int = 3) -> list[np.ndarray]:
    """Reordena cada estado para minimizar el recorrido respecto a sus vecinos."""
    n = len(states)
    states = [s.copy() for s in states]
    # orden inicial: cada estado respecto del anterior
    for i in range(1, n):
        r, c = linear_sum_assignment(cdist(states[i - 1], states[i], "sqeuclidean"))
        states[i] = states[i][c]
    for _ in range(sweeps):
        for i in range(1, n):
            prev, nxt = states[i - 1], states[(i + 1) % n]
            cost = cdist(prev, states[i], "sqeuclidean") + cdist(nxt, states[i], "sqeuclidean")
            r, c = linear_sum_assignment(cost)
            states[i] = states[i][c]
    return states


def timeline():
    """Devuelve (dur, [(estado, llega, sale)])."""
    t = 0.0
    segs = []
    for i in range(len(LABELS)):
        arrive = t
        hold = HOLD_PORTRAIT if i == 0 else HOLD_SHAPE
        leave = arrive + hold
        segs.append((i, arrive, leave))
        t = leave + MORPH
    return t, segs  # t = instante en que vuelve al retrato


def to_screen(pts: np.ndarray) -> np.ndarray:
    x0, y0, side = BOX
    return pts / 400.0 * side + np.array([x0, y0])


def path_d(pts: np.ndarray) -> str:
    return "".join(f"M{x:.1f} {y:.1f}h0" for x, y in pts)


def render(theme: str, states: list[np.ndarray]) -> str:
    th = THEMES[theme]
    back, segs = timeline()
    dur = back + 0.5  # margen para el retardo de los grupos
    groups = 6
    parts = []
    for g in range(groups):
        idx = np.arange(g, N, groups)
        delay = g * STAGGER
        values, key_times, splines = [], [], []

        def add(frame_idx: int, t: float, spline: str):
            values.append(path_d(to_screen(states[frame_idx][idx])))
            key_times.append(min(max(t / dur, 0.0), 1.0))
            splines.append(spline)

        for k, (si, arrive, leave) in enumerate(segs):
            a = arrive + (delay if k else 0.0)
            l = leave + delay
            add(si, a, "0.45 0 0.2 1")
            add(si, l, "0.45 0 0.2 1")
        add(0, back + delay, "0.45 0 0.2 1")
        # el último frame debe cerrar el ciclo en t=dur con el retrato
        values.append(values[0]); key_times.append(1.0)
        key_times[0] = 0.0
        # keySplines: uno por intervalo
        sp = []
        for j in range(len(values) - 1):
            sp.append("0.45 0 0.2 1" if j % 2 == 1 else "0 0 1 1")
        # intervalos pares = retención (lineal), impares = morph (suave)
        color = th["dots"][g % 3]
        parts.append(
            f'<path fill="none" stroke="{color}" stroke-width="{th["r"] * 2:.1f}" '
            f'stroke-linecap="round" d="{values[0]}">'
            f'<animate attributeName="d" dur="{dur:.2f}s" repeatCount="indefinite" '
            f'calcMode="spline" keyTimes="{";".join(f"{k:.4f}" for k in key_times)}" '
            f'keySplines="{";".join(sp)}" values="{"|".join(values).replace("|", ";")}"/></path>'
        )

    # etiquetas sincronizadas
    labels = []
    for k, (si, arrive, leave) in enumerate(segs):
        txt = LABELS[si][0]
        fade = 0.35
        kt = [0.0]
        op = [0 if k else 1]
        if k:
            kt += [arrive + 0.35, arrive + 0.35 + fade]
            op += [0, 1]
        kt += [leave + 0.05, leave + 0.05 + fade, dur]
        op += [1, 0, 1 if k == 0 else 0]
        if k == 0:
            kt = [0.0, leave + 0.05, leave + 0.05 + fade, back + 0.35, back + 0.35 + fade, dur]
            op = [1, 1, 0, 0, 1, 1]
        kt = [min(t / dur, 1.0) if i else 0.0 for i, t in enumerate(kt)]
        labels.append(
            f'<text x="480" y="245" font-size="27" fill="{th["text"]}" opacity="{op[0]}">{txt}'
            f'<animate attributeName="opacity" dur="{dur:.2f}s" repeatCount="indefinite" '
            f'keyTimes="{";".join(f"{t:.4f}" for t in kt)}" values="{";".join(str(o) for o in op)}"/></text>'
        )

    font = "ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace"
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">
<title id="t">AlbertiJ</title>
<desc id="d">Retrato en puntos que se desintegra en un escudo, un sombrero black hat, una terminal, código y un agente de IA con sus skills.</desc>
<rect width="{W}" height="{H}" fill="{th["bg"]}"/>
<rect x="8" y="8" width="{W - 16}" height="{H - 16}" fill="none" stroke="{th["frame"]}" stroke-width="2"/>
<rect x="14" y="14" width="{W - 28}" height="{H - 28}" fill="none" stroke="{th["frame"]}" stroke-width="2"/>
<g font-family="{font}">
<text x="478" y="170" font-size="66" font-weight="700" fill="{th["text"]}" letter-spacing="1">AlbertiJ</text>
<text x="480" y="318" font-size="18" fill="{th["muted"]}">Red Team · Blue Team · OSINT · CTF</text>
{"".join(labels)}
</g>
{"".join(parts)}
</svg>
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", default=str(ASSETS / "source/photo.jpg"))
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    shapes = {name: shape_points(name, rng) for _, name in LABELS if name != "portrait"}
    for theme in THEMES:
        order = [portrait_points(Path(args.photo), theme, rng)]
        order += [shapes[name] for _, name in LABELS if name != "portrait"]
        states = assign_cyclic(order)
        out = ASSETS / "banner.svg"
        out.write_text(render(theme, states), encoding="utf-8")
        print(f"{out.relative_to(ROOT)}  {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()

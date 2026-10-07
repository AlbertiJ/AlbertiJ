#!/usr/bin/env python3
"""Genera assets/stats.svg y assets/langs.svg con datos de la API de GitHub.

Solo usa la libreria estandar. Cuenta unicamente repositorios PUBLICOS propios
(sin forks). Pensado para correr en GitHub Actions (ver .github/workflows/stats.yml),
donde GITHUB_TOKEN sube el limite de la API; sin token tambien anda, con menos cupo.

Uso:
    python scripts/stats.py              # consulta la API y escribe los SVG
    python scripts/stats.py --demo       # datos inventados, para probar el diseno
    python scripts/stats.py --placeholder  # tarjetas vacias hasta la primera ejecucion
"""

from __future__ import annotations

import html
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

USER = os.environ.get("GH_USER", "AlbertiJ")
TOKEN = os.environ.get("GITHUB_TOKEN", "")
ASSETS = Path(__file__).resolve().parents[1] / "assets"

# mismo estilo que el banner: fondo negro, marco doble ambar, colores de terminal vieja
BG, FRAME = "#000000", "#FFB000"
TEXT, MUTED, TITLE = "#FFE55C", "#D98F00", "#FFB000"
BARS = ["#FFB000", "#FF8A1F", "#FFE55C", "#E5A5B5", "#E6C7A0", "#C38D9E"]
FONT = "ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace"
W, H = 480, 230


# ---------------------------------------------------------------- datos
def api(path: str, accept: str = "application/vnd.github+json"):
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        headers={"Accept": accept, "User-Agent": f"{USER}-profile-stats", "X-GitHub-Api-Version": "2022-11-28"},
    )
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def collect() -> dict:
    user = api(f"/users/{USER}")
    repos: list[dict] = []
    for page in range(1, 6):
        chunk = api(f"/users/{USER}/repos?per_page=100&type=owner&page={page}")
        repos += chunk
        if len(chunk) < 100:
            break
    propios = [r for r in repos if not r.get("fork")]

    langs: dict[str, int] = {}
    for r in propios:
        try:
            for name, n in api(f"/repos/{USER}/{r['name']}/languages").items():
                langs[name] = langs.get(name, 0) + n
        except urllib.error.URLError:
            continue

    commits = None
    try:
        commits = api(f"/search/commits?q=author:{USER}&per_page=1")["total_count"]
    except Exception:
        pass  # la busqueda tiene su propio limite: si falla, se muestra "-"

    created = datetime.fromisoformat(user["created_at"].replace("Z", "+00:00"))
    anios = (datetime.now(timezone.utc) - created).days / 365.25
    return {
        "repos": user["public_repos"],
        "stars": sum(r["stargazers_count"] for r in propios),
        "forks": sum(r["forks_count"] for r in propios),
        "followers": user["followers"],
        "commits": commits,
        "since": created.year,
        "years": anios,
        "langs": langs,
    }


DEMO = {
    "repos": 12, "stars": 7, "forks": 2, "followers": 5, "commits": 640, "since": 2023, "years": 2.9,
    "langs": {"Python": 520000, "PowerShell": 95000, "HTML": 70000, "JavaScript": 42000, "Shell": 21000, "CSS": 14000},
}
PLACEHOLDER = {"repos": None, "stars": None, "forks": None, "followers": None, "commits": None,
               "since": None, "years": None, "langs": {}}


# ---------------------------------------------------------------- dibujo
def frame(title: str, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">'
        f'<rect width="{W}" height="{H}" fill="{BG}"/>'
        f'<rect x="4" y="4" width="{W - 8}" height="{H - 8}" fill="none" stroke="{FRAME}" stroke-width="2"/>'
        f'<rect x="9" y="9" width="{W - 18}" height="{H - 18}" fill="none" stroke="{FRAME}" stroke-width="2"/>'
        f'<g font-family="{FONT}">'
        f'<text x="26" y="40" font-size="17" font-weight="700" fill="{TITLE}">&gt; {html.escape(title)}</text>'
        f"{body}</g></svg>\n"
    )


def fmt(v) -> str:
    return "-" if v is None else f"{v:,}".replace(",", ".")


def render_stats(d: dict) -> str:
    years = "-" if d["years"] is None else (f"{d['since']}  ({d['years']:.1f} anios)")
    rows = [
        ("Repos publicos", fmt(d["repos"])),
        ("Estrellas recibidas", fmt(d["stars"])),
        ("Forks", fmt(d["forks"])),
        ("Seguidores", fmt(d["followers"])),
        ("Commits publicos", fmt(d["commits"])),
        ("En GitHub desde", years),
    ]
    out = ""
    for i, (k, v) in enumerate(rows):
        y = 78 + i * 25
        out += f'<text x="26" y="{y}" font-size="15" fill="{MUTED}">{html.escape(k)}</text>'
        out += f'<text x="{W - 26}" y="{y}" font-size="15" fill="{TEXT}" text-anchor="end">{html.escape(v)}</text>'
    return frame("estadisticas.sh", out)


def render_langs(d: dict) -> str:
    langs = sorted(d["langs"].items(), key=lambda kv: kv[1], reverse=True)[:6]
    total = sum(d["langs"].values()) or 1
    out = ""
    if not langs:
        out += f'<text x="26" y="86" font-size="14" fill="{MUTED}">Sin datos todavia.</text>'
        out += f'<text x="26" y="{H - 22}" font-size="11" fill="{MUTED}">Se completa con la primera ejecucion del workflow.</text>'
    for i, (name, n) in enumerate(langs):
        y = 74 + i * 25
        pct = n / total * 100
        out += f'<text x="26" y="{y + 11}" font-size="14" fill="{TEXT}">{html.escape(name)}</text>'
        out += f'<rect x="150" y="{y}" width="230" height="12" fill="#1c1408"/>'
        out += f'<rect x="150" y="{y}" width="{max(2, 230 * pct / 100):.1f}" height="12" fill="{BARS[i % len(BARS)]}"/>'
        out += f'<text x="{W - 26}" y="{y + 11}" font-size="14" fill="{MUTED}" text-anchor="end">{pct:.1f}%</text>'
    return frame("lenguajes.sh", out)


def write_if_changed(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        print(f"sin cambios: {path.name}")
        return
    path.write_text(text, encoding="utf-8")
    print(f"escrito: {path.name}")


def main() -> None:
    if "--demo" in sys.argv:
        data = DEMO
    elif "--placeholder" in sys.argv:
        data = PLACEHOLDER
    else:
        data = collect()
    write_if_changed(ASSETS / "stats.svg", render_stats(data))
    write_if_changed(ASSETS / "langs.svg", render_langs(data))


if __name__ == "__main__":
    main()

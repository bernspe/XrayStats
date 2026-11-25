"""Implantat-Detektion (Pedikelschraube, Iliumschraube, Cage) mit dem ONNX-Modell aus ../Xray-Model/train_implants.ipynb.

Vor- und Nachverarbeitung 1:1 wie im Notebook (Zellen 2, 8, 14), ohne Torch:
  Graustufen → Höhe 896 (cache_h, bilinear) → content_box → base_matrix (Letterbox 800, Affinwarp bilinear)
  → /255, 3 Kanäle, ImageNet-Normierung → ONNX → Sigmoid-Heatmaps + Regression → Peaks (Maximum im 5×5-Fenster,
  Score ≥ conf je Klasse) → Enden in Eingangs-px → zurück in Cache-px → Peaks im Letterbox-Rand verwerfen.
Konstanten kommen aus spec in results.json des Laufs. Ausgabe in Stage-Koordinaten (Bildhöhe 600) wie die Nodes.
ends[0] = Start (Schraubenkopf / Einbringseite des Cages), ends[1] = Ende.

    python implant_infer.py --check     # gegen die GT in data/implant_annotations.csv (Bild-URLs aus degen_raw.pkl)
"""
from __future__ import annotations

import io
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from scipy.ndimage import maximum_filter

RUN_DIR = Path(__file__).resolve().parent.parent / 'Xray-Model' / 'runs' / 'implants-merge-20261120'
STAGE_H = 600
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


@lru_cache(maxsize=1)
def _model():
    import onnxruntime as ort
    spec = json.loads((RUN_DIR / 'results.json').read_text())['spec']
    assert spec['decode'] == 'implant-centernet-v2', spec['decode']
    sess = ort.InferenceSession(str(RUN_DIR / spec['onnx']), providers=['CPUExecutionProvider'])
    return sess, spec


def _span(profile, crop):
    n, idx = len(profile), np.flatnonzero(profile > crop['fill'])
    runs = [r for r in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(r) >= crop['run'] * n] if idx.size else []
    if not runs:
        return 0, n
    m = round(crop['margin'] * n)
    return max(0, int(runs[0][0]) - m), min(n, int(runs[-1][-1]) + 1 + m)


def _content_box(img, crop):
    a = np.asarray(img) > crop['dark']
    (x0, x1), (y0, y1) = _span(a.mean(0), crop), _span(a.mean(1), crop)
    return x0, y0, x1, y1


def _base_matrix(box, inp):
    w, h = box[2] - box[0], box[3] - box[1]
    s = min(inp / w, inp / h)
    pl, pt = (inp - round(w * s)) // 2, (inp - round(h * s)) // 2
    return np.array([[s, 0, pl - box[0] * s], [0, s, pt - box[1] * s], [0, 0, 1]])


def _warp(m, pts):
    return np.asarray(pts, float) @ m[:2, :2].T + m[:2, 2]


def detect(img: Image.Image) -> list[dict]:
    """Bild (beliebige Größe) → [{'cls', 'score', 'ends': [[x, y], [x, y]]}] in Stage-Koordinaten (Höhe 600)."""
    sess, spec = _model()
    inp, stride, classes = spec['input'], spec['stride'], spec['classes']
    g = img.convert('L')
    g = g.resize((max(1, round(g.width * spec['cache_h'] / g.height)), spec['cache_h']), Image.BILINEAR)
    m = _base_matrix(_content_box(g, spec['crop']), inp)
    inv = np.linalg.inv(m)
    x = np.asarray(g.transform((inp, inp), Image.AFFINE, data=tuple(inv[:2].ravel()), resample=Image.BILINEAR),
                   np.float32) / 255
    x = ((np.repeat(x[None], 3, 0) - MEAN) / STD)[None].astype(np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    nc = len(classes)
    prob, reg = 1 / (1 + np.exp(-out[:nc])), out[nc:]
    conf = np.array([spec['conf'][c] for c in classes])[:, None, None]
    k = int(spec['nms'].split('x')[-1]) if 'x' in spec['nms'] else 5
    peak = (prob == maximum_filter(prob, size=(1, k, k), mode='constant', cval=-np.inf)) & (prob > conf)
    dets, f = [], STAGE_H / spec['cache_h']
    for c, iy, ix in zip(*np.nonzero(peak)):
        ox, oy, hl, st, ct = (float(v) for v in reg[:, iy, ix])
        mid = np.array([ix + ox, iy + oy]) * stride
        th = math.atan2(st, ct)
        d = hl * spec['len_scale'] * np.array([math.cos(th), math.sin(th)])
        e = _warp(inv, np.stack([mid - d, mid + d]))                         # Eingangs-px → Cache-px
        if 0 <= e.mean(0)[0] < g.width and 0 <= e.mean(0)[1] < g.height:   # Peaks im Letterbox-Rand verwerfen
            dets.append({'cls': classes[c], 'score': float(prob[c, iy, ix]), 'ends': (e * f).round(2).tolist()})
    return dets


def detect_url(url: str) -> list[dict]:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return detect(Image.open(io.BytesIO(r.content)))


def check(n: int = 30) -> None:
    """Abnahme: Detektionen gegen die GT (implant_annotations.csv, Stage-Koordinaten). Seitlich überlagerte
    Schraubenpaare fasst das Modell zu einer Detektion zusammen (spec['merge']) → GT dort vorher ebenso mergen."""
    import pandas as pd
    _, spec = _model()
    ann = pd.read_csv('data/implant_annotations.csv')
    ann = ann[ann['cls'].isin(spec['classes'])]
    urls = pd.read_pickle('data/degen_raw.pkl').drop_duplicates('id').set_index('id')['img']
    ids = [i for i in ann['image_id'].unique() if i in urls.index][:n]
    rows = []
    for i in ids:
        g = ann[ann['image_id'] == i]
        gt = [(r.cls, np.array([[r.x1, r.y1], [r.x2, r.y2]])) for r in g.itertuples()]
        if g['img_type'].iloc[0] in spec['merge']['img_types']:
            gt = _merge(gt, spec['merge']['frac'])
        try:
            pr = detect_url(urls[i])
        except requests.RequestException as e:
            print(f'{i}: Bild nicht ladbar ({e})')
            continue
        free = [True] * len(pr)
        for c, e in gt:
            tol = max(0.5 * np.linalg.norm(e[1] - e[0]), 8 * STAGE_H / spec['cache_h'])
            cand = [(np.linalg.norm(np.mean(p['ends'], 0) - e.mean(0)), j) for j, p in enumerate(pr)
                    if free[j] and p['cls'] == c]
            d, j = min(cand) if cand else (np.inf, -1)
            hit = d <= tol
            if hit:
                free[j] = False
                same_dir = float(np.dot(np.subtract(*pr[j]['ends'][::-1]), e[1] - e[0]) > 0)
            rows.append((i, c, hit, d if hit else np.nan, same_dir if hit else np.nan))
        rows += [(i, p['cls'], None, np.nan, np.nan) for j, p in enumerate(pr) if free[j]]     # FP
    d = pd.DataFrame(rows, columns=['image_id', 'cls', 'hit', 'dist', 'dir'])
    for c, h in d.groupby('cls'):
        tp, fn, fp = (h['hit'] == True).sum(), (h['hit'] == False).sum(), h['hit'].isna().sum()   # noqa: E712
        print(f'{c:8s} GT {tp + fn:3d}  Recall {tp / max(tp + fn, 1):.0%}  Präzision {tp / max(tp + fp, 1):.0%}  '
              f'Mitte {h["dist"].median():.1f} Stage-px  Richtung {h["dir"].mean():.0%}')


def _merge(gt, frac):
    """merge_overlap aus dem Notebook (gierig, engstes Paar gleicher Klasse zuerst, Enden gemittelt)."""
    out, used = [], set()
    mid = [e.mean(0) for _, e in gt]
    ln = [np.linalg.norm(e[1] - e[0]) for _, e in gt]
    pairs = sorted((np.linalg.norm(mid[i] - mid[j]), i, j) for i in range(len(gt)) for j in range(i + 1, len(gt))
                   if gt[i][0] == gt[j][0])
    for d, i, j in pairs:
        if i in used or j in used or d >= frac * (ln[i] + ln[j]) / 2:
            continue
        ei, ej = gt[i][1], gt[j][1]
        b = ej if np.abs(ei - ej).sum() <= np.abs(ei - ej[::-1]).sum() else ej[::-1]
        used |= {i, j}
        out.append((gt[i][0], (ei + b) / 2))
    return out + [gt[k] for k in range(len(gt)) if k not in used]


if __name__ == '__main__':
    import sys
    if '--check' in sys.argv:
        check()

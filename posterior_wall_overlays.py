"""Overlay-Check für spine_geom.posterior_wall an Xray-LWS-lat-Bildern.

    XRAY_TOKEN=<token> python posterior_wall_overlays.py [n]

Ohne XRAY_TOKEN werden die gecachten (24 h gültigen) URLs aus data/img_urls.pkl benutzt.
PNGs landen in output/posterior_wall/.
"""
import io
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image

from spine_geom import SPINE_LABELS
from spine_geom.posterior_wall import posterior_walls, segment_slips

KONVA_STAGE_HEIGHT = 600
OUT = 'output/posterior_wall'


def main(n=20):
    if os.environ.get('XRAY_TOKEN'):
        from prepare_outlier_data import fetch_img_urls
        img_urls = fetch_img_urls(os.environ['XRAY_TOKEN'])
        with open('data/img_urls.pkl', 'wb') as fh:
            pickle.dump(img_urls, fh)
    else:
        img_urls = pd.read_pickle('data/img_urls.pkl')

    df = pd.read_pickle('data/df4d_corrected_spine_measurements.pkl')
    df = df[df.img_type == 'Xray-LWS-lat'].drop_duplicates('id')
    os.makedirs(OUT, exist_ok=True)
    done = 0
    for _, row in df.sample(frac=1, random_state=0).iterrows():
        so = row.xray_measurement_data.get('stateObject', {})
        nodes, vr = so.get('nodes', []), so.get('vertebra_range', [0, 0])
        labels = SPINE_LABELS[vr[0]:vr[1] + 1]
        if len(nodes) < 3 or len(nodes) != len(labels) or row.id not in img_urls:
            continue
        r = requests.get(img_urls[row.id], timeout=20)
        if not r.ok:
            sys.exit(f'Bild {row.id}: HTTP {r.status_code} – URLs abgelaufen? XRAY_TOKEN setzen.')
        im = Image.open(io.BytesIO(r.content)).convert('L')
        img = np.asarray(im, float) / 255
        cs = np.array([(p['x'], p['y']) for p in nodes], float) * im.height / KONVA_STAGE_HEIGHT
        walls = posterior_walls(img, cs, labels)
        slips = segment_slips(walls, cs, labels)

        fig, ax = plt.subplots(figsize=(8, 10))
        ax.imshow(img, cmap='gray')
        ax.plot(*cs.T, 'o', color='deepskyblue', ms=4)
        for i, (lab, w) in enumerate(zip(labels, walls)):
            if w is None:
                ax.text(*cs[i], f' {lab}: –', color='red', fontsize=8)
                continue
            ax.plot(*np.array(w).T, '-', color='orange' if lab == 'S1' else 'lime', lw=1.5)
            ax.plot(*w[0], 'o', color='yellow', ms=4)          # PS (bei S1: hinteres Deckplattenende)
            txt = lab
            if i < len(slips) and slips[i] is not None:
                txt += f'  slip {slips[i]:+.0f}px'
            ax.text(*cs[i], ' ' + txt, color='yellow', fontsize=8)
        s = np.median(np.linalg.norm(np.diff(cs, axis=0), axis=1))
        ax.set_xlim(cs[:, 0].min() - 1.5 * s, cs[:, 0].max() + 1.5 * s)
        ax.set_ylim(cs[:, 1].max() + s, cs[:, 1].min() - s)
        ax.set_title(f'id {row.id} – {row.patient_id} – {sum(w is None for w in walls)} ohne Wand')
        ax.axis('off')
        fig.savefig(f'{OUT}/{row.id}.png', dpi=110, bbox_inches='tight')
        plt.close(fig)
        done += 1
        print(f'{done:2d}/{n} id {row.id}')
        if done >= n:
            break


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)

"""Hinterkante der Wirbelkörper im seitlichen LWS-Röntgen (anterior links, posterior rechts).

Vom Centroid aus: Bildausschnitt entlang der lokalen Wirbelsäulenachse ausrichten, Vorderkante finden
und am Centroid spiegeln, dort die Hinterkante als hell→dunkel-Übergang nach posterior suchen,
Gerade robust fitten, Deck-/Grundplatte als stärksten kranio-kaudalen Gradienten knapp ventral der
Wand → PS- und PI-Ecke. S1 gesondert: Sakrum-Deckplatte als Linie, deren posteriores Ende = PS.
Alle Koordinaten (x, y) in Bildpixeln, y nach unten; Längen-Schwellen relativ zum Wirbelabstand s.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates

# Plausibilität (Kalibrier-Knöpfe, × s bzw. Grad)
WALL_LEN = (0.5, 1.0)        # Länge PS–PI; Wirbelkörperhöhe ≈ 0,75 s
MAX_WALL_ANGLE = 20.0        # Hinterkante gegen lokale Achse
MIN_EDGE_SNR = 1.0           # zeilengemittelte Wandkante / Pixel-Gradientenstreuung (MAD); Rauschen < 0,95
S1_LEN = (0.7, 2.0)          # Länge der Sakrum-Deckplatte
MAX_S1_ANGLE = 35.0          # Sakrum-Deckplatte gegen Normale der Achse L5→S1
MAX_SLIP = 0.5               # |Slip|; ≈ Meyerding II bei Wirbeltiefe ≈ 1,4 s


def _frame(c_prev, c, c_next):
    """Lokales Achsensystem: t kaudal, n posterior (+x), s = Wirbelabstand px."""
    c_prev, c, c_next = (np.asarray(p, float) for p in (c_prev, c, c_next))
    t = c_next - c_prev; s = np.linalg.norm(t) / 2; t /= 2 * s
    n = np.array([-t[1], t[0]]); n = n if n[0] > 0 else -n
    return c, t, n, s


def _roi(img, c, t, n, s, u, v):
    V, U = np.meshgrid(v, u, indexing='ij')
    roi = map_coordinates(img, [c[1] + U*n[1] + V*t[1], c[0] + U*n[0] + V*t[0]], order=1)
    gv, gu = np.gradient(gaussian_filter(roi, max(1.0, 0.03 * s)))
    return U, gv, gu


def _robust_line(x, y, tol, init_tol):
    """y = a·x + b mit Ausreißer-Abwurf. ponytail: Mini-RANSAC, skimage.ransac falls nötig."""
    inl = np.abs(y - np.median(y)) < init_tol
    for _ in range(3):
        if inl.sum() < 3:
            return None, inl
        ab = np.polyfit(x[inl], y[inl], 1)
        inl = np.abs(np.polyval(ab, x) - y) < tol
    return ab, inl


def posterior_wall(img, c_prev, c, c_next):
    """img: 2D-Graustufen (Knochen hell); c_*: Centroide (x, y) von kranialem Nachbar, Wirbel, kaudalem Nachbar.
    Gibt (PS, PI) als (x, y) zurück oder None, wenn keine plausible Wand gefunden wird."""
    c, t, n, s = _frame(c_prev, c, c_next)
    u, v = np.arange(-1.1 * s, 1.1 * s), np.arange(-0.6 * s, 0.6 * s)
    U, gv, gu = _roi(img, c, t, n, s, u, v)

    # 1) Vorderkante (kontrastreich) spiegeln → Erwartung für die Hinterkante (Centroid ≈ Körpermitte)
    rows = np.abs(v) < 0.35 * s
    gm = gu[rows].mean(axis=0)
    ant = (u > -1.0 * s) & (u < -0.3 * s)
    u_exp = -u[ant][np.argmax(gm[ant])]
    post = (u > 0.3 * s) & (u < 1.05 * s)
    i0 = np.argmax(-gm[post] * np.exp(-((u[post] - u_exp) / (0.2 * s)) ** 2))
    u0 = u[post][i0]
    if -gm[post][i0] < MIN_EDGE_SNR * 1.4826 * np.median(np.abs(gu - np.median(gu))):
        return None                                                     # Kante nicht stärker als Textur/Rauschen

    # 2) Hinterkante je Zeile nahe u0 verfeinern (hell→dunkel nach posterior), Gerade robust gefittet
    win = np.abs(u - u0) < 0.1 * s
    vr, ur = v[rows], u[win][np.argmin(gu[rows][:, win], axis=1)]
    ab, inl = _robust_line(vr, ur, max(2.0, 0.03 * s), np.inf)
    if ab is None or inl.mean() < 0.5:
        return None

    # 3) Ecken: Deck-/Grundplatte = stärkster kranio-kaudaler Gradient knapp ventral der Wand
    uw = np.polyval(ab, v)[:, None]
    prof = np.nanmean(np.where((U > uw - 0.4*s) & (U < uw - 0.05*s), gv, np.nan), axis=1)
    top, bot = (v > -0.55 * s) & (v < -0.2 * s), (v > 0.2 * s) & (v < 0.55 * s)
    v_ps, v_pi = v[top][np.argmax(prof[top])], v[bot][np.argmin(prof[bot])]

    # 4) Plausibilität: Wandlänge und Winkel zur Achse
    length = (v_pi - v_ps) * np.hypot(1, ab[0]) / s
    if not WALL_LEN[0] <= length <= WALL_LEN[1] or np.degrees(np.arctan(abs(ab[0]))) > MAX_WALL_ANGLE:
        return None
    to_img = lambda vv: c + np.polyval(ab, vv) * n + vv * t
    return to_img(v_ps), to_img(v_pi)


def sacral_endplate(img, c_l5, c_s1, c_next):
    """Sakrum-Deckplatte als Linie; gibt (PS, AS) = posteriores/anteriores Ende oder None."""
    c, t, n, s = _frame(c_l5, c_s1, c_next)
    u, v = np.arange(-0.9 * s, 0.9 * s), np.arange(-0.8 * s, 0.1 * s)
    _, gv, _ = _roi(img, c, t, n, s, u, v)

    # Disk (dunkel) → Sakrum (hell) nach kaudal: je Spalte stärkster positiver Gradient, Linie v(u)
    vp, gmax = v[np.argmax(gv, axis=0)], gv.max(axis=0)
    keep = gmax > 0.3 * gmax.max()                                      # nur Spalten mit deutlicher Kante
    ab, inl_k = _robust_line(u[keep], vp[keep], max(2.0, 0.04 * s), np.inf)
    if ab is None or inl_k.mean() < 0.5 or np.degrees(np.arctan(abs(ab[0]))) > MAX_S1_ANGLE:
        return None
    inl = np.zeros(len(u), bool); inl[np.flatnonzero(keep)[inl_k]] = True

    # Enden: entlang der Linie so weit, wie die Kante stark bleibt (zusammenhängend ab Inlier-Mitte)
    vi = np.clip(np.searchsorted(v, np.polyval(ab, u)), 0, len(v) - 1)
    g = gaussian_filter1d(gv[vi, np.arange(len(u))], max(1.0, 0.03 * s))
    strong = g > 0.4 * np.median(g[inl])
    i0 = int(np.median(np.flatnonzero(inl)))
    if not strong[i0]:
        return None
    i_post = i0 + (np.argmin(strong[i0:]) if not strong[i0:].all() else len(u) - i0) - 1
    i_ant = i0 - (np.argmin(strong[i0::-1]) if not strong[i0::-1].all() else i0 + 1) + 1
    if not S1_LEN[0] <= (u[i_post] - u[i_ant]) * np.hypot(1, ab[0]) / s <= S1_LEN[1]:
        return None
    to_img = lambda uu: c + uu * n + np.polyval(ab, uu) * t
    return to_img(u[i_post]), to_img(u[i_ant])


def posterior_walls(img, centroids, labels=None):
    """Für alle Centroide (kranial→kaudal); Endwirbel mit gespiegeltem Nachbarn.
    Liste von (PS, PI) oder None; ist das letzte Label 'S1', dort (PS, AS) der Sakrum-Deckplatte."""
    c = np.asarray(centroids, float)
    ext = np.vstack([2*c[0] - c[1], c, 2*c[-1] - c[-2]])
    walls = [posterior_wall(img, ext[i], ext[i+1], ext[i+2]) for i in range(len(c))]
    if labels is not None and labels[-1] == 'S1':
        walls[-1] = sacral_endplate(img, ext[-3], ext[-2], ext[-1])
    return walls


def slip(ps_lo, pi_lo, pi_up):
    """Abstand (px) der PI-Ecke des oberen Wirbels zur Hinterkantenlinie des unteren; >0 = Anterolisthese."""
    d = (pi_lo - ps_lo) / np.linalg.norm(pi_lo - ps_lo)
    w = pi_up - ps_lo
    return d[0]*w[1] - d[1]*w[0]


def sacral_slip(ps_s1, as_s1, pi_l5):
    """L5/S1 nach Meyerding: Versatz (px) der L5-PI-Ecke gegen S1-PS entlang der Deckplatte; >0 = anterior."""
    e = (ps_s1 - as_s1) / np.linalg.norm(ps_s1 - as_s1)
    return (ps_s1 - pi_l5) @ e


def segment_slips(walls, centroids, labels=None):
    """Slip (px) je Segment i/i+1; None bei fehlender Kante oder |Slip| > MAX_SLIP·s."""
    c = np.asarray(centroids, float)
    out = []
    for i in range(len(walls) - 1):
        up, lo = walls[i], walls[i + 1]
        if up is None or lo is None:
            out.append(None)
            continue
        is_s1 = labels is not None and labels[i + 1] == 'S1'
        d = sacral_slip(*lo, up[1]) if is_s1 else slip(*lo, up[1])
        out.append(d if abs(d) <= MAX_SLIP * np.linalg.norm(c[i + 1] - c[i]) else None)
    return out

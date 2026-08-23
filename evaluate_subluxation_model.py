"""
Subluxation Model v2 — Diskriminationsanalyse

Usage:
    /Users/peterbernstein/Python/ICFXStats/venv/bin/python evaluate_subluxation_model.py
    /Users/peterbernstein/Python/ICFXStats/venv/bin/python evaluate_subluxation_model.py --sigma 3.0 --folds 10 --plots

Output:
    Textbericht auf stdout
    --plots: Kurven nach output/subluxation_eval_*.png
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                              roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from annotation_analysis import V2_FEATURES, _V2_IMPUTE_ZERO, build_v2_training_table

DATA_DIR = "data"
DEFAULT_FEEDBACK = f"{DATA_DIR}/subluxation_v2_feedback.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def youden_threshold(fpr, tpr, thresholds):
    """Threshold that maximises Youden index (sensitivity + specificity − 1)."""
    idx = np.argmax(tpr - fpr)
    return thresholds[idx], tpr[idx], 1.0 - fpr[idx]


def f1_at(y_true, y_score, thr):
    pred = (y_score >= thr).astype(int)
    tp = ((pred == 1) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def bootstrap_group_auc(y_true, y_score, groups, n_boot=1000, seed=42):
    """95 % CI for ROC-AUC via group-level bootstrap (resample entire images)."""
    rng = np.random.RandomState(seed)
    unique_groups = np.unique(groups)
    group_map = {g: np.where(groups == g)[0] for g in unique_groups}
    aucs = []
    for _ in range(n_boot):
        chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_map[g] for g in chosen])
        yb, sb = y_true[idx], y_score[idx]
        if yb.sum() == 0 or (1 - yb).sum() == 0:
            continue
        aucs.append(roc_auc_score(yb, sb))
    aucs = np.array(aucs)
    return aucs.mean(), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cv(orig_df, full_df, n_folds):
    """
    GroupKFold CV:
      - groups by image_id (no image in both train and test)
      - train on augmented samples (full_df) belonging to train images
      - evaluate on original samples (orig_df) belonging to test images
    """
    gkf = GroupKFold(n_splits=n_folds)
    groups = orig_df["image_id"].values
    results = []
    oof_scores = np.full(len(orig_df), np.nan)
    oof_labels = orig_df["label"].values

    for fold, (train_idx, test_idx) in enumerate(gkf.split(orig_df, orig_df["label"], groups)):
        train_ids = set(orig_df.iloc[train_idx]["image_id"])
        test_ids  = set(orig_df.iloc[test_idx]["image_id"])

        # Train: augmented rows from train images
        train_df = full_df[full_df["image_id"].isin(train_ids)]
        Xtr = train_df[V2_FEATURES].copy()
        Xtr[_V2_IMPUTE_ZERO] = Xtr[_V2_IMPUTE_ZERO].fillna(0.0)
        mask_tr = Xtr.notna().all(axis=1)
        Xtr, ytr = Xtr[mask_tr].values, train_df.loc[mask_tr, "label"].values

        # Test: original rows from test images
        test_df = orig_df[orig_df["image_id"].isin(test_ids)]
        Xte = test_df[V2_FEATURES].copy()
        Xte[_V2_IMPUTE_ZERO] = Xte[_V2_IMPUTE_ZERO].fillna(0.0)
        mask_te = Xte.notna().all(axis=1)
        Xte_clean = Xte[mask_te].values
        yte = test_df.loc[mask_te, "label"].values
        te_positions = test_df.index[mask_te]

        scaler = StandardScaler().fit(Xtr)
        model = LogisticRegression(class_weight="balanced", max_iter=1000,
                                   random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(scaler.transform(Xtr), ytr)

        scores = model.predict_proba(scaler.transform(Xte_clean))[:, 1]
        # Store for OOF bootstrap
        oof_scores[te_positions] = scores

        if yte.sum() == 0 or (1 - yte).sum() == 0:
            continue  # degenerate fold

        auc = roc_auc_score(yte, scores)
        ap  = average_precision_score(yte, scores)
        fpr, tpr, thrs = roc_curve(yte, scores)
        thr_y, sens_y, spec_y = youden_threshold(fpr, tpr, thrs)
        f1_y = f1_at(yte, scores, thr_y)

        results.append({
            "fold": fold + 1,
            "n_train": int(mask_tr.sum()),
            "n_test":  int(mask_te.sum()),
            "n_pos_test": int(yte.sum()),
            "AUC":  auc,
            "AP":   ap,
            "Sens@Youden": sens_y,
            "Spec@Youden": spec_y,
            "F1@Youden":   f1_y,
            "fpr": fpr, "tpr": tpr,   # for plotting
            "scores": scores, "labels": yte,
        })

    return results, oof_scores, oof_labels, groups


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_header(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def print_cv_table(results):
    cols = ["AUC", "AP", "Sens@Youden", "Spec@Youden", "F1@Youden"]
    rows = [[r["fold"],
             r["n_train"], r["n_test"], r["n_pos_test"],
             *[f"{r[c]:.3f}" for c in cols]]
            for r in results]
    header = f"{'Fold':>4}  {'Tr':>5}  {'Te':>5}  {'Pos':>4}  " + \
             "  ".join(f"{c:>11}" for c in cols)
    print(header)
    print("─" * len(header))
    for row in rows:
        print(f"{row[0]:>4}  {row[1]:>5}  {row[2]:>5}  {row[3]:>4}  " +
              "  ".join(f"{v:>11}" for v in row[4:]))
    print("─" * len(header))
    vals = {c: np.array([r[c] for r in results]) for c in cols}
    print("Mean  " + " " * 17 +
          "  ".join(f"{vals[c].mean():>11.3f}" for c in cols))
    print(" ±SD  " + " " * 17 +
          "  ".join(f"{vals[c].std():>11.3f}" for c in cols))


def print_feature_importance(full_df):
    """Full-model fit on all data; standardised coefficients."""
    X = full_df[V2_FEATURES].copy()
    X[_V2_IMPUTE_ZERO] = X[_V2_IMPUTE_ZERO].fillna(0.0)
    mask = X.notna().all(axis=1)
    X_clean, y = X[mask].values, full_df.loc[mask, "label"].values
    scaler = StandardScaler().fit(X_clean)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0)
        m.fit(scaler.transform(X_clean), y)
    coefs = pd.Series(m.coef_[0], index=V2_FEATURES).sort_values(key=abs, ascending=False)
    for feat, coef in coefs.items():
        bar = "█" * int(abs(coef) / coefs.abs().max() * 20)
        print(f"  {feat:<22} {coef:+.3f}  {bar}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_plots(results, oof_scores, oof_labels, groups, output_dir="output"):
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update({"figure.dpi": 130, "font.size": 10})
    os.makedirs(output_dir, exist_ok=True)

    valid = ~np.isnan(oof_scores)

    # --- ROC ---
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in results:
        ax.plot(r["fpr"], r["tpr"], lw=1, alpha=0.4,
                label=f"Fold {r['fold']} (AUC={r['AUC']:.3f})")
    if valid.sum() > 0:
        fpr_oof, tpr_oof, _ = roc_curve(oof_labels[valid], oof_scores[valid])
        auc_oof = roc_auc_score(oof_labels[valid], oof_scores[valid])
        ax.plot(fpr_oof, tpr_oof, "k-", lw=2.5,
                label=f"OOF mean (AUC={auc_oof:.3f})")
    ax.plot([0, 1], [0, 1], "r--", lw=1)
    ax.set_xlabel("1 − Spezifizität (FPR)")
    ax.set_ylabel("Sensitivität (TPR)")
    ax.set_title("ROC-Kurve — Subluxation v2")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, "subluxation_eval_roc.png")
    plt.savefig(path)
    plt.close()
    print(f"  → {path}")

    # --- Precision-Recall ---
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in results:
        prec, rec, _ = precision_recall_curve(r["labels"], r["scores"])
        ax.plot(rec, prec, lw=1, alpha=0.4,
                label=f"Fold {r['fold']} (AP={r['AP']:.3f})")
    if valid.sum() > 0:
        prec_oof, rec_oof, _ = precision_recall_curve(
            oof_labels[valid], oof_scores[valid])
        ap_oof = average_precision_score(oof_labels[valid], oof_scores[valid])
        ax.plot(rec_oof, prec_oof, "k-", lw=2.5,
                label=f"OOF mean (AP={ap_oof:.3f})")
    prevalence = oof_labels[valid].mean() if valid.sum() > 0 else np.nan
    ax.axhline(prevalence, color="r", ls="--", lw=1,
               label=f"Baseline (Prävalenz={prevalence:.3f})")
    ax.set_xlabel("Recall (Sensitivität)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-Recall-Kurve — Subluxation v2")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, "subluxation_eval_pr.png")
    plt.savefig(path)
    plt.close()
    print(f"  → {path}")

    # --- Feature importance ---
    from annotation_analysis import V2_FEATURES as feats
    X = None
    for attr in ("full_df",):  # closure trick not needed; recompute inline
        pass
    print("  (Feature-Importance-Plot übersprungen — intern im Textbericht enthalten)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", default=DEFAULT_FEEDBACK,
                        help="Pfad zur Feedback-CSV")
    parser.add_argument("--sigma", type=float, default=2.0,
                        help="Glättungs-Sigma (default 2.0)")
    parser.add_argument("--folds", type=int, default=5,
                        help="Anzahl CV-Folds (default 5)")
    parser.add_argument("--plots", action="store_true",
                        help="ROC- und PR-Kurven als PNG speichern")
    args = parser.parse_args()

    # --- Load data ---
    if not os.path.exists(args.feedback):
        sys.exit(f"Feedback-Datei nicht gefunden: {args.feedback}")

    feedback_df = pd.read_csv(args.feedback)
    print(f"Feedback geladen: {len(feedback_df)} Zeilen aus {args.feedback}")

    try:
        df4d = pd.read_pickle(f"{DATA_DIR}/df4d_corrected_spine_measurements.pkl")
    except FileNotFoundError as e:
        sys.exit(f"Pickle nicht gefunden: {e}")

    # --- Build feature table ---
    print("Berechne Features … (kann einige Minuten dauern)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full_df = build_v2_training_table(feedback_df, df4d, sigma=args.sigma)

    if full_df.empty:
        sys.exit("Feature-Tabelle leer — prüfe Feedback-Datei und Pickle.")

    orig_df = full_df[~full_df["augmented"]].reset_index(drop=True)

    # --- Dataset statistics ---
    print_header("Datensatz")
    n_images = feedback_df["image_id"].nunique()
    n_segs   = len(orig_df)
    n_pos    = orig_df["label"].sum()
    prev     = n_pos / n_segs if n_segs else 0
    print(f"  Annotierte Bilder:           {n_images}")
    print(f"  Segmente (original):         {n_segs}")
    print(f"  Positive Segmente:           {n_pos}  ({prev:.1%})")
    print(f"  Augmentierte Trainingszeilen: {len(full_df)}")
    n_nan = orig_df[V2_FEATURES].isna().any(axis=1).sum()
    print(f"  Segmente mit NaN (Randwirb.): {n_nan}  ({n_nan/n_segs:.1%}) → werden je Fold dropped")
    print(f"  sigma={args.sigma}  folds={args.folds}")

    # --- Cross-validation ---
    print_header(f"{args.folds}-Fold GroupKFold CV  (Gruppen=image_id)")
    print("  Train: augmentierte Daten · Test: Originale, keine Augm.\n")
    results, oof_scores, oof_labels, groups = run_cv(orig_df, full_df, args.folds)

    if not results:
        sys.exit("Keine CV-Ergebnisse — zu wenig positive Samples?")

    print_cv_table(results)

    # OOF overall
    valid = ~np.isnan(oof_scores)
    if valid.sum() > 0:
        oof_auc = roc_auc_score(oof_labels[valid], oof_scores[valid])
        oof_ap  = average_precision_score(oof_labels[valid], oof_scores[valid])
        fpr_oof, tpr_oof, thrs_oof = roc_curve(oof_labels[valid], oof_scores[valid])
        thr_y, sens_y, spec_y = youden_threshold(fpr_oof, tpr_oof, thrs_oof)
        f1_y = f1_at(oof_labels[valid], oof_scores[valid], thr_y)
        print(f"\n  OOF gesamt (alle Folds zusammen)")
        print(f"  ROC-AUC:         {oof_auc:.4f}")
        print(f"  Avg Precision:   {oof_ap:.4f}  (Baseline: {prev:.4f})")
        print(f"  Lift (AP/Prev):  {oof_ap/prev:.2f}×")
        print(f"  Youden-Schwelle: {thr_y:.3f}")
        print(f"  Sensitivität:    {sens_y:.3f}")
        print(f"  Spezifizität:    {spec_y:.3f}")
        print(f"  F1:              {f1_y:.3f}")

    # --- Bootstrap CI ---
    print_header("Bootstrap 95%-KI für ROC-AUC  (n=1000, group-level)")
    if valid.sum() > 0:
        mean_b, lo, hi = bootstrap_group_auc(
            oof_labels[valid], oof_scores[valid], groups[valid])
        print(f"  AUC = {mean_b:.4f}  [{lo:.4f}, {hi:.4f}]")
    else:
        print("  Nicht genug OOF-Scores für Bootstrap.")

    # --- Feature importance ---
    print_header("Feature-Koeffizienten  (standardisiert, Full-Model-Fit)")
    print_feature_importance(full_df)

    # --- Plots ---
    if args.plots:
        print_header("Plots")
        save_plots(results, oof_scores, oof_labels, groups)

    print(f"\n{'─' * 60}\n")


if __name__ == "__main__":
    main()

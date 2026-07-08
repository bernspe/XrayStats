#!/usr/bin/env python3
"""
Fetches fresh image data from the xray-annotator API, runs the full analysis
pipeline and saves pickles consumed by explore_precision_outliers.ipynb.

Usage:
    python prepare_outlier_data.py --token YOUR_TOKEN
    python prepare_outlier_data.py          # prompts for token

Output files (in data/):
    df4d_corrected_spine_measurements.pkl   one row per measurement, k-Means corrected
    outlier_angles_df.pkl                   all cobbTypes incl. inflection @ 5/10/15°
    outlier_ranking_df.pkl                  per-angle-pair std, sorted descending
    img_urls.pkl                            image_id -> Google-signed URL (short-lived!)

Note: Google-signed image URLs expire. Open the notebook within ~1 h of running
this script. Re-run to refresh URLs.
"""

import argparse
import os
import pickle
import sys

import pandas as pd
import requests

import annotation_analysis as aa
from annotation_analysis import compute_outlier_ranking

API_URL = "https://xray-api.renecol.org"
DATA_DIR = "data"
INFLECTION_THRESHOLDS = [5, 10, 15]


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def fetch_data(token: str) -> pd.DataFrame:
    print("Fetching data from API …")
    response = requests.get(
        url=f"{API_URL}/api/imagedata/",
        headers={"Authorization": f"Bearer {token}"},
        params={"detailed": True, "complete": True},
        timeout=120,
    ).json()
    df = pd.DataFrame(response)
    print(f"  {len(df)} images, {df['measure_dataset'].map(len).sum()} measurements total")
    return df


def fetch_img_urls(token: str) -> dict:
    """image_id -> fresh Google-signed URL (valid 24h). Cheap re-fetch for when
    a cached img_urls.pkl has expired, without re-running the full pipeline."""
    df = fetch_data(token)
    return dict(zip(df["id"], df["img"]))


def build_pipeline(df: pd.DataFrame):
    print("Building df4d …")
    df4d = aa.build_df4d(df)
    print(f"  {len(df4d)} rows")

    print("Detecting label errors …")
    label_errors = aa.detect_label_errors(df4d)
    n_shift = (label_errors["error"] == "shift").sum()
    n_empty = (label_errors["error"] == "empty").sum()
    print(f"  {n_shift} shift errors, {n_empty} empty measurements")

    print("Applying corrections …")
    df4d_corrected = aa.apply_label_corrections(df4d, label_errors)
    return df4d, df4d_corrected, label_errors


def build_angles(df4d_corrected: pd.DataFrame) -> pd.DataFrame:
    """All cobbTypes: API angles, reconstructed, and inflection @ three thresholds."""
    print("Building angles_df …")
    base = aa.build_angles_df(df4d_corrected, include_reconstructed=True, include_inflection=False)
    print(f"  base cobbTypes: {sorted(base['cobbType'].unique())}")

    parts = [base]
    for thresh in INFLECTION_THRESHOLDS:
        inf = aa.build_angles_df(
            df4d_corrected,
            include_reconstructed=False,
            include_inflection=True,
            min_inflection_angle=float(thresh),
        )
        # build_angles_df always includes base API angles — keep only inflection rows
        inf = inf[inf["cobbType"].str.startswith("inflection_r")].copy()
        # Encode threshold in the cobbType name: inflection_r0 → inflection_r0@10
        inf["cobbType"] = inf["cobbType"].str.replace(
            r"^(inflection_r\d+)$", rf"\1@{thresh}", regex=True
        )
        print(f"  inflection {thresh}°: {len(inf)} rows, cobbTypes: {sorted(inf['cobbType'].unique())}")
        parts.append(inf)

    angles_df = pd.concat(parts, ignore_index=True)
    print(f"  total: {len(angles_df)} rows across {angles_df['cobbType'].nunique()} cobbTypes")
    return angles_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare outlier data for explore_precision_outliers.ipynb")
    parser.add_argument("--token", type=str, default="", help="Bearer token for xray-api.renecol.org")
    args = parser.parse_args()

    token = args.token.strip() or input("Enter Bearer token: ").strip()
    if not token:
        print("No token provided — aborting.")
        sys.exit(1)

    os.makedirs(DATA_DIR, exist_ok=True)

    df = fetch_data(token)
    _, df4d_corrected, _ = build_pipeline(df)

    # Extract Google-signed image URLs (short-lived) from the raw API response.
    # df4d_corrected also carries 'img' via build_df4d, but using the raw df
    # ensures every image_id is covered even if it has no measurements.
    img_urls: dict = dict(zip(df["id"], df["img"]))

    angles_df = build_angles(df4d_corrected)

    print("Computing outlier ranking …")
    outlier_df = compute_outlier_ranking(angles_df)
    print(f"  {len(outlier_df)} angle groups with ≥2 raters")

    print("Saving …")
    df4d_corrected.to_pickle(f"{DATA_DIR}/df4d_corrected_spine_measurements.pkl")
    angles_df.to_pickle(f"{DATA_DIR}/outlier_angles_df.pkl")
    outlier_df.to_pickle(f"{DATA_DIR}/outlier_ranking_df.pkl")
    with open(f"{DATA_DIR}/img_urls.pkl", "wb") as fh:
        pickle.dump(img_urls, fh)

    print("Done.")
    print(f"  {DATA_DIR}/df4d_corrected_spine_measurements.pkl")
    print(f"  {DATA_DIR}/outlier_angles_df.pkl")
    print(f"  {DATA_DIR}/outlier_ranking_df.pkl")
    print(f"  {DATA_DIR}/img_urls.pkl  ({len(img_urls)} URLs, valid ~1 h)")


if __name__ == "__main__":
    main()

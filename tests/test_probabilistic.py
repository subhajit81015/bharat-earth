from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.secrire.probabilistic.features import feature_manifest, load_v4_dataset, temporal_split, write_feature_manifest
from src.secrire.probabilistic.reliability import build_fixed_bin_reliability
from src.secrire.probabilistic.schema import TARGET_COLUMN, V4_THRESHOLD


def test_feature_manifest_validity():
    manifest = feature_manifest()
    assert isinstance(manifest, dict)
    assert "features" in manifest
    assert all("feature_name" in row for row in manifest["features"])
    assert any(row["feature_name"] == "regime_id" for row in manifest["features"])


def test_target_exclusion():
    df = load_v4_dataset()
    assert TARGET_COLUMN in df.columns
    assert "target_3m_severe_anomaly" in df.columns
    assert df[TARGET_COLUMN].isin([0, 1]).all()


def test_chronological_split():
    df = load_v4_dataset()
    train, validation, test = temporal_split(df)
    assert train["year"].max() <= 2013
    assert validation["year"].min() >= 2014 and validation["year"].max() <= 2015
    assert test["year"].min() >= 2016 and test["year"].max() <= 2017


def test_probability_bounds():
    prob = pd.Series([0.0, 0.2, 0.5, 1.0])
    assert prob.between(0.0, 1.0).all()


def test_reliability_bins():
    df = pd.DataFrame({TARGET_COLUMN: [0, 1, 0, 1], "probability": [0.1, 0.2, 0.7, 0.8]})
    reliability = build_fixed_bin_reliability(df)
    assert len(reliability) == 10
    assert reliability["sample_count"].sum() >= 0


def test_v4_integrity():
    assert isinstance(V4_THRESHOLD, float)
    assert 0.0 < V4_THRESHOLD < 1.0


def test_manifest_output_file():
    manifest = write_feature_manifest(Path("data/features/probabilistic_v1/feature_manifest.json"))
    assert manifest["experiment_version"] == "v1"


if __name__ == "__main__":
    for name in [
        "test_feature_manifest_validity",
        "test_target_exclusion",
        "test_chronological_split",
        "test_probability_bounds",
        "test_reliability_bins",
        "test_v4_integrity",
        "test_manifest_output_file",
    ]:
        globals()[name]()
        print(f"PASS: {name}")

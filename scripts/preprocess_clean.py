"""Round-6 review, P1 data items: the cleaned feature set for the sensitivity reruns.

Same raw files, same non-finite filter, same stratified 70/15/15 split with seed 42 as
src/data/preprocessor.py, with three changes the review asks for:
  - "Number" is dropped (it is 10 in 99.9 percent of rows);
  - "Protocol Type" (IANA numbers 0, 1, 6, 17) is one-hot encoded instead of min-max scaled;
  - exact duplicate records (same features and label) are removed BEFORE the split.
Writes data/processed_clean/ (paths from configs_clean/paths.yaml). The main pipeline is unchanged.

Run: PYTHONPATH=. .venv/bin/python scripts/preprocess_clean.py
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.data.preprocessor import encode_labels, load_raw_data, reshape_for_cnn, save_split
from src.utils.config import load_yaml, resolve_path

RAW_FEATURES = ["Header_Length", "Protocol Type", "Time_To_Live", "Rate", "ack_count", "syn_count",
                "Tot sum", "AVG", "IAT", "Number"]
PROTOCOLS = [0, 1, 6, 17]


def main() -> None:
    model_cfg, paths_cfg = load_yaml("configs_clean/model.yaml"), load_yaml("configs_clean/paths.yaml")
    ds = model_cfg["dataset"]
    out = resolve_path(paths_cfg["data"]["processed"]); out.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data(resolve_path(paths_cfg["data"]["raw"]), RAW_FEATURES, ds["label_column"],
                        ds["negative_label"])
    feats = raw[RAW_FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ok = ~(feats.isna().any(axis=1) | raw[ds["label_column"]].isna())
    df = feats.loc[ok].drop(columns=["Number"])
    df["label"] = encode_labels(raw.loc[ok, ds["label_column"]], ds["negative_label"])
    df["index"] = raw.index[ok]
    unknown = set(df["Protocol Type"].unique()) - set(PROTOCOLS)
    assert not unknown, f"unexpected protocol values {unknown}"
    for p in PROTOCOLS:
        df[f"proto_{p}"] = (df["Protocol Type"] == p).astype(np.float64)
    df = df.drop(columns=["Protocol Type"])
    features = list(ds["features"])                       # the order the model config declares

    before = len(df)
    dups = df.duplicated(subset=features + ["label"], keep="first")
    removed_by_class = df.loc[dups, "label"].value_counts().to_dict()
    df = df.loc[~dups]
    conflicting = int(df.duplicated(subset=features, keep=False).sum())   # same features, different label

    X, y, idx = df[features].to_numpy(np.float64), df["label"].to_numpy(np.int64), df["index"].to_numpy(np.int64)
    seed = int(model_cfg["random_seed"])
    tr_r, va_r, te_r = ds["train_ratio"], ds["val_ratio"], ds["test_ratio"]
    X_tr, X_tmp, y_tr, y_tmp, i_tr, i_tmp = train_test_split(X, y, idx, test_size=va_r + te_r, stratify=y,
                                                              random_state=seed)
    X_va, X_te, y_va, y_te, i_va, i_te = train_test_split(X_tmp, y_tmp, i_tmp, test_size=te_r / (va_r + te_r),
                                                          stratify=y_tmp, random_state=seed)
    scaler = MinMaxScaler().fit(X_tr)                     # one-hot columns stay 0/1 under min-max
    shape = tuple(model_cfg["model"]["input_shape"])
    sizes = {}
    for name, Xs, ys, ids in (("train", X_tr, y_tr, i_tr), ("val", X_va, y_va, i_va), ("test", X_te, y_te, i_te)):
        s = save_split(out / f"{name}.npz", reshape_for_cnn(scaler.transform(Xs).astype(np.float32), shape), ys, ids)
        sizes[name] = {"rows": s.size, "attacks": int(ys.sum())}
    pickle.dump(scaler, open(out / "scaler.pkl", "wb"))
    meta = {"features": features, "input_shape": list(shape), "raw_rows": int(len(raw)),
            "non_finite_dropped": int((~ok).sum()), "rows_before_dedup": before,
            "exact_duplicates_removed": int(dups.sum()),
            "duplicates_removed_by_class": {("attack" if k == 1 else "benign"): int(v) for k, v in removed_by_class.items()},
            "rows_with_conflicting_labels_kept": conflicting, "split_sizes": sizes,
            "changes": ["drop Number", "one-hot Protocol Type (0, 1, 6, 17)", "exact duplicates removed before split"]}
    (out / "preprocessing_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()

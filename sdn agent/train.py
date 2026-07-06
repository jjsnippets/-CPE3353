"""
train.py — Model Training Script
==================================
Phase: Training (run AFTER collect.py, BEFORE agent.py)

Purpose:
    Reads the labeled CSV produced by collect.py, trains a scikit-learn Pipeline
    (MinMaxScaler → KNeighborsClassifier), evaluates it on a held-out test split,
    and saves the fitted pipeline to disk as model.joblib.

    Using a Pipeline is the key design decision here: both the scaler and the
    classifier are packaged together into one object. This means agent.py can
    call pipeline.predict() directly — the scaling step is always applied
    automatically, so there is no risk of feeding unscaled features to KNN at
    inference time, which would silently produce wrong predictions.

Usage:
    python train.py
    python train.py --input traffic_data.csv --output model.joblib --k 3

    --input   : Path to the labeled CSV from collect.py (default: traffic_data.csv)
    --output  : Where to save the trained pipeline   (default: model.joblib)
    --k       : Number of neighbors for KNN          (default: 3)
    --test-size : Fraction of data held out for evaluation (default: 0.2)

Requirements:
    pip install scikit-learn pandas numpy joblib

Feature columns expected in the CSV (must match collect.py):
    icmp_count, icmp_rate, avg_pkt_size, iat_mean, iat_std
    label   — 0 = normal, 1 = attack

Output:
    model.joblib — a fitted sklearn Pipeline containing MinMaxScaler + KNN.
                   Load in agent.py with: pipeline = joblib.load("model.joblib")
"""

import argparse
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Feature columns — MUST match the columns written by collect.py exactly.
# If you add or remove a feature in collect.py, update this list too.
# ---------------------------------------------------------------------------
FEATURES = ["icmp_count", "icmp_rate", "avg_pkt_size", "iat_mean", "iat_std"]
LABEL    = "label"


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------
def load_and_validate(input_path: str) -> pd.DataFrame:
    """
    Reads the CSV and performs basic sanity checks before training.

    Checks:
        - All required columns are present
        - No NaN or infinite values in feature columns (common if a window had
          a single packet and IAT could not be computed — those rows get dropped)
        - Both class labels (0 and 1) are present in the dataset
        - Dataset is large enough to split (at least 10 rows)

    Returns:
        Cleaned DataFrame ready for training.
    """
    print(f"[TRAIN] Loading data from: {input_path}")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"[ERROR] File not found: '{input_path}'. "
              f"Run collect.py first to generate training data.")
        sys.exit(1)

    # Check required columns
    required = FEATURES + [LABEL]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns in CSV: {missing}")
        print(f"        Expected columns: {required}")
        print(f"        Found columns:    {list(df.columns)}")
        sys.exit(1)

    initial_len = len(df)

    # Drop rows with NaN or infinite values in feature columns
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=FEATURES, inplace=True)
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"[WARN]  Dropped {dropped} rows with NaN/Inf values "
              f"(likely single-packet windows with undefined IAT).")

    # Validate class balance
    class_counts = df[LABEL].value_counts().sort_index()
    print(f"\n[TRAIN] Class distribution:")
    for cls, cnt in class_counts.items():
        label_name = "normal" if cls == 0 else "attack"
        print(f"        Label {cls} ({label_name:6s}): {cnt:4d} rows  "
              f"({cnt/len(df)*100:.1f}%)")

    if len(class_counts) < 2:
        print("[ERROR] Dataset contains only one class. "
              "Collect both normal (label=0) and attack (label=1) data.")
        sys.exit(1)

    # Warn about class imbalance — KNN is distance-based and biases toward
    # the majority class when imbalance is severe (> 3:1 ratio)
    counts = class_counts.values
    ratio  = max(counts) / min(counts)
    if ratio > 3.0:
        print(f"\n[WARN]  Class imbalance detected (ratio {ratio:.1f}:1). "
              f"Consider collecting more data for the minority class.")

    if len(df) < 10:
        print(f"[ERROR] Only {len(df)} rows found. Collect more data before training.")
        sys.exit(1)

    print(f"\n[TRAIN] Total usable rows: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Model building and training
# ---------------------------------------------------------------------------
def build_and_train(df: pd.DataFrame, k: int, test_size: float, output_path: str):
    """
    Constructs the sklearn Pipeline, trains it, evaluates it, and saves it.

    Pipeline design:
        Step 1 — MinMaxScaler: scales each feature to [0, 1].
            This is MANDATORY for KNN. KNN classifies by Euclidean distance
            between feature vectors. Without scaling, icmp_count (0–1000s) would
            dominate the distance calculation and make iat_mean (0.0–1.0) nearly
            invisible — effectively turning KNN into a packet-count threshold.
            MinMaxScaler is preferred over StandardScaler here because the feature
            ranges have physical meaning (0 packets min, some max) and are not
            normally distributed.

        Step 2 — KNeighborsClassifier(n_neighbors=k):
            Binary classification: 0 = normal, 1 = attack. k=3 is the default.
            Odd values of k avoid tie-breaking in binary classification.
            At inference time the pipeline receives raw (unscaled) feature values
            and automatically applies the fitted scaler before calling KNN.predict().

    Evaluation:
        80/20 stratified train-test split (stratify=y preserves class ratios in
        both splits, important when the dataset is moderately imbalanced).
        Classification report: shows per-class precision, recall, and F1.
        Recall for class 1 (attack) is the most important metric — a missed attack
        is worse than a false alarm in a security context.
        Confusion matrix: rows = true labels, columns = predicted labels.
        5-fold cross-validation: gives a more stable accuracy estimate across the
        whole dataset, compensating for variance in a single 80/20 split.
    """

    X = df[FEATURES].values
    y = df[LABEL].values

    # -----------------------------------------------------------------------
    # Train / test split — stratified to preserve class proportions
    # -----------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = test_size,
        random_state = 42,          # fixed seed for reproducibility
        stratify     = y,           # preserve class ratio in both splits
    )
    print(f"\n[TRAIN] Train set: {len(X_train)} rows | "
          f"Test set: {len(X_test)} rows (stratified {int((1-test_size)*100)}/{int(test_size*100)} split)")

    # -----------------------------------------------------------------------
    # Build the pipeline — scaler and classifier coupled together
    # -----------------------------------------------------------------------
    pipeline = make_pipeline(
        MinMaxScaler(),                         # step 1: scale to [0, 1]
        KNeighborsClassifier(n_neighbors=k),    # step 2: KNN classifier
    )

    # -----------------------------------------------------------------------
    # Fit on training data only.
    # IMPORTANT: The scaler is fitted on X_train only. If it were fitted on
    # the full dataset before splitting, information about the test set would
    # leak into the scaler's min/max bounds — this is called data leakage and
    # would inflate reported accuracy. make_pipeline handles this correctly.
    # -----------------------------------------------------------------------
    print(f"[TRAIN] Fitting pipeline (MinMaxScaler → KNN, k={k})...")
    pipeline.fit(X_train, y_train)

    # -----------------------------------------------------------------------
    # Evaluation on the held-out test set
    # -----------------------------------------------------------------------
    y_pred = pipeline.predict(X_test)

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(
        y_test, y_pred,
        target_names=["normal (0)", "attack (1)"],
        digits=4,
    ))

    print("CONFUSION MATRIX")
    print("="*60)
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Predicted →  Normal   Attack")
    print(f"  True Normal: {cm[0][0]:6d}   {cm[0][1]:6d}")
    print(f"  True Attack: {cm[1][0]:6d}   {cm[1][1]:6d}")

    # Derive key metrics manually for a clear summary
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # also called detection rate
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # false positive rate
    print(f"\n  Attack Detection Rate (Recall): {recall*100:.2f}%")
    print(f"  False Positive Rate:            {fpr*100:.2f}%")
    print(f"  Precision:                      {precision*100:.2f}%")

    # -----------------------------------------------------------------------
    # Cross-validation — trains and evaluates on 5 different folds of the
    # ENTIRE dataset for a more stable accuracy estimate
    # -----------------------------------------------------------------------
    print("\n[TRAIN] Running 5-fold cross-validation on full dataset...")
    cv_scores = cross_val_score(
        make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors=k)),
        X, y,
        cv      = 5,
        scoring = "recall",         # recall for class 1 — we care most about this
    )
    print(f"  Cross-val Attack Recall: {cv_scores.mean()*100:.2f}% "
          f"(± {cv_scores.std()*100:.2f}%)")

    # Guidance on interpreting results
    print("\n[TRAIN] Interpretation guide:")
    print("  - Attack Recall < 90%  → consider lowering k, or collecting more attack data")
    print("  - False Positive > 10% → consider raising k, or collecting more normal data")
    print("  - If results look poor, inspect whether collect.py --label values were correct")

    # -----------------------------------------------------------------------
    # Save the fitted pipeline to disk
    # agent.py loads this file at startup with: pipeline = joblib.load("model.joblib")
    # The saved file includes the fitted MinMaxScaler bounds — agent.py must NOT
    # re-fit the scaler on live data, only transform with it.
    # -----------------------------------------------------------------------
    print(f"\n[TRAIN] Saving pipeline to: {output_path}")
    joblib.dump(pipeline, output_path)
    print(f"[TRAIN] Done. Model saved successfully.")
    print(f"        To use: pipeline = joblib.load('{output_path}')")
    print(f"        Then:   pred = pipeline.predict([[count, rate, avg_size, iat_mean, iat_std]])")

    # Print feature importance proxy — for KNN this is the mean absolute value
    # of each feature after scaling (normalized by the scaler already fitted).
    # This is not a true importance score but gives a rough sense of each
    # feature's contribution to the distance metric.
    scaler = pipeline.named_steps["minmaxscaler"]
    print(f"\n[TRAIN] Feature scaling bounds (min → max from training data):")
    for fname, fmin, fmax in zip(FEATURES, scaler.data_min_, scaler.data_max_):
        print(f"  {fname:15s}: {fmin:.4f} → {fmax:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SDN SOAR — Model Training Script (train.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str, default="traffic_data.csv",
        help="Path to labeled CSV from collect.py (default: traffic_data.csv)"
    )
    parser.add_argument(
        "--output", type=str, default="model.joblib",
        help="Output path for the trained pipeline (default: model.joblib)"
    )
    parser.add_argument(
        "--k", type=int, default=3,
        help="Number of nearest neighbors for KNN (default: 3, must be odd)"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data to hold out for evaluation (default: 0.2)"
    )

    args = parser.parse_args()

    # Validate k is odd — even k can cause ties in binary classification
    if args.k % 2 == 0:
        print(f"[WARN]  k={args.k} is even. Using k={args.k + 1} instead to avoid "
              f"tie-breaking issues in binary classification.")
        args.k += 1

    df = load_and_validate(args.input)
    build_and_train(
        df          = df,
        k           = args.k,
        test_size   = args.test_size,
        output_path = args.output,
    )

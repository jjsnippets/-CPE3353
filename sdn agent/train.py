"""
train.py: Model Training Script

Purpose:
    Reads the labeled CSV produced by collect.py, trains a scikit-learn Pipeline
    (MinMaxScaler → KNeighborsClassifier), evaluates it on a held-out test split,
    and saves the fitted pipeline to disk as model.joblib.

Usage:
    python train.py
    python train.py --input traffic_data.csv --output model.joblib --k 3

    --input     : Path to the labeled CSV from collect.py (default: traffic_data.csv)
    --output    : Where to save the trained pipeline   (default: model.joblib)
    --k         : Number of neighbors for KNN          (default: 3)
    --test-size : Fraction of data held out for evaluation (default: 0.2)

Requirements:
    pip install scikit-learn pandas numpy joblib
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

# Feature columns
# Must match with those in other files exactly
FEATURES = ["icmp_count", "icmp_rate", "avg_pkt_size", "iat_mean", "iat_std"]
LABEL    = "label"

def load_and_validate(input_path: str) -> pd.DataFrame:
    """
    Data loading and validation.
    """
    print(f"Loading data from: {input_path}")

    # Check if file exists
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"[ERROR] File not found: '{input_path}'. ")
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
        print(f"[WARN]  Dropped {dropped} rows with NaN/Inf values ")

    # Check dataset size
    if len(df) < 10:
        print(f"[ERROR] Only {len(df)} rows found. Collect more data before training.")
        sys.exit(1)

    # Check if both classes are present
    class_counts = df[LABEL].value_counts().sort_index()
    if len(class_counts) < 2:
        print("[ERROR] Dataset contains only one class.")
        sys.exit(1)

    # Class distribution summary
    print(f"\nClass distribution:")
    for cls, cnt in class_counts.items():
        label_name = "normal" if cls == 0 else "attack"
        print(f"        Label {cls} ({label_name:6s}): {cnt:4d} rows  "
              f"({cnt/len(df)*100:.1f}%)")
    print(f"Total usable rows: {len(df)}")

    # Warn about class imbalance
    counts = class_counts.values
    ratio  = max(counts) / min(counts)
    if ratio > 3.0:
        print(f"\n[WARN]  Class imbalance detected (ratio {ratio:.1f}:1). "
              f"Consider collecting more data for the minority class.")

    return df

def build_and_train(df: pd.DataFrame, k: int, test_size: float, output_path: str):
    """
    Model building, training, evaluation, and saving.
     - MinMaxScaler: scales each feature to [0, 1].
     - KNeighborsClassifier(n_neighbors=k)
    """

    X = df[FEATURES].values
    y = df[LABEL].values

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = test_size,
        random_state = 67,          # fixed seed
        stratify     = y,           # preserve class ratio in both splits
    )
    print(f"\nTrain set: {len(X_train)} rows"
          f"\n Test set: {len(X_test)} rows (stratified {int((1-test_size)*100)}/{int(test_size*100)} split)")

    # Scaler and classifier coupled together
    pipeline = make_pipeline(
        MinMaxScaler(),                         # Scale to [0, 1]
        KNeighborsClassifier(n_neighbors=k),    # KNN classifier
    )

    # Fit on training data only.
    print(f"[TRAIN] Fitting pipeline (MinMaxScaler → KNN, k={k})...")
    pipeline.fit(X_train, y_train)

    # Evaluation on the held-out test set
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

    # Key metrics summary
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # also called detection rate
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # false positive rate
    print(f"\n  Attack Detection Rate (Recall): {recall*100:.2f}%")
    print(f"  False Positive Rate:            {fpr*100:.2f}%")
    print(f"  Precision:                      {precision*100:.2f}%")

    cv_scores = cross_val_score(
        make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors=k)),
        X, y,
        cv      = 5,
        scoring = "recall",         # recall for class 1
    )
    print(f"  Cross-val Attack Recall: {cv_scores.mean()*100:.2f}% "
          f"(± {cv_scores.std()*100:.2f}%)")

    # Save the fitted pipeline to disk
    joblib.dump(pipeline, output_path)
    print(f"Model saved successfully to: {output_path}.")

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SDN SOAR Agent: Model Training Script",
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

    # Validate k is odd
    if args.k % 2 == 0:
        print(f"[WARN]  k={args.k} is even. Using k={args.k + 1} instead to avoid "
              f"tie-breaking issues in binary classification.")
        args.k += 1

    # Start the training process
    df = load_and_validate(args.input)
    build_and_train(
        df          = df,
        k           = args.k,
        test_size   = args.test_size,
        output_path = args.output,
    )

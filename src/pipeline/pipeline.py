import argparse
import joblib
import yaml
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report

from utils import engineer_features


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── 1. LOAD ───────────────────────────────────────────────────────────────────

def load_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)


# ── 2. CLEAN ──────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.drop(columns=config["features"]["drop"], errors="ignore")
    return df


# ── 3. FEATURES ───────────────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = engineer_features(df)
    return df


# ── 4. TRAIN ──────────────────────────────────────────────────────────────────

def build_pipeline(config: dict) -> Pipeline:
    params = config["model"]
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        random_state=params["random_state"],
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def train(config_path: str, data_path: str):
    config = load_config(config_path)

    df = load_data(data_path)
    df = clean(df, config)
    df = add_features(df)

    target = config["training"]["target"]
    X = df.drop(columns=[target])
    y = df[target]

    # Encode categoricals
    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["training"]["test_size"],
        random_state=42,
    )

    pipeline = build_pipeline(config)
    pipeline.fit(X_train, y_train)

    # ── 5. EVALUATE ───────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")
    print(classification_report(y_test, y_pred))

    cv_scores = cross_val_score(
        pipeline, X, y,
        cv=config["training"]["cv_folds"],
        scoring=config["training"]["scoring"],
    )
    print(f"CV {config['training']['scoring']}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save model
    model_path = Path("models/final_model.joblib")
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Model saved → {model_path}")


# ── 6. BATCH PREDICT ──────────────────────────────────────────────────────────

def batch_predict(config_path: str, input_path: str, output_path: str):
    config = load_config(config_path)

    pipeline = joblib.load("models/final_model.joblib")

    df = load_data(input_path)
    df = clean(df, config)
    df = add_features(df)

    if config["training"]["target"] in df.columns:
        df = df.drop(columns=[config["training"]["target"]])

    df = pd.get_dummies(df)

    predictions = pipeline.predict(df)
    probabilities = pipeline.predict_proba(df)[:, 1]

    output = df.copy()
    output["prediction"] = predictions
    output["probability"] = probabilities.round(4)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Predictions saved → {output_path}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data",   default="data/raw/titanic.csv")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--input",  default="data/raw/titanic.csv")
    parser.add_argument("--output", default="models/batch_outputs/predictions.csv")
    args = parser.parse_args()

    if args.predict:
        batch_predict(args.config, args.input, args.output)
    else:
        train(args.config, args.data)

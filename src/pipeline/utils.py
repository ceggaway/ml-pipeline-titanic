import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def extract_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract helper features from raw text/mixed columns before dropping them."""
    df = df.copy()
    df["Name"]       = df["Name"].astype(str)
    df["Cabin"]      = df["Cabin"].astype(str)
    df["Name_title"] = df["Name"].str.extract(r",\s*([^.]+)\.")
    df["Cabin_deck"] = df["Cabin"].str[0].where(df["Cabin"] != "nan", other=float("nan"))
    return df


def group_titles(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Normalise and group rare Name_title values."""
    df = df.copy()
    mapping     = config["features"]["title_mapping"]
    keep        = set(config["features"]["title_keep"])
    rare_label  = config["features"]["title_rare_label"]
    df["Name_title"] = df["Name_title"].replace(mapping)
    df["Name_title"] = df["Name_title"].apply(lambda t: t if t in keep else rare_label)
    return df


def impute(df: pd.DataFrame, config: dict, train_stats: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Impute missing values. Fit stats on train, apply to both train and test.

    Parameters
    ----------
    df          : DataFrame to impute (in-place copy)
    config      : pipeline config dict
    train_stats : pre-computed stats dict (None when fitting on train set)

    Returns
    -------
    df          : imputed DataFrame
    stats       : dict of imputation values (pass to test set call)
    """
    df    = df.copy()
    rules = config["preprocessing"]["impute"]
    stats = train_stats or {}

    for col, strategy in rules.items():
        if col not in df.columns:
            continue
        if strategy == "median":
            val = stats.get(col, df[col].median())
        elif strategy == "mode":
            val = stats.get(col, df[col].mode()[0])
        else:
            val = strategy   # literal fill value e.g. "Unknown"
        df[col] = df[col].fillna(val)
        stats[col] = val

    return df, stats


def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add all engineered features defined in config."""
    df = df.copy()

    # Family features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)

    # Log transform
    df["LogFare"] = np.log1p(df["Fare"])

    # Age binning
    bins   = config["features"]["age_bins"]
    labels = config["features"]["age_labels"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)

    # Interaction and ratio features
    df["Pclass_x_Fare"] = df["Pclass"] * df["Fare"]
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    return df


def drop_raw_columns(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Drop columns listed in config features.drop."""
    return df.drop(columns=config["features"]["drop"], errors="ignore")


def encode(X_train: pd.DataFrame, X_test: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode categorical columns. Align test to train columns."""
    encode_cols = config["preprocessing"]["encode"]
    X_train = pd.get_dummies(X_train, columns=encode_cols, drop_first=True)
    X_test  = pd.get_dummies(X_test,  columns=encode_cols, drop_first=True)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)
    return X_train, X_test


def scale(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    config: dict,
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit scaler on train, apply to both. Returns updated DataFrames and fitted scaler.
    Pass a pre-fitted scaler for inference (test/batch) to avoid refit.
    """
    cols = [c for c in config["preprocessing"]["scale"] if c in X_train.columns]
    if scaler is None:
        scaler = StandardScaler()
        X_train[cols] = scaler.fit_transform(X_train[cols])
    else:
        X_train[cols] = scaler.transform(X_train[cols])
    X_test[cols] = scaler.transform(X_test[cols])
    return X_train, X_test, scaler


def preprocess_train(
    df: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict, StandardScaler]:
    """
    Full preprocessing pipeline for training. Returns train/test splits and
    fitted artefacts (train_stats, scaler) needed for inference.
    """
    from sklearn.model_selection import train_test_split

    df = extract_raw_features(df)
    df = group_titles(df, config)

    target = config["training"]["target"]
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["training"]["test_size"],
        random_state=42,
        stratify=y,
    )

    X_train, train_stats = impute(X_train, config)
    X_test,  _           = impute(X_test,  config, train_stats)

    X_train = engineer_features(X_train, config)
    X_test  = engineer_features(X_test,  config)

    X_train = drop_raw_columns(X_train, config)
    X_test  = drop_raw_columns(X_test,  config)

    X_train, X_test = encode(X_train, X_test, config)
    X_train, X_test, scaler = scale(X_train, X_test, config)

    return X_train, X_test, y_train, y_test, train_stats, scaler


def preprocess_inference(
    df: pd.DataFrame,
    config: dict,
    train_stats: dict,
    scaler: StandardScaler,
    train_columns: list[str],
) -> pd.DataFrame:
    """
    Preprocessing for batch prediction. Uses artefacts fitted on training data.
    """
    df = extract_raw_features(df)
    df = group_titles(df, config)

    target = config["training"]["target"]
    if target in df.columns:
        df = df.drop(columns=[target])

    df, _ = impute(df, config, train_stats)
    df    = engineer_features(df, config)
    df    = drop_raw_columns(df, config)

    encode_cols = config["preprocessing"]["encode"]
    df = pd.get_dummies(df, columns=encode_cols, drop_first=True)
    df = df.reindex(columns=train_columns, fill_value=0)

    scale_cols = [c for c in config["preprocessing"]["scale"] if c in df.columns]
    df[scale_cols] = scaler.transform(df[scale_cols])

    return df

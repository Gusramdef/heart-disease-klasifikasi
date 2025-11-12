"""
utils/preprocessing.py
Fungsi-fungsi untuk memuat data, pembersihan dasar, pembagian fitur/target, dan scaling.
Bahasa: Indonesia
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path: str) -> pd.DataFrame:
    """Membaca CSV dan mengembalikan DataFrame."""
    return pd.read_csv(path)

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Rename kolom target bila perlu dan kembalikan salinan dataframe."""
    df = df.copy()
    if 'condition' not in df.columns and 'target' in df.columns:
        df = df.rename(columns={'target': 'condition'})
    return df

def split_features_target(df: pd.DataFrame, target: str = 'condition'):
    """Pisah fitur (X) dan target (y)."""
    X = df.drop(columns=[target])
    y = df[target].copy()
    return X, y

def scale_numeric(X_train: pd.DataFrame, X_test: pd.DataFrame, numeric_cols: list):
    """
    Scale kolom numerik menggunakan StandardScaler.
    Kembalikan X_train_scaled, X_test_scaled, dan objek scaler.
    """
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    return X_train_scaled, X_test_scaled, scaler

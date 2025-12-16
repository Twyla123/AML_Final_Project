def _normalize_yf_download(df: pd.DataFrame) -> pd.DataFrame:

    if df is None or df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):

        df.columns = df.columns.get_level_values(0)

    return df

def _safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, (pd.Series, pd.DataFrame)):
            return None
        return float(x)
    except Exception:
        return None

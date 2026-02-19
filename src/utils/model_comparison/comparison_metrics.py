import pandas as pd
import numpy as np


def expand_lists_to_cols(df):
    expanded_data = {}

    for col in df.columns:
        expanded_data[f"{col}_x"] = df[col].apply(
            lambda val: val[0] if isinstance(val, (tuple, list, np.ndarray)) else np.nan
        )
        expanded_data[f"{col}_y"] = df[col].apply(
            lambda val: val[1] if isinstance(val, (tuple, list, np.ndarray)) else np.nan
        )

    return pd.DataFrame(expanded_data)

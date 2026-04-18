from features.registry import FEATURES


def assemble_features(df, groups):
    cols = []

    for g in groups:
        if g not in FEATURES:
            raise ValueError(f"Unknown feature group: {g}")
        cols.extend(FEATURES[g])

    # Remove duplicates while preserving order
    cols = list(dict.fromkeys(cols))

    return df[cols].copy()

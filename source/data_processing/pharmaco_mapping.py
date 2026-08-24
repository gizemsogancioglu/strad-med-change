import pandas as pd
import re

# ---------------------------------------------------------------------------
# Mapping dictionaries
# ---------------------------------------------------------------------------

# 1. Exact-match lookup (highest priority)
EXACT_MAP = {
    # --- UM ---
    "UM":               "UM",
    "*17/*17 = UM":     "UM",
    "*17/*17":          "UM",
    "(*1/*1)x2":        "UM",
    "(*1/*2)x2 = UM":   "UM",
    "(*1x2)/*5":        "UM",

    # --- EM/NM ---
    "EM":               "EM/NM",
    "NM":               "EM/NM",
    "*1/*1 = EM":       "EM/NM",
    "*1/*17 = EM":      "EM/NM",
    "*1/*1 = NM":       "EM/NM",
    "*2/*9 = NM":       "EM/NM",
    "*1/*41 = NM":      "EM/NM",
    "*1/*1":            "EM/NM",
    "*1/*17":           "EM/NM",
    "*1/*2 = NM":       "EM/NM",
    "AS 2":             "EM/NM",   # CYP2D6 activity score 2 → normal function

    # --- IM ---
    "IM":               "IM",
    "*1/*2 = IM":       "IM",
    "*1/*3 = IM":       "IM",
    "*1/*4 = IM":       "IM",
    "*1/*5 = IM":       "IM",
    "*2/*4 = IM":       "IM",
    "*2/*17 = IM":      "IM",
    "*41/*41 = IM":     "IM",
    "*1/*41":           "IM",
    "*2/*17":           "IM",

    # --- PM ---
    "PM":               "PM",
    "*2/*3 = PM":       "PM",
    "*4/*4":            "PM",
    "*2/*2":            "PM",
    "*2/*3":            "PM",
    "*2/*4":            "PM",
    "*2/*41":           "PM",
    "*41/*41":          "PM",
    "*10/*10":          "PM",
    "*1/*3":            "PM",
    "*1/*10":           "PM",
    "*1/*4":            "PM",
    "*1/*5":            "PM",
    "*1/1173C>T":       "PM",
    "1173C>T/1173C>T":  "PM",
    "*1/*2":            "PM",

    # --- Other (expression flags) ---
    "Expressor":        "Other",
    "Non-expr":         "Other",
    "NF":               "Other",
    "PF":               "Other",
    "DF":               "Other",

    # --- Unknown / missing ---
    "Zie opm":          "Unknown",
    "<Memo>":           "Unknown",
    "-volgt-":          "Unknown",
}

# 2. Suffix-based rules: if the value ends with "= XX", extract the phenotype
SUFFIX_PHENOTYPE_MAP = {
    "UM": "UM",
    "EM": "EM/NM",
    "NM": "EM/NM",
    "IM": "IM",
    "PM": "PM",
}

# ---------------------------------------------------------------------------
# Core mapping function
# ---------------------------------------------------------------------------

def map_pgx_phenotype(value: str) -> str:
    """
    Map a raw pharmacogenetics value to a standardised phenotype category.

    Categories
    ----------
    UM       – Ultrarapid metabolizer
    EM/NM    – Extensive / Normal metabolizer
    IM       – Intermediate metabolizer
    PM       – Poor metabolizer
    Other    – Expression-level flags (Expressor, NF, PF, DF, Non-expr)
    Unknown  – Missing, pending, or ambiguous entries

    Parameters
    ----------
    value : str
        A single raw value from df['value'].

    Returns
    -------
    str
        One of: 'UM', 'EM/NM', 'IM', 'PM', 'Other', 'Unknown'
    """
    if pd.isna(value):
        return "Unknown"

    v = str(value).strip()

    # Step 1 – exact match
    if v in EXACT_MAP:
        return EXACT_MAP[v]

    # Step 2 – stated phenotype suffix, e.g. "*1/*2 = IM"
    suffix_match = re.search(r"=\s*([A-Z]+)\s*$", v)
    if suffix_match:
        stated = suffix_match.group(1).upper()
        if stated in SUFFIX_PHENOTYPE_MAP:
            return SUFFIX_PHENOTYPE_MAP[stated]

    # Step 3 – gene-duplication patterns without explicit phenotype,
    #           e.g. "(*1/*1)x3", "(*2/*17)x2"
    if re.search(r"x\d+", v, re.IGNORECASE):
        return "UM"

    # Step 4 – fallback
    return "Unknown"


# ---------------------------------------------------------------------------
# Apply to a DataFrame
# ---------------------------------------------------------------------------

def apply_pgx_mapping(df: pd.DataFrame, col: str = "value") -> pd.DataFrame:
    """
    Add a 'phenotype' column to *df* by mapping df[col] through
    map_pgx_phenotype().

    Parameters
    ----------
    df  : pd.DataFrame  – input dataframe
    col : str           – column containing raw PGx values (default 'value')

    Returns
    -------
    pd.DataFrame with an added 'phenotype' column.
    """
    df = df.copy()
    df["phenotype"] = df[col].apply(map_pgx_phenotype)
    return df


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_values = [
        'IM', 'UM', 'EM', '*1/*1 = EM', 'Zie opm', '*1/*17 = EM', '*1/*17',
        '*1/*41', '<Memo>', '*1/*1', '*10/*10', '*2/*17', '*1/*2',
        '*2/*17 = IM', '*2/*4', 'Non-expr', '*2/*9 = NM', 'NM', 'AS 2',
        'NF', '(*1/*1)x2', '*1/*1 = NM', '*17/*17 = UM', '*17/*17',
        '*1/*2 = IM', '*1/*3', '*41/*41', '*2/*2', '*1/*2 = NM',
        '*1/1173C>T', 'PM', '*1/*3 = IM', '1173C>T/1173C>T', 'PF',
        'Expressor', 'DF', '*1/*4 = IM', '*41/*41 = IM', '*4/*4', '*1/*10',
        '(*1/*2)x2 = UM', '*1/*4', '*1/*5 = IM', '*2/*3 = PM', '*2/*3',
        '*2/*41', '(*1x2)/*5', '*1/*41 = NM', '*2/*4 = IM', '-volgt-',
    ]

    df_test = pd.DataFrame({"value": test_values})
    df_test = apply_pgx_mapping(df_test)

    # Show counts per category
    print(df_test["phenotype"].value_counts().to_string())
    print()
    print(df_test.to_string(index=False))
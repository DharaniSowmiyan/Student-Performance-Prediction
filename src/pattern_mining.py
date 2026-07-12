import pandas as pd
from prefixspan import PrefixSpan


def load_student_sequences(processed_path: str) -> pd.DataFrame:
    return pd.read_csv(processed_path + "student_sequences.csv")


def split_by_group(student_sequences: pd.DataFrame):
   
    high_df = student_sequences[student_sequences["performance_group"] == "High"]
    low_df  = student_sequences[student_sequences["performance_group"] == "Low"]

    high_sequences = [seq.split(",") for seq in high_df["sequence"]]
    low_sequences  = [seq.split(",") for seq in low_df["sequence"]]

    return high_sequences, low_sequences



def mine_patterns(sequences: list, min_support: int, max_length: int = 4) -> list:
  
    ps = PrefixSpan(sequences)
    patterns = ps.frequent(
        min_support,
        filter=lambda patt, matches: len(patt) <= max_length
    )
    return patterns


def build_pattern_table(
    patterns_high: list,
    patterns_low: list,
    n_high: int,
    n_low: int,
) -> pd.DataFrame:
   
    high_dict = {tuple(patt): sup for sup, patt in patterns_high}
    low_dict  = {tuple(patt): sup for sup, patt in patterns_low}

    all_patterns = set(high_dict.keys()) | set(low_dict.keys())

    rows = []
    for patt in all_patterns:
        sup_high = high_dict.get(patt, 0)
        sup_low  = low_dict.get(patt, 0)

        pct_high = round(sup_high / n_high * 100, 2)
        pct_low  = round(sup_low  / n_low  * 100, 2)

        rows.append({
            "pattern":          ",".join(patt),
            "support_high":     sup_high,
            "support_low":      sup_low,
            "support_high_pct": pct_high,
            "support_low_pct":  pct_low,
            "difference":       round(pct_high - pct_low, 2),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("difference", ascending=False).reset_index(drop=True)
    return df


def select_discriminative_patterns(
    pattern_table: pd.DataFrame,
    top_n: int = 10,
    min_diff: float = 5.0,
) -> pd.DataFrame:
    
    filtered = pattern_table[pattern_table["difference"].abs() >= min_diff].copy()

    top_high = (
        filtered[filtered["difference"] > 0]
        .nlargest(top_n, "difference")
        .copy()
    )
    top_high["group"] = "High"

    top_low = (
        filtered[filtered["difference"] < 0]
        .nsmallest(top_n, "difference")
        .copy()
    )
    top_low["group"] = "Low"

    selected = pd.concat([top_high, top_low], ignore_index=True)
    selected = selected[[
        "pattern", "support_high_pct", "support_low_pct", "difference", "group"
    ]]

    return selected


def run_pattern_mining(
    processed_path: str,
    results_path: str,
    min_support_pct: float = 0.30,
    max_length: int = 4,
) -> pd.DataFrame:
    
    student_sequences = load_student_sequences(processed_path)
    high_sequences, low_sequences = split_by_group(student_sequences)

    n_high = len(high_sequences)
    n_low  = len(low_sequences)

    min_sup_high = int(min_support_pct * n_high)
    min_sup_low  = int(min_support_pct * n_low)

    print(f"High group: {n_high:,} students  |  min_support = {min_sup_high:,}")
    print(f"Low  group: {n_low:,} students  |  min_support = {min_sup_low:,}")
    print()

    print("Mining High patterns...")
    patterns_high = mine_patterns(high_sequences, min_sup_high, max_length)
    print(f"  Found {len(patterns_high):,} patterns")

    print("Mining Low patterns...")
    patterns_low = mine_patterns(low_sequences, min_sup_low, max_length)
    print(f"  Found {len(patterns_low):,} patterns")
    print()

    pattern_table = build_pattern_table(patterns_high, patterns_low, n_high, n_low)

    output_path = results_path + "patterns.csv"
    pattern_table.to_csv(output_path, index=False)
    print(f"Saved {len(pattern_table):,} patterns to {output_path}")

    return pattern_table
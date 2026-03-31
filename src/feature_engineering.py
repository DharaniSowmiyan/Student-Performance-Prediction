import pandas as pd
import numpy as np


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def contains_subsequence(sequence: list, pattern: list) -> bool:
    """
    Returns True if pattern appears as a subsequence anywhere in sequence.
    Order matters but items do not need to be consecutive.

    Example:
        sequence = [A, B, C, D]
        pattern  = [A, C]   → True  (A comes before C)
        pattern  = [C, A]   → False (C does not come before A)
    """
    it = iter(sequence)
    return all(item in it for item in pattern)


# ─── PATTERN FEATURES ─────────────────────────────────────────────────────────
def build_pattern_features(
    student_sequences: pd.DataFrame,
    selected_patterns: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each student, checks whether each selected pattern appears
    as a subsequence in their activity sequence.

    Returns a DataFrame:
        rows    = students
        columns = one binary column per pattern (0 / 1)
    """
    rows = []

    for _, row in student_sequences.iterrows():
        seq = row["sequence"].split(",")
        student_row = {"id_student": row["id_student"]}

        for _, pat_row in selected_patterns.iterrows():
            pattern  = pat_row["pattern"].split(",")
            col_name = "pat_" + "_".join(pattern)
            student_row[col_name] = int(contains_subsequence(seq, pattern))

        rows.append(student_row)

    return pd.DataFrame(rows)


# ─── STATIC FEATURES ──────────────────────────────────────────────────────────
def build_static_features(clean_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Computes per-student aggregate features from clean_logs.csv.

    Features:
        total_clicks           — sum of all clicks
        total_interactions     — total number of interaction rows
        unique_activity_types  — number of distinct activity categories visited
        pre_course_interactions— interactions before course start (date < 0)
        quiz_interactions      — rows where activity_category == Quiz
        study_interactions     — rows where activity_category == StudyMaterial
        discussion_interactions— rows where activity_category == Discussion
        navigation_interactions— rows where activity_category == Navigation
        external_interactions  — rows where activity_category == External
    """
    grp = clean_logs.groupby("id_student")

    total_clicks = grp["sum_click"].sum().rename("total_clicks")
    total_interactions = grp.size().rename("total_interactions")
    unique_activity_types = grp["activity_category"].nunique().rename("unique_activity_types")
    pre_course = (
        clean_logs[clean_logs["date"] < 0]
        .groupby("id_student")
        .size()
        .rename("pre_course_interactions")
    )

    category_counts = (
        clean_logs.groupby(["id_student", "activity_category"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={
            "Quiz":          "quiz_interactions",
            "StudyMaterial": "study_interactions",
            "Discussion":    "discussion_interactions",
            "Navigation":    "navigation_interactions",
            "External":      "external_interactions",
            "DataTool":      "datatool_interactions",
        })
    )

    # Keep only columns that exist
    wanted = [
        "quiz_interactions",
        "study_interactions",
        "discussion_interactions",
        "navigation_interactions",
        "external_interactions",
        "datatool_interactions",
    ]
    category_counts = category_counts[
        [c for c in wanted if c in category_counts.columns]
    ]

    static = pd.concat(
        [total_clicks, total_interactions, unique_activity_types, pre_course],
        axis=1,
    ).fillna(0)

    static = static.join(category_counts, how="left").fillna(0)
    static = static.reset_index()

    return static


# ─── COMBINE ──────────────────────────────────────────────────────────────────
def build_feature_matrix(
    student_sequences: pd.DataFrame,
    selected_patterns: pd.DataFrame,
    clean_logs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Builds the full feature matrix by combining:
        - Pattern features  (binary subsequence flags)
        - Static features   (aggregate click / interaction counts)
        - Label             (performance_group: High / Low)

    Returns one row per student.
    """
    print("Building pattern features...")
    pattern_features = build_pattern_features(student_sequences, selected_patterns)
    print(f"  Shape: {pattern_features.shape}")

    print("Building static features...")
    static_features = build_static_features(clean_logs)
    print(f"  Shape: {static_features.shape}")

    # Labels
    labels = student_sequences[["id_student", "performance_group"]]

    # Merge everything on id_student
    features = (
        pattern_features
        .merge(static_features, on="id_student", how="left")
        .merge(labels,          on="id_student", how="left")
    )

    # Fill any remaining nulls
    features = features.fillna(0)

    print(f"\nFinal feature matrix shape: {features.shape}")
    return features


# ─── FULL PIPELINE ────────────────────────────────────────────────────────────
def run_feature_engineering(
    processed_path: str,
    results_path: str,
) -> pd.DataFrame:
    """
    End-to-end pipeline:
        1. Load student_sequences.csv
        2. Load selected_patterns.csv
        3. Load clean_logs.csv
        4. Build feature matrix
        5. Save to data/processed/features.csv
    """
    student_sequences  = pd.read_csv(processed_path + "student_sequences.csv")
    selected_patterns  = pd.read_csv(results_path   + "selected_patterns.csv")
    clean_logs         = pd.read_csv(processed_path + "clean_logs.csv")

    features = build_feature_matrix(student_sequences, selected_patterns, clean_logs)

    output_path = processed_path + "features.csv"
    features.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    return features

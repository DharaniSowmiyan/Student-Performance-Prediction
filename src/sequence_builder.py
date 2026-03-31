import pandas as pd


def load_clean_logs(processed_path: str) -> pd.DataFrame:
    logs = pd.read_csv(processed_path + "clean_logs.csv")
    return logs


def build_sequences(logs: pd.DataFrame) -> pd.DataFrame:
    # Step 1 — sort chronologically (negative dates first = pre-course)
    logs = logs.sort_values(["id_student", "date"]).reset_index(drop=True)

    # Step 2 — group activity_category into ordered list per student
    sequences = (
        logs.groupby("id_student")["activity_category"]
        .apply(list)
        .reset_index()
        .rename(columns={"activity_category": "sequence_list"})
    )

    # Step 3 — extract performance label (same value for all rows of a student)
    labels = (
        logs.groupby("id_student")["final_result"]
        .first()
        .reset_index()
        .rename(columns={"final_result": "performance_label"})
    )

    # Step 4 — merge sequences with labels
    student_sequences = sequences.merge(labels, on="id_student")

    # Step 5 — convert list → comma-separated string for CSV storage
    student_sequences["sequence"] = student_sequences["sequence_list"].apply(
        lambda x: ",".join(x)
    )

    # Step 6 — drop the intermediate list column, reorder
    student_sequences = student_sequences[
        ["id_student", "sequence", "performance_label"]
    ]

    return student_sequences


def get_sequence_as_list(sequence_str: str) -> list:
    return sequence_str.split(",")


def get_sequence_lengths(student_sequences: pd.DataFrame) -> pd.Series:
    return student_sequences["sequence"].apply(lambda x: len(x.split(",")))


def apply_performance_grouping(student_sequences: pd.DataFrame) -> pd.DataFrame:
    """
    Maps binary performance_label to performance_group.

    Mapping:
        Pass → High
        Fail → Low
    """
    mapping = {
        "Pass": "High",
        "Fail": "Low",
    }
    student_sequences = student_sequences.copy()
    student_sequences["performance_group"] = (
        student_sequences["performance_label"]
        .map(mapping)
        .fillna("Unknown")
    )
    return student_sequences


def run_sequence_builder(processed_path: str) -> pd.DataFrame:
    logs = load_clean_logs(processed_path)
    student_sequences = build_sequences(logs)
    return student_sequences
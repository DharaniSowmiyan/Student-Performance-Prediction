import pandas as pd
def load_clean_logs(processed_path: str) -> pd.DataFrame:
    logs = pd.read_csv(processed_path + "clean_logs.csv")
    return logs

def build_sequences(logs: pd.DataFrame) -> pd.DataFrame:
    logs = logs.sort_values(["id_student", "date"]).reset_index(drop=True)

    sequences = (
        logs.groupby("id_student")["activity_category"]
        .apply(list)
        .reset_index()
        .rename(columns={"activity_category": "sequence_list"})
    )

    labels = (
        logs.groupby("id_student")["final_result"]
        .first()
        .reset_index()
        .rename(columns={"final_result": "performance_label"})
    )

    student_sequences = sequences.merge(labels, on="id_student")

    student_sequences["sequence"] = student_sequences["sequence_list"].apply(
        lambda x: ",".join(x)
    )

    student_sequences = student_sequences[
        ["id_student", "sequence", "performance_label"]
    ]

    return student_sequences


def get_sequence_as_list(sequence_str: str) -> list:
    return sequence_str.split(",")


def get_sequence_lengths(student_sequences: pd.DataFrame) -> pd.Series:
    return student_sequences["sequence"].apply(lambda x: len(x.split(",")))


def apply_performance_grouping(student_sequences: pd.DataFrame) -> pd.DataFrame:
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
import pandas as pd

ACTIVITY_CATEGORY_MAP = {
    "oucontent":      "StudyMaterial",
    "resource":       "StudyMaterial",
    "page":           "StudyMaterial",
    "subpage":        "StudyMaterial",
    "sharedsubpage":  "StudyMaterial",
    "htmlactivity":   "StudyMaterial",
    "folder":         "StudyMaterial",

    "quiz":           "Quiz",
    "questionnaire":  "Quiz",
    "externalquiz":   "Quiz",
    "repeatactivity": "Quiz",

    "forumng":        "Discussion",
    "oucollaborate":  "Discussion",
    "ouwiki":         "Discussion",
    "ouelluminate":   "Discussion",
    "glossary":       "Discussion",

    "url":            "External",

    "homepage":       "Navigation",
    "dualpane":       "Navigation",

    "dataplus":       "DataTool",
}


def load_data(data_path):
    student_vle = pd.read_csv(data_path + "studentVle.csv")
    student_info = pd.read_csv(data_path + "studentInfo.csv")
    vle = pd.read_csv(data_path + "vle.csv")
    return student_vle, student_info, vle


def merge_datasets(student_vle, student_info, vle):
    logs = student_vle.merge(vle, on="id_site", how="left")
    logs = logs.merge(
        student_info[["id_student", "final_result"]],
        on="id_student",
        how="left"
    )
    return logs


def select_columns(logs):
    return logs[[
        "id_student",
        "date",
        "activity_type",
        "sum_click",
        "final_result"
    ]]


def remove_withdrawn(logs):
    return logs[logs["final_result"] != "Withdrawn"]


def normalize_labels(logs):
    logs = logs.copy()
    logs["final_result"] = logs["final_result"].replace({"Distinction": "Pass"})
    return logs


def clean_activity_type(logs):
    logs = logs.copy()
    logs["activity_type"] = logs["activity_type"].fillna("unknown")
    return logs


def apply_activity_category(logs):   
    logs = logs.copy()
    logs["activity_category"] = (
        logs["activity_type"]
        .map(ACTIVITY_CATEGORY_MAP)
        .fillna("Unknown")
    )
    return logs


def sort_logs(logs):
    return logs.sort_values(["id_student", "date"]).reset_index(drop=True)


def run_preprocessing(data_path):
    student_vle, student_info, vle = load_data(data_path)

    logs = merge_datasets(student_vle, student_info, vle)
    logs = select_columns(logs)
    logs = remove_withdrawn(logs)
    logs = normalize_labels(logs)
    logs = clean_activity_type(logs)
    logs = apply_activity_category(logs)   
    logs = sort_logs(logs)

    return logs
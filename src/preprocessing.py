import pandas as pd


# ─── ACTIVITY CATEGORY MAPPING ────────────────────────────────────────────────
# Maps all 20 raw VLE activity_type values into 6 clean educational categories.
# Based on actual unique values found in vle.csv.

ACTIVITY_CATEGORY_MAP = {
    # StudyMaterial — core course content students read/watch
    "oucontent":      "StudyMaterial",
    "resource":       "StudyMaterial",
    "page":           "StudyMaterial",
    "subpage":        "StudyMaterial",
    "sharedsubpage":  "StudyMaterial",
    "htmlactivity":   "StudyMaterial",
    "folder":         "StudyMaterial",

    # Quiz — any form of self-assessment or graded test
    "quiz":           "Quiz",
    "questionnaire":  "Quiz",
    "externalquiz":   "Quiz",
    "repeatactivity": "Quiz",

    # Discussion — collaborative and communication activities
    "forumng":        "Discussion",
    "oucollaborate":  "Discussion",
    "ouwiki":         "Discussion",
    "ouelluminate":   "Discussion",
    "glossary":       "Discussion",

    # External — links outside the VLE platform
    "url":            "External",

    # Navigation — structural pages that help students orient
    "homepage":       "Navigation",
    "dualpane":       "Navigation",

    # DataTool — interactive data exploration tools
    "dataplus":       "DataTool",
}


# ─── LOAD ─────────────────────────────────────────────────────────────────────
def load_data(data_path):
    student_vle = pd.read_csv(data_path + "studentVle.csv")
    student_info = pd.read_csv(data_path + "studentInfo.csv")
    vle = pd.read_csv(data_path + "vle.csv")
    return student_vle, student_info, vle


# ─── MERGE ────────────────────────────────────────────────────────────────────
def merge_datasets(student_vle, student_info, vle):
    logs = student_vle.merge(vle, on="id_site", how="left")
    logs = logs.merge(
        student_info[["id_student", "final_result"]],
        on="id_student",
        how="left"
    )
    return logs


# ─── SELECT COLUMNS ───────────────────────────────────────────────────────────
def select_columns(logs):
    return logs[[
        "id_student",
        "date",
        "activity_type",
        "sum_click",
        "final_result"
    ]]


# ─── REMOVE WITHDRAWN ─────────────────────────────────────────────────────────
def remove_withdrawn(logs):
    return logs[logs["final_result"] != "Withdrawn"]


# ─── NORMALIZE LABELS ─────────────────────────────────────────────────────────
def normalize_labels(logs):
    logs = logs.copy()
    logs["final_result"] = logs["final_result"].replace({"Distinction": "Pass"})
    return logs


# ─── CLEAN ACTIVITY TYPE ──────────────────────────────────────────────────────
def clean_activity_type(logs):
    logs = logs.copy()
    logs["activity_type"] = logs["activity_type"].fillna("unknown")
    return logs


# ─── ACTIVITY CATEGORIZATION (Day 3) ─────────────────────────────────────────
def apply_activity_category(logs):
    """
    Maps raw activity_type values to clean educational categories.

    Categories:
        StudyMaterial  — course content (oucontent, resource, page, subpage…)
        Quiz           — assessments (quiz, questionnaire, externalquiz…)
        Discussion     — collaboration (forumng, ouwiki, oucollaborate…)
        External       — outside links (url)
        Navigation     — structural pages (homepage, dualpane)
        DataTool       — data exploration tools (dataplus)
        Unknown        — any unmapped or null activity type
    """
    logs = logs.copy()
    logs["activity_category"] = (
        logs["activity_type"]
        .map(ACTIVITY_CATEGORY_MAP)
        .fillna("Unknown")
    )
    return logs


# ─── SORT ─────────────────────────────────────────────────────────────────────
def sort_logs(logs):
    return logs.sort_values(["id_student", "date"]).reset_index(drop=True)


# ─── FULL PIPELINE ────────────────────────────────────────────────────────────
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
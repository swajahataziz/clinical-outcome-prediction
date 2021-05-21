import os
from typing import List
import sys

import argparse
import warnings
import csv

sys.path.append('.')
import pandas as pd


def pro_groups_3_digits_mimic(seed: int, admission_only: bool):
    """
    Extracts information needed for the task from the MIMIC dataset. Namely "TEXT" column from NOTEEVENTS.csv and
    "ICD9_CODE" from PROCEDURES_ICD.csv. Divide all ICD9 codes' first two digits and group them per admission into
    column "SHORT_CODES".
    Creates 70/10/20 split over patients for train/val/test sets.
    """
    input_path = '/opt/ml/processing/input'
    output_path = '/opt/ml/processing/output'

    # set task name
    task_name = "PRO_GROUPS_3_DIGITS"

    if admission_only:
        task_name = f"{task_name}_adm"

    # load dataframes
    mimic_procedures = pd.read_csv(
        os.path.join(input_path, "PROCEDURES_ICD.csv"))
    mimic_notes = pd.read_csv(os.path.join(input_path, "NOTEEVENTS.csv"))
    mimic_admissions = pd.read_csv(os.path.join(input_path, "ADMISSIONS.csv"))

    # filter notes
    mimic_notes = filter_notes(
        mimic_notes, mimic_admissions, admission_text_only=admission_only)

    # only keep relevant columns
    mimic_procedures = mimic_procedures[['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE']]

    # drop all rows without procedure codes
    mimic_procedures = mimic_procedures.dropna(
        how='any', subset=['ICD9_CODE'], axis=0)

    # convert data type of ICD9_CODE from integer to string
    mimic_procedures.ICD9_CODE = mimic_procedures.ICD9_CODE.astype(str)

    # create column SHORT_CODE including first 2 digits of ICD9 code
    mimic_procedures["SHORT_CODE"] = mimic_procedures.ICD9_CODE.astype(
        str).str[:3]

    icd9_codes = mimic_procedures.SHORT_CODE.unique().tolist()

    # remove duplicated code groups per admission
    mimic_procedures = mimic_procedures.drop_duplicates(
        ["HADM_ID", "SHORT_CODE"])

    grouped_codes = mimic_procedures.groupby(['HADM_ID', 'SUBJECT_ID'])['SHORT_CODE'].apply(
        lambda d: ",".join(d.astype(str))).reset_index()

    # rename column
    grouped_codes = grouped_codes.rename(columns={'SHORT_CODE': 'SHORT_CODES'})

    # merge discharge summaries into procedures table
    notes_procedures_df = pd.merge(
        grouped_codes[['HADM_ID', 'SHORT_CODES']], mimic_notes, how='inner', on='HADM_ID')

    save_mimic_split_patient_wise(notes_procedures_df,
                                    label_column='SHORT_CODES',
                                    input_dir=input_path,
                                    output_dir=output_path,
                                    task_name=task_name,
                                    seed=seed)

    # save file with all occuring codes
    write_icd_codes_to_file(icd9_codes, output_path+'/labels')


def write_icd_codes_to_file(icd_codes: List[str], data_path):
    # Create path to task data
    os.makedirs(data_path, exist_ok=True)
    # save ICD codes in an extra file
    with open(os.path.join(data_path, "ALL_3_DIGIT_PRO_CODES.txt"), "w", encoding="utf-8") as icd_file:
        icd_file.write(" ".join(icd_codes))


def filter_notes(notes_df: pd.DataFrame, admissions_df: pd.DataFrame, admission_text_only=False) -> pd.DataFrame:
    """
    Keep only Discharge Summaries and filter out Newborn admissions. Replace duplicates and join reports with
    their addendums. If admission_text_only is True, filter all sections that are not known at admission time.
    """
    # filter out newborns
    adm_grownups = admissions_df[admissions_df.ADMISSION_TYPE != "NEWBORN"]
    notes_df = notes_df[notes_df.HADM_ID.isin(adm_grownups.HADM_ID)]

    # remove notes with no TEXT or HADM_ID
    notes_df = notes_df.dropna(subset=["TEXT", "HADM_ID"])

    # filter discharge summaries
    notes_df = notes_df[notes_df.CATEGORY == "Discharge summary"]

    # remove duplicates and keep the later ones
    notes_df = notes_df.sort_values(by=["CHARTDATE"])
    notes_df = notes_df.drop_duplicates(subset=["TEXT"], keep="last")

    # combine text of same admissions (those are usually addendums)
    combined_adm_texts = notes_df.groupby('HADM_ID')['TEXT'].apply(lambda x: '\n\n'.join(x)).reset_index()
    notes_df = notes_df[notes_df.DESCRIPTION == "Report"]
    notes_df = notes_df[["HADM_ID", "ROW_ID", "SUBJECT_ID", "CHARTDATE"]]
    notes_df = notes_df.drop_duplicates(subset=["HADM_ID"], keep="last")
    notes_df = pd.merge(combined_adm_texts, notes_df, on="HADM_ID", how="inner")

    # strip texts from leading and trailing and white spaces
    notes_df["TEXT"] = notes_df["TEXT"].str.strip()

    # remove entries without admission id, subject id or text
    notes_df = notes_df.dropna(subset=["HADM_ID", "SUBJECT_ID", "TEXT"])

    if admission_text_only:
        # reduce text to admission-only text
        notes_df = filter_admission_text(notes_df)

    return notes_df

def save_mimic_split_patient_wise(df, label_column, input_dir, output_dir, task_name, seed, column_list=None):
    """
    Splits a MIMIC dataframe into 70/10/20 train, val, test with no patient occuring in more than one set.
    Uses ROW_ID as ID column and save to save_path.
    """
    if column_list is None:
        column_list = ["ID", "TEXT", label_column]

    # Load prebuilt MIMIC patient splits
    data_split = {"train": pd.read_csv(os.path.join(input_dir,"mimic_train.csv")),
                  "val": pd.read_csv(os.path.join(input_dir,"mimic_val.csv")),
                  "test": pd.read_csv(os.path.join(input_dir,"mimic_test.csv"))}

    # Use row id as general id and cast to int
    df = df.rename(columns={'HADM_ID': 'ID'})
    df.ID = df.ID.astype(int)

    # Save splits to data folder
    for split_name in ["train", "val", "test"]:
        split_set = df[df.SUBJECT_ID.isin(data_split[split_name].SUBJECT_ID)].sample(frac=1,
                                                                                     random_state=seed)[column_list]

        # lower case column names
        split_set.columns = map(str.lower, split_set.columns)
        
        # Create path to task data
        os.makedirs(output_dir+'/'+split_name, exist_ok=True)

        split_set.to_csv(os.path.join(output_dir, split_name, "{}_{}.csv".format(task_name, split_name)),
                         index=False,
                         quoting=csv.QUOTE_ALL)


def filter_admission_text(notes_df) -> pd.DataFrame:
    """
    Filter text information by section and only keep sections that are known on admission time.
    """
    admission_sections = {
        "CHIEF_COMPLAINT": "chief complaint:",
        "PRESENT_ILLNESS": "present illness:",
        "MEDICAL_HISTORY": "medical history:",
        "MEDICATION_ADM": "medications on admission:",
        "ALLERGIES": "allergies:",
        "PHYSICAL_EXAM": "physical exam:",
        "FAMILY_HISTORY": "family history:",
        "SOCIAL_HISTORY": "social history:"
    }

    # replace linebreak indicators
    notes_df['TEXT'] = notes_df['TEXT'].str.replace(r"\n", r"\\n")

    # extract each section by regex
    for key in admission_sections.keys():
        section = admission_sections[key]
        notes_df[key] = notes_df.TEXT.str.extract(r'(?i){}(.+?)\\n\\n[^(\\|\d|\.)]+?:'
                                                  .format(section))

        notes_df[key] = notes_df[key].str.replace(r'\\n', r' ')
        notes_df[key] = notes_df[key].str.strip()
        notes_df[key] = notes_df[key].fillna("")
        notes_df[notes_df[key].str.startswith("[]")][key] = ""

    # filter notes with missing main information
    notes_df = notes_df[(notes_df.CHIEF_COMPLAINT != "") | (notes_df.PRESENT_ILLNESS != "") |
                        (notes_df.MEDICAL_HISTORY != "")]

    # add section headers and combine into TEXT_ADMISSION
    notes_df = notes_df.assign(TEXT="CHIEF COMPLAINT: " + notes_df.CHIEF_COMPLAINT.astype(str)
                                    + '\n\n' +
                                    "PRESENT ILLNESS: " + notes_df.PRESENT_ILLNESS.astype(str)
                                    + '\n\n' +
                                    "MEDICAL HISTORY: " + notes_df.MEDICAL_HISTORY.astype(str)
                                    + '\n\n' +
                                    "MEDICATION ON ADMISSION: " + notes_df.MEDICATION_ADM.astype(str)
                                    + '\n\n' +
                                    "ALLERGIES: " + notes_df.ALLERGIES.astype(str)
                                    + '\n\n' +
                                    "PHYSICAL EXAM: " + notes_df.PHYSICAL_EXAM.astype(str)
                                    + '\n\n' +
                                    "FAMILY HISTORY: " + notes_df.FAMILY_HISTORY.astype(str)
                                    + '\n\n' +
                                    "SOCIAL HISTORY: " + notes_df.SOCIAL_HISTORY.astype(str))

    return notes_df

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't', '1'):
        return True
    elif v.lower() in ('false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--admission-only', type=str2bool, nargs='?', default=True)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    pro_groups_3_digits_mimic(args.seed, args.admission_only)

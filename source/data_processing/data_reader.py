import json
import pandas as pd
import os
import numpy as np
from datetime import timedelta
from decouple import config
from pathlib import Path
from source.data_processing.data_model import CurrentDate, Dataset, Trajectory, Admission, ClinicalText, Diagnosis, LabResults, PharmacoGenetics, \
    Medication, Patient, Employee, Appointment
from source.data_processing.medication_grouping import set_IS_ANTIDEPRESSANT
from dataclasses import dataclass

from source.data_processing.pharmaco_mapping import map_pgx_phenotype

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Set data path from environment variable
data_path = Path(config("DATA_PATH"))
model_path = Path(config("MODEL_PATH"))
results_path = Path(config("RESULTS_PATH"))

def normalize_hhmm_to_hhmmss(series: pd.Series) -> pd.Series:
    s = series.astype("string")

    # If value is exactly hh:mm, append :00
    hhmm_mask = s.str.match(r"^\d{1,2}:\d{2}$", na=False)
    s = s.mask(hhmm_mask, s + ":00")

    return s

def process_dates(data: pd.DataFrame, dataset: Dataset) -> pd.DataFrame:
    """Convert date columns to datetime and compute durations or age.
    Args:
        data (pd.DataFrame): Input DataFrame.
        dataset (str): Type of dataset.
    Returns:
        pd.DataFrame: DataFrame with processed date columns and computed durations/age.
    """
    # Map datasets to their date columns
    date_columns_map = {
        Dataset.TRAJECTORY: [Trajectory.START.value, Trajectory.STOP.value],
        Dataset.ADMISSION: [Admission.START.value, Admission.STOP.value],
        Dataset.TEXT: [ClinicalText.DATE_LAST.value, ClinicalText.DATE.value],
        Dataset.DIAGNOSIS: [Diagnosis.DATE.value],
        Dataset.LAB: [LabResults.DATE.value],
        Dataset.PHARMACOGENETICS : [PharmacoGenetics.DATE.value],
        Dataset.MEDICATION: [Medication.START.value, Medication.STOP.value],
        Dataset.PATIENT: [Patient.BIRTH_DATE.value],
        Dataset.EMPLOYEE: [Employee.START.value],
        Dataset.APPOINTMENT: [Appointment.DATE.value],
        Dataset.CURRENT_DATE: [CurrentDate.CURRENT_TIME.value],
    }

    # Get the relevant date columns
    dates = date_columns_map.get(dataset, [])
    # Convert to datetime; invalid parsing becomes NaT thanks to coerce param.
    if dates:
        # Convert and localize date columns
        for date in dates:
            data[date] = pd.to_datetime(data[date], errors="coerce")
            if data[date].dt.tz is not None:
                data[date] = data[date].dt.tz_localize(None)
    else:
        print("Dates are empty.")

    # Compute durations or age
    if dataset == Dataset.TRAJECTORY:
        # if trajectory was not ended yet, set the stop date as release date.
        data.loc[
            ~data[Trajectory.IS_CLOSED.value] & data[
                Trajectory.STOP.value].isna(), Trajectory.STOP.value] = release_date
        data[Trajectory.DURATION.value] = (data[Trajectory.STOP.value] - data[Trajectory.START.value]).dt.days

    elif dataset == Dataset.ADMISSION:
        data.loc[
            ~data[Admission.IS_CLOSED.value] & data[
                Admission.STOP.value].isna(), Admission.STOP.value] = release_date

        data[Admission.DURATION.value] = (data[Admission.STOP.value] - data[Admission.START.value]).dt.days

    elif dataset == Dataset.MEDICATION:
        data.loc[
            ~data[Medication.IS_STOPPED.value] & data[
                Medication.STOP.value].isna(), Medication.STOP.value] = release_date
        data[Medication.DURATION.value] = (data[Medication.STOP.value] - data[Medication.START.value]).dt.days

    elif dataset == Dataset.APPOINTMENT:
        # Convert start and stop times to timedelta
        start_col = normalize_hhmm_to_hhmmss(data[Appointment.START.value])
        stop_col = normalize_hhmm_to_hhmmss(data[Appointment.STOP.value])
        start_times = pd.to_timedelta(start_col)
        stop_times = pd.to_timedelta(stop_col)

        # Compute duration
        duration = stop_times - start_times
        # Add 24h for negative durations (crossed midnight)
        duration = duration + pd.to_timedelta("1 days") * (duration < pd.Timedelta(0))
        # duration in hours.
        data[Appointment.DURATION.value] = duration.dt.total_seconds() / 3600

    return data


def merge_employee_asof(data, employee_data, id_col, role_col, start_col, date_col):
    """
    Merges employee data into data using asof merge (latest employee record <= note date).

    Args:
        id_col:        ClinicalText column for employee ID (join key)
        role_col:      ClinicalText column for employee role
        start_col:     ClinicalText column for employee start date
        date_col: ClinicalText column for the note/mutation date
    """
    employee_renamed = employee_data.rename(columns={
        Employee.ID.value: id_col,
        Employee.ROLE.value: role_col,
        Employee.START.value: start_col,
    })

    # Drop NA
    employee_renamed = employee_renamed.dropna(subset=[id_col, start_col])
    data = data.dropna(subset=[date_col])

    # Sort values
    employee_renamed = employee_renamed.sort_values([start_col])
    data = data.sort_values([date_col])


    return pd.merge_asof(
        data,
        employee_renamed[[id_col, role_col, start_col]],
        left_on=date_col,
        right_on=start_col,
        by=id_col,
        direction="backward"
    )



def extract_note_type(text: str) -> str:
    """
    Extracts the note type from the first line of a Dutch clinical note.
    E.g. "Anamnese\n<PERSOON-1> ingelicht..." -> "Anamnese"
    """
    if not isinstance(text, str) or not text.strip():
        return "UNKNOWN"
    return text.strip().split("\n")[0].strip()


def preprocess(data: pd.DataFrame, dataset: Dataset, merge_df, employee_data=None) -> pd.DataFrame:
    """

    Args:
        data (pd.DataFrame): Input DataFrame.
        dataset (Dataset class): Type of dataset.

    Returns:
        pd.DataFrame: 
    """

    if dataset == Dataset.TEXT:
        if not ClinicalText.NOTE_ID.value in data:
            data[ClinicalText.NOTE_ID.value] = range(len(data))

        # Replace missing NaN text with "".
        data[ClinicalText.TEXT.value] = (
            data[ClinicalText.TEXT.value]
            .fillna("")
            .astype("string")
        )
        # Extract note type from first line of text
        data[ClinicalText.NOTE_TYPE.value] = (
            data[ClinicalText.TEXT.value]
            .apply(extract_note_type)
            .astype("string")
        )
        # Compute word count (excluding the note type first line)
        data[ClinicalText.TOKEN_NUMBER.value] = (
            data[ClinicalText.TEXT.value]
                .str.split()
                .str.len()
                .fillna(0)
                .astype("int32")
        )
        # ------------------- Merge employee info if provided -------------------
        if employee_data is not None:
            # Ensure start date is datetime
            employee_data[Employee.START.value] = pd.to_datetime(employee_data[Employee.START.value])

            # Merge creation employee info
            data = merge_employee_asof(
                data, employee_data,
                id_col=ClinicalText.CREATION_PRACTITIONER_CODE.value,
                role_col=ClinicalText.CREATION_EMPLOYEE_ROLE.value,
                start_col=ClinicalText.CREATION_EMPLOYEE_START.value,
                date_col=ClinicalText.DATE.value
            )
            # Merge mutation employee info (optional)
            data = merge_employee_asof(
                data, employee_data,
                id_col=ClinicalText.MUTATION_PRACTITIONER_CODE.value,
                role_col=ClinicalText.MUTATION_EMPLOYEE_ROLE.value,
                start_col=ClinicalText.MUTATION_EMPLOYEE_START.value,
                date_col=ClinicalText.DATE_LAST.value
            )
            # discard the notes written by nurse..
            data = data[
                data[ClinicalText.CREATION_EMPLOYEE_ROLE.value] != 'nurse']

        # ------------------- Merge clinical notes with trajectory -------------------

        data = data.merge(
            merge_df,
            how='left',
            on=Patient.ID.value,
        ).query(
            f"{ClinicalText.DATE.value} >= {Trajectory.START.value} and {ClinicalText.DATE.value} <= {Trajectory.STOP.value}"
        )[[item.value for item in ClinicalText]]
        # We remove duplicated clinical text entries if date and text values are the same.
        data = data.drop_duplicates(subset=[ClinicalText.DATE.value, ClinicalText.TEXT.value])
        # We remove rows where 'text' column is empty or NaN, and the number of words less than 5.
        data = data[data[ClinicalText.TEXT.value].notna() & (data[ClinicalText.TEXT.value].str.strip() != '') &
                    (data[ClinicalText.TEXT.value].str.split().str.len() >= 5)]



    elif dataset == Dataset.TRAJECTORY:
        # calculate the age of the patient based on the start of each of their trajectories.
        data = data.merge(
            merge_df,
            how='left',
            on=Trajectory.PATIENT_ID.value
        )
        # calculate age at trajectory start
        data[Trajectory.PATIENT_AGE_AT_START.value] = data[Trajectory.START.value].dt.year - data[
            Patient.BIRTH_DATE.value].dt.year
        # drop the birthdate column after calculating age, but keep patient ID
        data = data.drop(columns=[item.value for item in Patient if not item.value == Patient.ID.value])

    elif dataset == Dataset.DIAGNOSIS:
        # merge diagnosis description with diagnosis data
        diagnoses_desc = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "diagnosis_desc.csv"), encoding='windows-1252')
        data = data.merge(diagnoses_desc, how='left', on=Diagnosis.DSM5_CODE.value)

        # logic: all diagnoses that are made before or during trajectory are relevant for a given trajectory.
        data = (data.merge(
            merge_df,
            how='left',
            on=Patient.ID.value,
        )
                # Keep only diagnoses before trajectory stop date
                .query(f"{Diagnosis.DATE.value} <= {Trajectory.STOP.value}")
                )
        # in case DSM5 code is empty or Nan, we use available DSM4 code.
        data[Diagnosis.DSM_CODE.value] = data[Diagnosis.DSM5_CODE.value].fillna(data[Diagnosis.DSM4_CODE.value])
        data = data[[item.value for item in Diagnosis]]

    elif dataset == Dataset.APPOINTMENT:
        # ------------------- Merge employee info if provided -------------------
        if employee_data is not None:
            # Ensure start date is datetime
            employee_data[Employee.START.value] = pd.to_datetime(employee_data[Employee.START.value])

            # Merge creation employee info
            data = merge_employee_asof(
                data, employee_data,
                id_col=Appointment.EMPLOYEE_ID.value,
                role_col=Appointment.EMPLOYEE_ROLE.value,
                start_col=Appointment.EMPLOYEE_START.value,
                date_col=Appointment.DATE.value
            )
        data = data.merge(merge_df, how='left', on=Patient.ID.value).query(
            f"{Appointment.DATE.value} >= {Trajectory.START.value} and {Appointment.DATE.value} <= {Trajectory.STOP.value}"
        )[[item.value for item in Appointment]]

    elif dataset == Dataset.LAB:
        data = data.merge(
            merge_df,
            how='left',
            on=Patient.ID.value,
        ).query(
            f"{LabResults.DATE.value} >= {Trajectory.START.value} and {LabResults.DATE.value} <= {Trajectory.STOP.value}"
        )[[item.value for item in LabResults if item != LabResults.IN_RANGE]]

        # --- Derive in_range from reference bounds ---
        val = pd.to_numeric(data[LabResults.VALUE.value], errors="coerce")
        low = pd.to_numeric(data[LabResults.RANGE_LOW.value], errors="coerce")
        high = pd.to_numeric(data[LabResults.RANGE_HIGH.value], errors="coerce")

        conditions = [
            val.isna() | (low.isna() & high.isna()),    # unknown: value or both bounds missing
            val < low,                                   # below range
            val > high,                                  # above range
        ]
        choices = ["unknown", "low", "high"]
        data[LabResults.IN_RANGE.value] = np.select(conditions, choices, default="in_range")
        
    
    elif dataset == Dataset.PHARMACOGENETICS:
      data = data.merge(
         merge_df,
        how='left',
       on=Patient.ID.value,
    ).query(
      f"{PharmacoGenetics.DATE.value} >= {Trajectory.START.value} and {PharmacoGenetics.DATE.value} <= {Trajectory.STOP.value}"
    )[[item.value for item in PharmacoGenetics if item != PharmacoGenetics.PHENOTYPE]]
      # Apply phenotype mapping
      data[PharmacoGenetics.PHENOTYPE.value] = data[PharmacoGenetics.VALUE.value].apply(map_pgx_phenotype)

    elif dataset == Dataset.ADMISSION:
        data = data.merge(
            merge_df,
            how='left',
            on=Patient.ID.value,
        ).query(
            # if patient is admitted to the hospital within the trajectory, include them.
            f"{Admission.START.value} >= {Trajectory.START.value} and {Admission.START.value} <= {Trajectory.STOP.value}"
        )
        # logic: if admission is not stopped yet, set its end date as trajectory end date.
        data.loc[~data[Admission.IS_CLOSED.value], Admission.STOP.value] = data.loc[
            ~data[Admission.IS_CLOSED.value], Trajectory.STOP.value]
        data = data[[item.value for item in Admission]]

    elif dataset == Dataset.MEDICATION:
        # default dose unit is mg..
        data[Medication.DOSE_UNIT.value] = "mg" if not Medication.DOSE_UNIT.value in data else data[
            Medication.DOSE_UNIT.value]
        # default type is other:
        data[Medication.ANTIDEPRESSANT_TYPE.value] = "other"

        # Count how many were NaN
        num_missing = data[Medication.ATC_CODE.value].isna().sum()
        print(f"Number of missing ATC codes in medications data: {num_missing}")
        data[Medication.ATC_CODE.value] = data[Medication.ATC_CODE.value].fillna('')
        data[Medication.DAY_DOSES.value] = data[Medication.DAY_DOSES.value].fillna(0)

        # TODO: if medication started earlier and then stopped after one week; treatment trajectory has begin medication verification by pharmacy

        # merge medication if its start date fall into trajectory, or it was started earlier, but stopped within or later than trajectory end date.
        data = data.merge(
            merge_df,
            how='left',
            on=Patient.ID.value,
        ).query(
            f"({Medication.START.value} >= {Trajectory.START.value} and {Medication.START.value} <= {Trajectory.STOP.value}) "
            f"or "
            f"{Medication.START.value} <= {Trajectory.START.value} and {Medication.STOP.value} >= {Trajectory.START.value}"
        )

        # cleaning: we remove duplicated medication entries if start/end and name values are the same.
        data = data.drop_duplicates(subset=[Medication.START.value, Medication.STOP.value, Medication.NAME.value])
        # convert g into mg..
        data = correct_dosage_unit(data)
        # merge same medications if their stopped period is less than 14 days,
        # as this is potentially a noise in the data, where the registration was delayed.
        data = process_medications(data)
        
        # Processing: set as main, modifier or other, drug class and base atc code is also set here. 
        data = set_IS_ANTIDEPRESSANT(data) if not data.empty else data
        data = data[[item.value for item in Medication]]
      

    return data


def convert_mg_unit(value, unit):
    # if unit is mcg or g, convert to mg for consistency.
    if unit == 'mcg':
        return value * 0.001, "mg"
    elif unit == 'g':
        return value * 1000, "mg"
    return value, unit


def correct_dosage_unit(data):
    data_after = data.copy(deep=True)
    # Apply to dataframe
    data_after[[Medication.DAY_DOSES.value, Medication.DOSE_UNIT.value]] = (
        data_after.apply(
            lambda row: convert_mg_unit(row[Medication.DAY_DOSES.value], row[Medication.DOSE_UNIT.value]),
            axis=1,
            result_type="expand"
        )
    )
    return data_after


# merge medications with same ATC code if stopping period is less than 2 weeks.
def process_medications(data):
    """
    Process and merge consecutive identical medication prescriptions for each patient trajectory.

    Merges consecutive prescriptions of the same medication if the gap
       between them is 0–14 days.
       - When merging, the stop date is extended to the latest end date.
       - The latest daily dosage is retained for consecutive prescriptions.
    Recalculates the duration of each medication period after merging.
    Returns a new DataFrame with merged medication periods.

    Parameters:
    ----------
    data : pd.DataFrame
        Input DataFrame containing medication records with columns for
        patient trajectory, medication code, start date, stop date, and daily dose.

    Returns:
    -------
    pd.DataFrame
        DataFrame with merged medication records and updated durations.
    """

    # TODO: check with Cas: if the same medication prescribed again, shall we merge them update the dosage information?
    processed_data = data.copy(deep=True)
    merged_rows = []

    for traj_id, group in processed_data.groupby(Trajectory.ID.value):
        for med_code, med_group in group.groupby(Medication.ATC_CODE.value):
            if len(med_group) <= 1:
                merged_rows.append(med_group.iloc[0])
                continue
            med_group.sort_values(
                by=[Medication.START.value, Medication.STOP.value],
                inplace=True
            )
            merged = []
            current = med_group.iloc[0].copy()
            for i in range(1, len(med_group)):
                next_row = med_group.iloc[i].copy()
                if pd.isnull(current[Medication.STOP.value]):
                    merged.append(current)
                    current = next_row
                    continue
                gap = next_row[Medication.START.value] - current[Medication.STOP.value]
                current_dose = current.get(Medication.DAY_DOSES.value)
                next_dose = next_row.get(Medication.DAY_DOSES.value)
                # if doses matches, or any of them is 0 or NaN, merge them..
                dose_compatible = (
                        current_dose == next_dose
                        or any(
                    d is None or pd.isna(d) or d == 0
                    for d in (current_dose, next_dose)
                ))
                if timedelta(days=0) <= gap <= timedelta(days=14) and dose_compatible:
                    # Merge by extending the end date
                    next_stop = next_row[Medication.STOP.value]
                    if next_stop is not None and next_stop > current[Medication.STOP.value]:
                        current[Medication.STOP.value] = next_stop
                        current[Medication.IS_STOPPED.value] = next_row[Medication.IS_STOPPED.value]
                    # TODO: we use the latest medication dosage in case these are prescribed consecutively (with less than 14 days break).
                    if next_dose not in (None, 0):
                        current[Medication.DAY_DOSES.value] = next_dose
                else:
                    # Not merge-able, add current and move on
                    merged.append(current)
                    current = next_row

            merged.append(current)
            merged_rows.extend(merged)

    merged_data = pd.DataFrame(merged_rows).reset_index(drop=True)
    # update duration information as the end dates might change...
    merged_data[Medication.DURATION.value] = (
            merged_data[Medication.STOP.value] - merged_data[Medication.START.value]).dt.days
    return merged_data


def read_data(dataset: Dataset) -> pd.DataFrame:
    """Reads data from CSV or Parquet file based on the dataset type.

    Args:
        dataset (Dataset): The type of dataset to read.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    filename = os.path.join(data_path, dataset.value)

    if os.path.exists(filename):
        if filename.endswith('.csv'):
            data = pd.read_csv(filename).drop_duplicates()
        elif filename.endswith('.parquet'):
            data = pd.read_parquet(filename).drop_duplicates()
            # data[Patient.ID.value] = data[Patient.ID.value].astype(int)
        else:
            print("File is not in desired format (either csv or parquet)")
            return pd.DataFrame()

        # convert date columns into datetime object and remove timezone information.
        data = process_dates(data, dataset)
        print(f"Data {dataset} is read and it has a shape of {data.shape}")
        return data
    else:
        print(f"{filename} not found...")
        return pd.DataFrame()

def filter_antidepressant_depression_trajectories(trajectories_data, diagnoses, medications):
    """
            we only keep trajectories where there is an active main depression diagnosis in the time of treatment,
            and also at least one antidepressant is registered.
            Parameters:
            trajectories (df): dataframe.
            Returns:
            df: filtered trajectories.
        """
    if trajectories_data.empty:
        print("trajectories empty")
        return trajectories_data
    else:
        # Select medications where IS_ANTIDEPRESSANT is either 'main' or 'modifier'
        antidepressant_meds = medications[
            medications[Medication.ANTIDEPRESSANT_TYPE.value].isin(["main"])
            # at least primary medication is prescribed during trajectory
        ]
        antidepressant_traj_ids = set(
            antidepressant_meds[Trajectory.ID.value].unique()
        )

        # --- 2. Filter trajectories with MAIN diagnosis =  depression ---
        # we expect that there is a depression diagnosis either as a main or secondary diagnosis + at least 1 antidepressant prescribed during trajectory.
        exclude_codes = ["D5_4.02.02.04", "D5_4.02.01.04"]  # DSM5 psychotic depression
        exclude_codes_icd = ["as1_6.01.01.01.04", "as1_6.01.01.02.04"]  # DSM4 psychotic depression

        depression_dx = diagnoses[
            (
                    (diagnoses[Diagnosis.DSM5_CODE.value].str.startswith("D5_4")) &
                    (~diagnoses[Diagnosis.DSM5_CODE.value].isin(exclude_codes))
            )
            |
            (
                    (diagnoses[Diagnosis.DSM4_CODE.value].str.startswith("as1_6.01")) &
                    (~diagnoses[Diagnosis.DSM4_CODE.value].isin(exclude_codes_icd))
            )
            ]

        depression_traj_ids = set(
            depression_dx[Diagnosis.TRAJECTORY_ID.value].unique()
        )

        # --- 3. Require BOTH conditions, at least one antidepressant AND depression diagnosis---
        valid_traj_ids = antidepressant_traj_ids & depression_traj_ids

        filtered_df = trajectories_data[
            trajectories_data[Trajectory.ID.value].isin(valid_traj_ids)
        ]
        print("trajectories satisfy following conditions: "
              "1. The trajectory must contain at least one antidepressant medication (as defined in the primary medication list), "
              "2. The trajectory must contain a depression diagnosis (either as main or secondary) where DSM5 starts with D5_4.",
              "3. Patients any history with psychosis depression are excluded.. ")

        return filtered_df.reset_index(drop=True)


def filter_by_trajectory_id(df, trajectories):
    """
    Filters a dataframe to keep only rows whose Trajectory.ID exists
    in the filtered trajectories dataframe.
    """
    valid_ids = set(trajectories[Trajectory.ID.value])

    if Trajectory.ID.value not in df.columns:
        raise ValueError(f"Dataframe is missing required column: {Trajectory.ID.value}")

    return df[df[Trajectory.ID.value].isin(valid_ids)]


def filter_empty_trajectories(trajectories_data, clinical_texts, medications):
    """
       filter trajectories based on patient ID. we only keep trajectories where
       there is at least a medication or a clinical text information about given patient over time.
       Parameters:
       trajectories (df): dataframe.
       Returns:
       df: filtered trajectories.
    """
    if trajectories_data.empty:
        print("trajectories empty")
        return trajectories_data
    else:
        filtered_df = trajectories_data[trajectories_data[Trajectory.ID.value].isin(
            clinical_texts[Trajectory.ID.value]) |
                                        trajectories_data[
                                            Trajectory.ID.value].isin(
                                            medications[Trajectory.ID.value])]
        print("trajectories filtered")
        return filtered_df.reset_index(drop=True)

def set_release_date():
    current_date = read_data(Dataset.CURRENT_DATE)
    if current_date.empty or CurrentDate.CURRENT_TIME.value not in current_date.columns:
        release_date = pd.to_datetime(config("RELEASE_DATE"), errors="raise")  # e.g. "2025-12-05"
    else:
        release_date = pd.to_datetime(current_date[CurrentDate.CURRENT_TIME.value].iloc[0], errors="raise")

    if getattr(release_date, 'tz', None) is not None:
        release_date = release_date.tz_convert(None)

    return release_date

release_date = set_release_date()
print("Release date is set to:", release_date)


@dataclass
class DatasetBundle:
    patients: any
    trajectories: any
    admissions: any
    clinical_texts: any
    diagnoses: any
    appointments: any
    lab_results: any
    pharmaco_results: any
    medications: any
    employees: any


# Module-level variable, initialized lazily
_DATA_BUNDLE: DatasetBundle | None = None


def load_datasets() -> DatasetBundle:
    global _DATA_BUNDLE
    if _DATA_BUNDLE is None:
        """Loading all data, and applying processing for: patients, trajectories, medications, texts, diagnoses..."""
        # convert date columns into datetime object and remove timezone information.
        patients = read_data(Dataset.PATIENT)
        trajectories = read_data(Dataset.TRAJECTORY)
        if trajectories.empty:
            print("No trajectories found")
        print("Patients and trajectories dataset are loaded, reading other data sources..")
        admissions = read_data(Dataset.ADMISSION)
        clinical_texts = read_data(Dataset.TEXT)
        diagnoses = read_data(Dataset.DIAGNOSIS)
        lab_results = read_data(Dataset.LAB)
        pharmaco_results = read_data(Dataset.PHARMACOGENETICS)
        medications = read_data(Dataset.MEDICATION)
        employees = read_data(Dataset.EMPLOYEE)
        appointments = read_data(Dataset.APPOINTMENT)

        # Preprocess and extend each dataset by aligning with trajectories
        trajectories = preprocess(trajectories, Dataset.TRAJECTORY,
                                  patients) if not trajectories.empty and not patients.empty else trajectories
        medications = preprocess(medications, Dataset.MEDICATION,
                                 trajectories) if not medications.empty and not trajectories.empty else medications
        diagnoses = preprocess(diagnoses, Dataset.DIAGNOSIS,
                               trajectories) if not diagnoses.empty and not trajectories.empty else diagnoses
        clinical_texts = preprocess(clinical_texts, Dataset.TEXT, trajectories,
                                    employee_data=employees) if not clinical_texts.empty and not trajectories.empty else clinical_texts
        lab_results = preprocess(lab_results, Dataset.LAB, trajectories) if not lab_results.empty and not trajectories.empty else lab_results
        pharmaco_results = preprocess(pharmaco_results, Dataset.PHARMACOGENETICS, trajectories) if not pharmaco_results.empty and not trajectories.empty else pharmaco_results

        admissions = preprocess(admissions, Dataset.ADMISSION,
                                trajectories) if not admissions.empty and not trajectories.empty else admissions
        appointments = preprocess(appointments, Dataset.APPOINTMENT, trajectories,
                                    employee_data=employees) if not appointments.empty and not trajectories.empty else appointments

        # Filter trajectories: there should be at least one note or medication linked to trajectory.  
        trajectories = filter_empty_trajectories(trajectories, clinical_texts,
                                                 medications) if not trajectories.empty else trajectories
        trajectories = filter_antidepressant_depression_trajectories(trajectories, diagnoses,
                                                                     medications) if not trajectories.empty else trajectories

        # Filter dependent datasets by trajectory ID
        admissions, medications, clinical_texts, diagnoses, appointments = (
            filter_by_trajectory_id(df, trajectories) if not df.empty and not trajectories.empty else df
            for df in (admissions, medications, clinical_texts, diagnoses, appointments)
        )

        _DATA_BUNDLE = DatasetBundle(
            patients=patients,
            trajectories=trajectories,
            admissions=admissions,
            clinical_texts=clinical_texts,
            diagnoses=diagnoses,
            appointments=appointments,
            lab_results=lab_results,
            pharmaco_results=pharmaco_results,
            medications=medications,
            employees=employees,
        )
    return _DATA_BUNDLE


bundle = load_datasets()
# Access datasets
trajectories = bundle.trajectories
clinical_texts = bundle.clinical_texts
# employees = bundle.employees
medications = bundle.medications
patients = bundle.patients
lab_results = bundle.lab_results
pharmaco_results = bundle.pharmaco_results
admissions = bundle.admissions
diagnoses = bundle.diagnoses
appointments = bundle.appointments
for data, data_name in [[trajectories, 'trajectories'], [clinical_texts, 'text'], [medications, 'medications'],
                        [diagnoses, 'diagnoses'], [appointments, 'appointments']]:
    print(f"Final {data_name} size {data.shape} after processing, will be used for modeling outcomes")

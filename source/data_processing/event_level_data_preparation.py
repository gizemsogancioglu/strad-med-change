import collections
import os

from click import Path
import pandas as pd
from source.data_processing.data_reader import PROJECT_ROOT, data_path, pharmaco_results, lab_results
from source.data_processing.data_model import *
from source.models.medication_change_prediction import add_primary_medication_change_outcomes
import numpy as np
from source.models.model_cfgs.feature_config import CATEGORICAL_COLUMNS

med_count_cols = [
    TemporalEvent.PRIMARY_ADDED.value, TemporalEvent.MODIFIER_ADDED.value, TemporalEvent.OTHER_ADDED.value,
    TemporalEvent.PRIMARY_REMOVED.value, TemporalEvent.MODIFIER_REMOVED.value, TemporalEvent.OTHER_REMOVED.value,
    TemporalEvent.PRIMARY_DOSE_INCREASED.value, TemporalEvent.MODIFIER_DOSE_INCREASED.value,
    TemporalEvent.OTHER_DOSE_INCREASED.value,
    TemporalEvent.PRIMARY_DOSE_DECREASED.value, TemporalEvent.MODIFIER_DOSE_DECREASED.value,
    TemporalEvent.OTHER_DOSE_DECREASED.value,
]


def add_days_since_med_event(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TemporalEvent.DATE.value] = pd.to_datetime(df[TemporalEvent.DATE.value])
    df = df.sort_values([Trajectory.ID.value, TemporalEvent.DATE.value]).reset_index(drop=True)

    med_mask = df[TemporalEvent.TYPE.value].isin([
        EventType.MEDICATION_START.value,
        EventType.MEDICATION_STOP.value
    ])

    result = []
    for traj_id, group in df.groupby(Trajectory.ID.value, sort=False):
        med_dates = group.loc[med_mask.loc[group.index], TemporalEvent.DATE.value].values
        event_dates = group[TemporalEvent.DATE.value].values

        if len(med_dates) == 0:
            days = np.full(len(group), -1, dtype=int)
        else:
            # for each event, find the most recent med date <= event date
            idx = np.searchsorted(med_dates, event_dates, side='right') - 1
            days = np.where(idx >= 0, (event_dates - med_dates[idx]).astype('timedelta64[D]').astype(int), -1)

        result.append(days)

    df[TemporalEvent.DAYS_SINCE_MED_EVENT.value] = np.concatenate(result)

    n_filled = (df[TemporalEvent.DAYS_SINCE_MED_EVENT.value] >= 0).sum()
    n_total = len(df)
    print(f"days_since_med_event added.")
    print(f"  Filled  : {n_filled} / {n_total} ({n_filled / n_total:.1%})")
    print(f"  No prior med event: {n_total - n_filled} rows (set to -1)")

    return df

def add_lab_features(df: pd.DataFrame, lab_results: pd.DataFrame) -> pd.DataFrame:
    """
    Add three new lab-related features to the temporal event dataframe.
    Does NOT recompute has_recent_abnormal_lab — that already exists.

    New columns:
    1. recent_lab_measure (categorical, string):
       Most recent lab result status in past 14 days:
       'high' / 'low' / 'normal' / 'none' (no lab in window)

    2. has_recent_lab_test (boolean, float32):
       1.0 if any lab test was performed in the past 14 days,
       regardless of whether it was abnormal. 0.0 otherwise.

    3. recent_lab_test_type (categorical, string):
       Type of the most recent lab test in past 14 days
       (from LabResults.TEST.value). 'none' if no lab in window.
       Stored as space-separated string of all unique test types
       in the window, compatible with CountVectorizer.

    Parameters
    ----------
    df         : temporal event dataframe
    lab_results: raw lab results dataframe with
                 Trajectory.ID, LabResults.DATE, LabResults.IN_RANGE,
                 LabResults.TEST
    """
    from source.data_processing.data_model import TemporalEvent, Trajectory, LabResults

    WINDOW_DAYS = 14

    lab_measure     = []
    has_recent_lab  = []
    recent_lab_type = []

    # pre-group for speed
    lab_by_traj = {
        traj_id: grp.copy()
        for traj_id, grp in lab_results.groupby(Trajectory.ID.value)
    }

    for _, row in df.iterrows():
        traj_id      = row[Trajectory.ID.value]
        event_dt     = pd.to_datetime(row[TemporalEvent.DATE.value])
        window_start = event_dt - pd.Timedelta(days=WINDOW_DAYS)

        traj_labs = lab_by_traj.get(traj_id, pd.DataFrame())

        if traj_labs.empty:
            lab_measure.append('none')
            has_recent_lab.append(0.0)
            recent_lab_type.append('none')
            continue

        recent = traj_labs[
            (pd.to_datetime(traj_labs[LabResults.DATE.value]) >= window_start) &
            (pd.to_datetime(traj_labs[LabResults.DATE.value]) <  event_dt)
        ].sort_values(LabResults.DATE.value, ascending=False)

        if recent.empty:
            lab_measure.append('none')
            has_recent_lab.append(0.0)
            recent_lab_type.append('none')
            continue

        # any lab test done in window
        has_recent_lab.append(1.0)

        # most recent lab result status (from most recent row)
        most_recent_status = str(recent[LabResults.IN_RANGE.value].iloc[0]).lower()
        if most_recent_status in ('low', 'high', 'normal'):
            lab_measure.append(most_recent_status)
        else:
            lab_measure.append('none')

        # most recent lab test type in window — consistent with recent_lab_measure
        if LabResults.TEST.value in recent.columns:
            most_recent_test = str(recent[LabResults.TEST.value].iloc[0]).strip().lower()
            if most_recent_test and most_recent_test != 'nan':
                recent_lab_type.append(most_recent_test)
            else:
                recent_lab_type.append('none')
        else:
            recent_lab_type.append('none')

    df[TemporalEvent.RECENT_LAB_MEASURE.value]    = pd.array(lab_measure,     dtype='string')
    df[TemporalEvent.HAS_RECENT_LAB_TEST.value]   = pd.array(has_recent_lab,  dtype='float32')
    df[TemporalEvent.RECENT_LAB_TEST_TYPE.value]  = pd.array(recent_lab_type, dtype='string')

    print(f"Lab features added.")
    print(f"  recent_lab_measure:\n{df[TemporalEvent.RECENT_LAB_MEASURE.value].value_counts()}")
    print(f"  has_recent_lab_test rate: {df[TemporalEvent.HAS_RECENT_LAB_TEST.value].mean():.3f}")
    print(f"  recent_lab_test_type (top 10):\n{df[TemporalEvent.RECENT_LAB_TEST_TYPE.value].value_counts().head(10)}")

    return df

def add_ses_postcode_feature(
    df: pd.DataFrame,
    patients: pd.DataFrame,
    trajectories: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enrich the temporal event dataframe with SES-WOA score per postal code,
    matched via patient_id -> postcode -> SES score for the nearest available year.

    """
    df = df.copy()

    ses_dir = PROJECT_ROOT / "data/ses.xlsx"

    raw = pd.read_excel(ses_dir, sheet_name='Tabel 1', header=2, skiprows=[3])

    # col 0 = Verslagjaar, col 1 = Viercijferige postcode, col 22 = Gemiddelde SES WOA
    raw = raw.iloc[:, [0, 1, 22]]
    raw.columns = ['year', 'pc4', 'ses_score']

    ses_lookup = (
        raw
        .assign(
            pc4=lambda x: pd.to_numeric(x['pc4'], errors='coerce'),
            year=lambda x: pd.to_numeric(x['year'], errors='coerce'),
            ses_score=lambda x: pd.to_numeric(x['ses_score'], errors='coerce'),
        )
        .dropna(subset=['pc4', 'year', 'ses_score'])
        .assign(
            pc4=lambda x: x['pc4'].astype(int),
            year=lambda x: x['year'].astype(int),
            ses_score=lambda x: x['ses_score'].astype('float32'),
        )
        .drop_duplicates(subset=['pc4', 'year'])
        .sort_values(['pc4', 'year'])
        .reset_index(drop=True)
    )

    year_min = ses_lookup['year'].min()
    year_max = ses_lookup['year'].max()

    print(f"SES lookup: {len(ses_lookup)} rows | "
          f"years {year_min}–{year_max} | "
          f"{ses_lookup['pc4'].nunique()} unique PC4 codes")
    print(f"SES score range: {ses_lookup['ses_score'].min():.3f} – "
          f"{ses_lookup['ses_score'].max():.3f}")

    # ── 3. Build trajectory_id → pc4 map via patients ─────────────────────────
    pc4_map = (
        patients[[Patient.ID.value, Patient.POST_CODE.value]]
        .drop_duplicates(subset=[Patient.ID.value])
        .assign(pc4=lambda x: (
            x[Patient.POST_CODE.value]
            .astype(str)
            .str.extract(r"(\d{4})", expand=False)
            .pipe(pd.to_numeric, errors='coerce')
            .astype('Int64')
        ))
        .dropna(subset=['pc4'])
        .rename(columns={Patient.ID.value: 'patient_id'})[['patient_id', 'pc4']]
    )

    traj_to_patient = (
        trajectories[[Trajectory.ID.value, Patient.ID.value]]
        .drop_duplicates(subset=[Trajectory.ID.value])
        .rename(columns={
            Trajectory.ID.value: "trajectory_id",
            Patient.ID.value:    "patient_id",
        })
    )

    traj_to_pc4 = traj_to_patient.merge(pc4_map, on="patient_id", how="left")

    # ── 4. Build working dataframe with event year ────────────────────────────
    working = (
        df[[Trajectory.ID.value, TemporalEvent.DATE.value]]
        .copy()
        .assign(_year=lambda x: pd.to_datetime(x[TemporalEvent.DATE.value]).dt.year)
        .merge(
            traj_to_pc4,
            left_on=Trajectory.ID.value,
            right_on="trajectory_id",
            how="left",
        )
    )
    # Clamp event year to [year_min, year_max] so out-of-range years map to
    # the nearest boundary rather than returning NaN
    working['_year_clamped'] = working['_year'].clip(
        lower=year_min, upper=year_max
    )
    # ── 5. Nearest-year merge using merge_asof ────────────────────────────────
    DEFAULT_YEAR = 2021

    ses_2021 = (
        ses_lookup[ses_lookup['year'] == DEFAULT_YEAR][['pc4', 'ses_score']]
        .copy()
    )
    ses_2021['pc4'] = ses_2021['pc4'].astype('Int64')

    working['pc4'] = working['pc4'].astype('Int64')

    working = working.merge(
        ses_2021,
        on='pc4',
        how='left',
    )
    df[TemporalEvent.PATIENT_SES.value] = (
        working['ses_score']
        .astype('float32')
        .values
    )
    # ── 6. Diagnostics ─────────────────────────────────────────────────────────
    n_total  = len(df)
    n_filled = df[TemporalEvent.PATIENT_SES.value].notna().sum()
    n_no_pc4 = working["pc4"].isna().sum()
    n_year_gap = working['_year'].lt(year_min).sum() + working['_year'].gt(year_max).sum()

    print(f"SES score feature added.")
    print(f"  Filled      : {n_filled} / {n_total} ({n_filled / n_total:.1%})")
    print(f"  Missing PC4 : {n_no_pc4} rows (patient had no postal code)")
    print(f"  Year clamped: {n_year_gap} rows (event year outside "
          f"{year_min}–{year_max}, nearest year used)")
    if n_filled > 0:
        print(f"  Score range : {df[TemporalEvent.PATIENT_SES.value].min():.2f} – "
              f"{df[TemporalEvent.PATIENT_SES.value].max():.2f}")

    return df

def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ratio-based features for note author familiarity/diversity
    and appointment clinician familiarity/diversity.
    
    These replace raw counts with length-invariant ratios that are
    comparable across patients with different trajectory lengths.
    
    Should be applied after build_event_dataframe() on the loaded parquet.
    """

    note_mask = df[TemporalEvent.TYPE.value] == EventType.NOTE.value
    appt_mask = df[TemporalEvent.TYPE.value] == EventType.APPOINTMENT.value

    # ── NOTE: cumulative total notes so far per trajectory ────────────────────
    # We need total notes up to and including current event as denominator
    note_cumcount = (
        note_mask.astype(int)
        .groupby(df[Trajectory.ID.value])
        .cumsum()
    )  # 1-indexed at note events, frozen at last value for non-note events

    # author_familiarity: fraction of notes written by this author so far
    # AUTHOR_NOTE_COUNT_SO_FAR is 0-indexed (count before current), so add 1
    # for current note, divide by total notes so far (also 1-indexed)
    note_total = note_cumcount.where(note_mask, other=0)
    author_count_including_current = df[TemporalEvent.AUTHOR_NOTE_COUNT_SO_FAR.value] + note_mask.astype(float)

    df[TemporalEvent.AUTHOR_FAMILIARITY.value] = (
        (author_count_including_current / note_total.clip(lower=1))
        .where(note_mask, other=0.0)
        .astype("float32")
    )

    # author_diversity: fraction of unique authors to total notes so far
    df[TemporalEvent.AUTHOR_DIVERSITY.value] = (
        (df[TemporalEvent.NUM_UNIQUE_AUTHORS_SO_FAR.value] / note_total.clip(lower=1))
        .where(note_mask, other=0.0)
        .astype("float32")
    )

    # ── APPOINTMENT: cumulative total appointments so far ─────────────────────
    appt_cumcount = (
        appt_mask.astype(int)
        .groupby(df[Trajectory.ID.value])
        .cumsum()
    )
    appt_total = appt_cumcount.where(appt_mask, other=0)

    # clinician_familiarity: fraction of appointments with this clinician so far
    # APPOINTMENT_EMPLOYEE_COUNT_SO_FAR is 0-indexed, add 1 for current
    appt_count_including_current = df[TemporalEvent.APPOINTMENT_EMPLOYEE_COUNT_SO_FAR.value] + appt_mask.astype(float)

    df[TemporalEvent.APPOINTMENT_CLINICIAN_FAMILIARITY.value] = (
        (appt_count_including_current / appt_total.clip(lower=1))
        .where(appt_mask, other=0.0)
        .astype("float32")
    )

    # clinician_diversity: fraction of unique clinicians to total appointments so far
    df[TemporalEvent.APPOINTMENT_CLINICIAN_DIVERSITY.value] = (
        (df[TemporalEvent.APPOINTMENT_NUM_UNIQUE_EMPLOYEES_SO_FAR.value] / appt_total.clip(lower=1))
        .where(appt_mask, other=0.0)
        .astype("float32")
    )

    return df

def construct_trajectory_med_changes(
        df: pd.DataFrame
):
    """
    Parameters
    ----------
    df : pd.DataFrame
        Event-level dataframe (one row per medication event).
    Returns
    -------
    pd.DataFrame
    DataFrame with columns:
        - trajectory_id
        - med_changes (defaultdict(list): date -> list of changes)
    """

    # Ensure date is a string (YYYY-MM-DD) for dictionary keys
    df = df.copy()
    df[TemporalEvent.DATE.value] = pd.to_datetime(df[TemporalEvent.DATE.value]).dt.date.astype(str)
    med_events = df[
        df[TemporalEvent.TYPE.value].isin(
            [EventType.MEDICATION_START.value, EventType.MEDICATION_STOP.value]
        )
    ]
    rows = []

    for trajectory_id, traj_df in med_events.groupby(Trajectory.ID.value):
        med_changes = collections.defaultdict(list)

        for _, row in traj_df.iterrows():
            date = row[TemporalEvent.DATE.value]
            for col in med_count_cols:
                if col in row and row[col] > 0:
                    med_changes[date].append(col)

        rows.append({
            TrajectoryLevelOutcomes.TRAJECTORY_ID.value: trajectory_id,
            TrajectoryLevelOutcomes.MEDICATION_CHANGES.value: med_changes
        })

    return pd.DataFrame(rows)


def get_active_meds_long(df_sorted, medication_data):
    """
    Returns a long-format dataframe with one row per (event_id, medication),
    including the duration of the medication up to the event.

    Assumes df_sorted already has:
        - TemporalEvent.ID.value (unique per event)
        - TemporalEvent.DATE.value
        - Trajectory.ID.value
    """
    # Make copies to avoid modifying originals
    events = df_sorted[[Trajectory.ID.value, TemporalEvent.ID.value, TemporalEvent.DATE.value]].copy()
    meds = medication_data.copy()

    # Ensure datetime
    meds[Medication.START.value] = pd.to_datetime(meds[Medication.START.value])
    meds[Medication.STOP.value] = pd.to_datetime(meds[Medication.STOP.value])
    events[TemporalEvent.DATE.value] = pd.to_datetime(events[TemporalEvent.DATE.value])

    # Merge events with meds by trajectory
    merged = events.merge(
        meds,
        on=Trajectory.ID.value,
        how="left",
        suffixes=("", "_med")
    )

    # Filter only medications active at event time
    active = merged[
        (merged[Medication.START.value] <= merged[TemporalEvent.DATE.value]) &
        (merged[Medication.STOP.value] >= merged[TemporalEvent.DATE.value])
        ].copy()

    # Compute duration up to the event (in days)
    active[Medication.DURATION.value] = (
                                                active[TemporalEvent.DATE.value] - active[Medication.START.value]
                                        ).dt.days + 1

    # Keep only relevant columns
    active = active[[
        TemporalEvent.ID.value,
        Medication.BASE_ATC_CODE.value,
        Medication.DURATION.value,
        Medication.ANTIDEPRESSANT_TYPE.value,
        Medication.START.value,
    ]].drop_duplicates()

    # Convert ATC codes to category for memory efficiency
    active[Medication.BASE_ATC_CODE.value] = active[Medication.BASE_ATC_CODE.value].astype("category")
    active[Medication.ANTIDEPRESSANT_TYPE.value] = active[Medication.ANTIDEPRESSANT_TYPE.value].astype("category")

    n_before = len(active)
    active = (
        active
        .sort_values(Medication.DURATION.value, ascending=False)
        .drop_duplicates(
            subset=[TemporalEvent.ID.value, Medication.BASE_ATC_CODE.value],
            keep='first',
        )
    )
    n_after = len(active)
    print(f"  Active meds unique-per-event dedup: {n_before} -> {n_after} rows "
          f"({n_before - n_after} duplicate rows dropped)")

    # Convert ATC codes to category for memory efficiency
    active[Medication.BASE_ATC_CODE.value] = active[Medication.BASE_ATC_CODE.value].astype("category")

    return active



def get_historical_meds_long(df_sorted: pd.DataFrame, medication_data: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a long-format dataframe with one row per (event_id, medication)
    for medications that were previously prescribed but are NOT currently
    active at the time of the event.

    Output columns:
        - TemporalEvent.ID.value
        - Medication.BASE_ATC_CODE.value
        - Medication.DURATION.value  (total duration in days: stop - start + 1)
        - time_since_stopped         (days between medication stop and event date)
    """
    events = df_sorted[[
        Trajectory.ID.value,
        TemporalEvent.ID.value,
        TemporalEvent.DATE.value
    ]].copy()
    meds = medication_data.copy()

    # Ensure datetime
    meds[Medication.START.value] = pd.to_datetime(meds[Medication.START.value])
    meds[Medication.STOP.value]  = pd.to_datetime(meds[Medication.STOP.value])
    events[TemporalEvent.DATE.value] = pd.to_datetime(events[TemporalEvent.DATE.value])

    # Merge events with meds by trajectory
    merged = events.merge(
        meds,
        on=Trajectory.ID.value,
        how="left",
        suffixes=("", "_med")
    )

    # Previously started AND already stopped before this event
    historical = merged[
        (merged[Medication.START.value] <  merged[TemporalEvent.DATE.value]) &
        (merged[Medication.STOP.value]  <  merged[TemporalEvent.DATE.value])
    ].copy()

    # Exclude medications that are also active at this event
    # (a patient may have restarted the same ATC code — keep only truly stopped ones)
    active = merged[
        (merged[Medication.START.value] <= merged[TemporalEvent.DATE.value]) &
        (merged[Medication.STOP.value]  >= merged[TemporalEvent.DATE.value])
    ][[TemporalEvent.ID.value, Medication.BASE_ATC_CODE.value]].drop_duplicates()
    active["_is_active"] = True

    historical = historical.merge(
        active,
        on=[TemporalEvent.ID.value, Medication.BASE_ATC_CODE.value],
        how="left"
    )
    historical = historical[historical["_is_active"].isna()].drop(columns=["_is_active"])

    # Total duration of the medication course (days)
    historical[Medication.DURATION.value] = (
        historical[Medication.STOP.value] - historical[Medication.START.value]
    ).dt.days + 1

    # How long ago the medication was stopped relative to this event
    historical["time_since_stopped"] = (
        historical[TemporalEvent.DATE.value] - historical[Medication.STOP.value]
    ).dt.days

    # Keep only relevant columns; one row per (event, medication course)
    historical = historical[[
        TemporalEvent.ID.value,
        Medication.BASE_ATC_CODE.value,
        Medication.DURATION.value,
        "time_since_stopped",
    ]].drop_duplicates()

    historical[Medication.BASE_ATC_CODE.value]  = historical[Medication.BASE_ATC_CODE.value].astype("category")
    historical[Medication.DURATION.value]  = historical[Medication.DURATION.value].astype("float32")
    historical["time_since_stopped"]       = historical["time_since_stopped"].astype("float32")

    n_events_with_history = historical[TemporalEvent.ID.value].nunique()
    n_total_events        = len(df_sorted)
    print(f"Historical medication long table built.")
    print(f"  Rows           : {len(historical)}")
    print(f"  Events covered : {n_events_with_history} / {n_total_events} ({n_events_with_history / n_total_events:.1%})")
    print(f"  Unique ATC codes: {historical[Medication.BASE_ATC_CODE.value].nunique()}")

    return historical

def get_historical_diagnoses_long(
    df_sorted: pd.DataFrame,
    diagnosis_data: pd.DataFrame,
    diag_type: str = "main",  # "main" or "secondary"
) -> pd.DataFrame:
    """
    Returns a long-format dataframe with one row per (event_id, diagnosis)
    for diagnoses that were previously active but are NOT the current active
    diagnosis at the time of the event.

    """
    active_col = (
        TemporalEvent.ACTIVE_DIAGNOSIS_MAIN.value
        if diag_type == "main"
        else TemporalEvent.ACTIVE_DIAGNOSIS_SECONDARY.value
    )

    events = df_sorted[[
        Trajectory.ID.value,
        TemporalEvent.ID.value,
        TemporalEvent.DATE.value,
        active_col,
    ]].copy()
    events[TemporalEvent.DATE.value] = pd.to_datetime(events[TemporalEvent.DATE.value])

    dx = diagnosis_data[diagnosis_data[Diagnosis.TYPE.value] == diag_type].copy()
    dx[Diagnosis.DATE.value] = pd.to_datetime(dx[Diagnosis.DATE.value])

    # Compute diagnosis duration: days until next diagnosis of same type
    # in the same trajectory, used as a proxy for how long the episode was active
    dx = dx.sort_values([Trajectory.ID.value, Diagnosis.DATE.value])
    dx["_next_date"] = (
        dx.groupby(Trajectory.ID.value)[Diagnosis.DATE.value]
        .shift(-1)
    )

    # Merge events with all diagnoses of this type for the same trajectory
    merged = events.merge(
        dx[[Trajectory.ID.value, Diagnosis.DATE.value, Diagnosis.DSM_CODE.value, "_next_date"]],
        on=Trajectory.ID.value,
        how="left",
    )

    # Keep only diagnoses that occurred strictly before this event
    historical = merged[
        merged[Diagnosis.DATE.value] < merged[TemporalEvent.DATE.value]
    ].copy()

    # Exclude diagnoses that match the current active diagnosis at this event
    # using raw DSM codes directly
    historical = historical[
        historical[Diagnosis.DSM_CODE.value] != historical[active_col]
    ].copy()

    # Exclude UNKNOWN / missing
    historical = historical[
        historical[Diagnosis.DSM_CODE.value].notna() &
        (historical[Diagnosis.DSM_CODE.value] != "UNKNOWN")
    ]

    # Days since this diagnosis was recorded, relative to the event
    historical["time_since_diagnosis"] = (
        historical[TemporalEvent.DATE.value] - historical[Diagnosis.DATE.value]
    ).dt.days.astype("float32")

    # Duration: days until next diagnosis of same type, or until event date if no successor
    next_or_event = historical["_next_date"].fillna(historical[TemporalEvent.DATE.value])
    historical["diagnosis_duration"] = (
        next_or_event - historical[Diagnosis.DATE.value]
    ).dt.days.clip(lower=0).astype("float32")

    historical = historical[[
        TemporalEvent.ID.value,
        Diagnosis.DSM_CODE.value,
        "time_since_diagnosis",
        "diagnosis_duration",
    ]].drop_duplicates()

    historical[Diagnosis.DSM_CODE.value] = historical[Diagnosis.DSM_CODE.value].astype("category")

    n_events_with_history = historical[TemporalEvent.ID.value].nunique()
    n_total_events        = len(df_sorted)
    print(f"Historical {diag_type} diagnosis long table built.")
    print(f"  Rows            : {len(historical)}")
    print(f"  Events covered  : {n_events_with_history} / {n_total_events} "
          f"({n_events_with_history / n_total_events:.1%})")
    print(f"  Unique DSM codes: {historical[Diagnosis.DSM_CODE.value].nunique()}")

    return historical


def get_pre_trajectory_diagnoses_long(
    df_sorted: pd.DataFrame,
    diagnosis_data: pd.DataFrame,
    trajectory_data: pd.DataFrame,
    diag_type: str = "main",  # "main" or "secondary"
) -> pd.DataFrame:
    """
    Returns a long-format dataframe with one row per (event_id, diagnosis)
    for diagnoses registered BEFORE the trajectory start date for the same patient.

    Mirrors get_historical_diagnoses_long but scoped to pre-trajectory history
    rather than within-trajectory history.

    Output columns:
        - TemporalEvent.ID.value
        - Diagnosis.DSM_CODE.value
        - time_since_diagnosis   (days from diagnosis date to event date)
        - diagnosis_duration     (days from diagnosis date to trajectory start, as proxy)
    """
    traj_start = trajectory_data[[
        Trajectory.ID.value,
        Patient.ID.value,
        Trajectory.START.value,
    ]].copy()
    traj_start[Trajectory.START.value] = pd.to_datetime(traj_start[Trajectory.START.value])

    dx_data = diagnosis_data[diagnosis_data[Diagnosis.TYPE.value] == diag_type].copy()
    dx_data[Diagnosis.DATE.value] = pd.to_datetime(dx_data[Diagnosis.DATE.value])

    # Merge events with trajectory start info to get patient_id and traj start date
    events = df_sorted[[
        Trajectory.ID.value,
        TemporalEvent.ID.value,
        TemporalEvent.DATE.value,
    ]].copy()
    events[TemporalEvent.DATE.value] = pd.to_datetime(events[TemporalEvent.DATE.value])
    events = events.merge(traj_start, on=Trajectory.ID.value, how="left")

    # Merge events with diagnoses on patient_id
    merged = events.merge(
        dx_data[[Patient.ID.value, Diagnosis.DATE.value, Diagnosis.DSM_CODE.value]],
        on=Patient.ID.value,
        how="left",
    )

    # Keep only diagnoses that occurred before the trajectory start
    pre_traj = merged[
        merged[Diagnosis.DATE.value] < merged[Trajectory.START.value]
    ].copy()

    # Exclude UNKNOWN / missing
    pre_traj = pre_traj[
        pre_traj[Diagnosis.DSM_CODE.value].notna() &
        (pre_traj[Diagnosis.DSM_CODE.value] != "UNKNOWN")
    ]

    # Days from diagnosis to event date
    pre_traj["time_since_diagnosis"] = (
        pre_traj[TemporalEvent.DATE.value] - pre_traj[Diagnosis.DATE.value]
    ).dt.days.astype("float32")

    # Duration proxy: days from diagnosis date to trajectory start
    pre_traj["diagnosis_duration"] = (
        pre_traj[Trajectory.START.value] - pre_traj[Diagnosis.DATE.value]
    ).dt.days.clip(lower=0).astype("float32")

    pre_traj = pre_traj[[
        TemporalEvent.ID.value,
        Diagnosis.DSM_CODE.value,
        "time_since_diagnosis",
        "diagnosis_duration",
    ]].drop_duplicates()

    pre_traj[Diagnosis.DSM_CODE.value] = pre_traj[Diagnosis.DSM_CODE.value].astype("category")

    n_events_with_history = pre_traj[TemporalEvent.ID.value].nunique()
    n_total_events        = len(df_sorted)
    print(f"Pre-trajectory {diag_type} diagnosis long table built.")
    print(f"  Rows            : {len(pre_traj)}")
    print(f"  Events covered  : {n_events_with_history} / {n_total_events} "
          f"({n_events_with_history / n_total_events:.1%})")
    print(f"  Unique DSM codes: {pre_traj[Diagnosis.DSM_CODE.value].nunique()}")

    return pre_traj

def enrich_clinical_note_features(df_sorted: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the event dataframe with features derived from clinical note metadata.
    Specifically processes NOTE events to extract:
        - Creation employee experience (years)
        - Mutation employee experience (years)
        - Creation employee role
        - Mutation employee role
    """

    # clinical note: text, employee years of experience, role are processed.
    note_mask = df_sorted[TemporalEvent.TYPE.value] == EventType.NOTE.value

    event_dates = pd.to_datetime(df_sorted.loc[note_mask, TemporalEvent.DATE.value])
    df_sorted.loc[note_mask, TemporalEvent.CREATION_EMPLOYEE_EXPERIENCE.value] = (
            (event_dates - pd.to_datetime(df_sorted.loc[note_mask, ClinicalText.CREATION_EMPLOYEE_START.value])).dt.days / 365.25
    )
    df_sorted.loc[note_mask, TemporalEvent.MUTATION_EMPLOYEE_EXPERIENCE.value] = (
            (event_dates - pd.to_datetime(df_sorted.loc[note_mask, ClinicalText.MUTATION_EMPLOYEE_START.value])).dt.days / 365.25
    )

    # Fill NaN values with 0, Zeros representing actual missing data from clinical note event or
    # missingness due to other event type can be differentiated with event flag.
    df_sorted[TemporalEvent.CREATION_EMPLOYEE_EXPERIENCE.value] = (
        df_sorted[TemporalEvent.CREATION_EMPLOYEE_EXPERIENCE.value]
        .fillna(0)
        .astype("float32")
    )

    df_sorted[TemporalEvent.MUTATION_EMPLOYEE_EXPERIENCE.value] = (
        df_sorted[TemporalEvent.MUTATION_EMPLOYEE_EXPERIENCE.value]
        .fillna(0)
        .astype("float32")
    )

    df_sorted[TemporalEvent.NOTE_CREATION_EMPLOYEE_ROLE.value] = (
        df_sorted[TemporalEvent.NOTE_CREATION_EMPLOYEE_ROLE.value]
        .fillna("")
        .astype("string")
    )
    df_sorted[TemporalEvent.NOTE_MUTATION_EMPLOYEE_ROLE.value] = (
        df_sorted[TemporalEvent.NOTE_MUTATION_EMPLOYEE_ROLE.value]
        .fillna("")
        .astype("string")
    )

    note_df = df_sorted.loc[note_mask].copy()
    note_df["_author_cumcount"] = (
        note_df.groupby([Trajectory.ID.value, ClinicalText.CREATION_PRACTITIONER_CODE.value])
        .cumcount()  # 0-indexed count before current row
    )
    df_sorted.loc[note_mask, TemporalEvent.AUTHOR_NOTE_COUNT_SO_FAR.value] = note_df["_author_cumcount"].values

    # num_unique_authors_in_trajectory: distinct authors seen up to and including current event.
    def count_unique_authors_so_far(group):
        authors = group[ClinicalText.CREATION_PRACTITIONER_CODE.value]
        return pd.Series(
            [authors.iloc[:i+1].nunique() for i in range(len(authors))],
            index=group.index
        )

    df_sorted.loc[note_mask, TemporalEvent.NUM_UNIQUE_AUTHORS_SO_FAR.value] = (
        note_df.groupby(Trajectory.ID.value, group_keys=False)
        .apply(count_unique_authors_so_far)
    )

    # time_of_day_bucket: morning / afternoon / evening / night
    note_hours = pd.to_datetime(df_sorted.loc[note_mask, TemporalEvent.DATE.value]).dt.hour
    df_sorted.loc[note_mask, TemporalEvent.TIME_OF_DAY_BUCKET.value] = pd.cut(
        note_hours,
        bins=[0, 6, 12, 18, 24],
        labels=["night", "morning", "afternoon", "evening"],
        right=False
    ).astype("string")

    # Fill NaN values for all non-note events
    df_sorted[TemporalEvent.AUTHOR_NOTE_COUNT_SO_FAR.value] = df_sorted[TemporalEvent.AUTHOR_NOTE_COUNT_SO_FAR.value].fillna(0).astype("float32")
    df_sorted[TemporalEvent.NUM_UNIQUE_AUTHORS_SO_FAR.value] = df_sorted[TemporalEvent.NUM_UNIQUE_AUTHORS_SO_FAR.value].fillna(0).astype("float32")
    df_sorted[TemporalEvent.TIME_OF_DAY_BUCKET.value] = df_sorted[TemporalEvent.TIME_OF_DAY_BUCKET.value].fillna("").astype("string")
    ####################################################################################################################

    return df_sorted  # placeholder if no enrichment is needed

def enrich_appointment_features(df_sorted: pd.DataFrame, appointment_data: pd.DataFrame) -> pd.DataFrame:
    df_sorted[TemporalEvent.APPOINTMENT_DURATION.value] = df_sorted[TemporalEvent.APPOINTMENT_DURATION.value].astype("float32").fillna(0)

    appointment_mask = df_sorted[TemporalEvent.TYPE.value] == EventType.APPOINTMENT.value
    appt_df = df_sorted.loc[appointment_mask].copy()

    event_dates = pd.to_datetime(df_sorted.loc[appointment_mask, TemporalEvent.DATE.value])
    df_sorted.loc[appointment_mask, TemporalEvent.APPOINTMENT_EMPLOYEE_EXPERIENCE.value] = (
            (event_dates - pd.to_datetime(df_sorted.loc[appointment_mask, Appointment.EMPLOYEE_START.value])).dt.days / 365.25
    )
    # missingness due to other event type can be differentiated with event flag.
    df_sorted[TemporalEvent.APPOINTMENT_EMPLOYEE_EXPERIENCE.value] = (
        df_sorted[TemporalEvent.APPOINTMENT_EMPLOYEE_EXPERIENCE.value]
        .fillna(0)
        .astype("float32")
    )
    
    df_sorted[TemporalEvent.APPOINTMENT_ROLE.value] = (
        df_sorted[TemporalEvent.APPOINTMENT_ROLE.value]
        .fillna("")
        .astype("string")
    )

    # indirect_to_duration_ratio: high ratio may indicate complex case coordination outside appointment
    df_sorted.loc[appointment_mask, TemporalEvent.INDIRECT_TO_DURATION_RATIO.value] = (
        appt_df[Appointment.INDIRECT_TIME.value].astype("float32") /
        (appt_df[Appointment.DURATION.value].astype("float32") + 1)
    )
    # time_of_day_bucket from appointment start time
    appt_hours = pd.to_datetime(appt_df[Appointment.START.value]).dt.hour
    df_sorted.loc[appointment_mask, TemporalEvent.APPOINTMENT_TIME_OF_DAY_BUCKET.value] = pd.cut(
        appt_hours,
        bins=[0, 6, 12, 18, 24],
        labels=["night", "morning", "afternoon", "evening"],
        right=False
    ).astype("string")

    # employee_appointment_count_so_far: prior appointments with this specific employee in trajectory
    df_sorted.loc[appointment_mask, TemporalEvent.APPOINTMENT_EMPLOYEE_COUNT_SO_FAR.value] = (
        appt_df.groupby([Trajectory.ID.value, Appointment.EMPLOYEE_ID.value])
        .cumcount()
    ).astype("float32")

    # num_unique_employees_so_far: distinct employees seen up to and including current appointment
    def count_unique_employees_so_far(group):
        employees = group[Appointment.EMPLOYEE_ID.value]
        return pd.Series(
            [employees.iloc[:i + 1].nunique() for i in range(len(employees))],
            index=group.index
        )

    df_sorted.loc[appointment_mask, TemporalEvent.APPOINTMENT_NUM_UNIQUE_EMPLOYEES_SO_FAR.value] = (
        appt_df.groupby(Trajectory.ID.value, group_keys=False)
        .apply(count_unique_employees_so_far)
    )
    # is_same_employee_as_last: 1 if same employee as previous appointment in trajectory
    df_sorted.loc[appointment_mask, TemporalEvent.APPOINTMENT_IS_SAME_EMPLOYEE_AS_LAST.value] = (
        appt_df.groupby(Trajectory.ID.value)[Appointment.EMPLOYEE_ID.value]
        .transform(lambda x: (x == x.shift(1)).astype("float32"))
    )
    # Fill NaN for non-appointment events
    appt_float_cols = [
        TemporalEvent.INDIRECT_TO_DURATION_RATIO.value,
        TemporalEvent.APPOINTMENT_EMPLOYEE_COUNT_SO_FAR.value,
        TemporalEvent.APPOINTMENT_NUM_UNIQUE_EMPLOYEES_SO_FAR.value,
        TemporalEvent.APPOINTMENT_IS_SAME_EMPLOYEE_AS_LAST.value,
    ]
    for col in appt_float_cols:
        df_sorted[col] = df_sorted[col].fillna(0).astype("float32")

    df_sorted[TemporalEvent.APPOINTMENT_TIME_OF_DAY_BUCKET.value] = (
        df_sorted[TemporalEvent.APPOINTMENT_TIME_OF_DAY_BUCKET.value].fillna("").astype("string")
    )
    return df_sorted

def enrich_lab_results(df_sorted: pd.DataFrame, lab_results: pd.DataFrame) -> pd.DataFrame:
    def has_recent_abnormal_lab(row, lab_results):
        traj_labs = lab_results[
            lab_results[Trajectory.ID.value] == row[Trajectory.ID.value]
        ]
        if traj_labs.empty:
            return 0.0

        event_dt = pd.to_datetime(row[TemporalEvent.DATE.value])
        window_start = event_dt - pd.Timedelta(days=7)

        recent_labs = traj_labs[
            (pd.to_datetime(traj_labs[LabResults.DATE.value]) >= window_start) &
            (pd.to_datetime(traj_labs[LabResults.DATE.value]) < event_dt)
        ]

        if recent_labs.empty:
            return 0.0

        return float(
            recent_labs[LabResults.IN_RANGE.value]
            .isin(["low", "high"])
            .any()
        )

    df_sorted[TemporalEvent.HAS_RECENT_ABNORMAL_LAB.value] = df_sorted.apply(
        lambda row: has_recent_abnormal_lab(row, lab_results),
        axis=1
    ).astype("float32")

    df_sorted = add_lab_features(df_sorted, lab_results=lab_results)  # diagnoses has the relevant lab columns
    ####################################################################################################################
    return df_sorted

def enrich_temporal_context(df_sorted: pd.DataFrame) -> pd.DataFrame:
    # these values should never be the NaN unless there is a bug or issue with the dataset.
    df_sorted[TemporalEvent.TIME_SINCE_START.value] = (
            pd.to_datetime(df_sorted[TemporalEvent.DATE.value]) -
            pd.to_datetime(df_sorted[Trajectory.START.value])
    ).dt.days.astype("float32")

    df_sorted[TemporalEvent.TIME_SINCE_LAST_EVENT.value] = (
        df_sorted.groupby(Trajectory.ID.value, group_keys=False)
        .apply(lambda g: g[TemporalEvent.TIME_SINCE_START.value].diff().fillna(0))
        .astype("float32")
        .reset_index(drop=True)
    )
    # trajectory_phase: early (0-90 days), middle (91-180 days), late (>180 days)
    # based on time_since_start which is already computed above
    df_sorted[TemporalEvent.TRAJECTORY_PHASE.value] = pd.cut(
        df_sorted[TemporalEvent.TIME_SINCE_START.value],
        bins=[-1, 90, 180, float("inf")],
        labels=["early", "middle", "late"]
    ).astype("string")
    return df_sorted

def enrich_admission_features(df_sorted: pd.DataFrame, admission_data: pd.DataFrame) -> pd.DataFrame:
    ########################################## ADMISSION ##############################################################
    def admission_info(row):
        # Select admissions for this patient
        adms = admission_data[admission_data[Trajectory.ID.value] == row[Trajectory.ID.value]]

        # Mask for admissions that include the event
        mask = (row[TemporalEvent.DATE.value] >= adms[Admission.START.value]) & \
               (row[TemporalEvent.DATE.value] <= adms[Admission.STOP.value])

        if mask.any():
            admission_start = adms[Admission.START.value][mask].iloc[0]
            during = 1.0  # during admission
            # added 1 to ensure time_elapsed for admission start is 1, but not 0.
            time_elapsed = (row[TemporalEvent.DATE.value] - admission_start).days + 1

        else:
            during = 0.0  # not during admission
            time_elapsed = 0

        return pd.Series({
            TemporalEvent.DURING_ADMISSION.value: during,
            TemporalEvent.TIME_SINCE_ADMISSION.value: time_elapsed
        })

    # Apply the function to the dataframe
    df_sorted[[TemporalEvent.DURING_ADMISSION.value, TemporalEvent.TIME_SINCE_ADMISSION.value]] = df_sorted.apply(
        admission_info, axis=1)

    # num_admissions_in_trajectory_so_far: cumulative admission_start events within trajectory, excluding current
    admission_start_mask = df_sorted[TemporalEvent.TYPE.value] == EventType.ADMISSION_START.value
    admission_start_int = admission_start_mask.astype(int)

    df_sorted[TemporalEvent.NUM_ADMISSIONS_SO_FAR.value] = (
        admission_start_int
        .groupby(df_sorted[Trajectory.ID.value])
        .cumsum()
        - admission_start_int
    ).astype("float32")

    return df_sorted

def enrich_active_diagnoses(df_sorted: pd.DataFrame, diagnosis_data: pd.DataFrame) -> pd.DataFrame:
     ########################################## DIAGNOSIS ###############################################################
    def get_active_latest_diagnoses(row, diagnosis_data):
        # Filter diagnoses for this trajectory up to the current event time
        traj_dx = diagnosis_data[
            (diagnosis_data[Trajectory.ID.value] == row[Trajectory.ID.value]) &
            (diagnosis_data[Diagnosis.DATE.value] <= row[TemporalEvent.DATE.value])
            ]

        def latest_of_type(dx_type):
            dx = traj_dx[traj_dx[Diagnosis.TYPE.value] == dx_type]
            if dx.empty:
                return "UNKNOWN", 0  # if active diagnosis is nan, it is set to NaN, and time elapsed to 0.

            # Pick diagnosis with latest date (as of current event)
            dx_latest = dx.loc[dx[Diagnosis.DATE.value].idxmax()]
            # this is a mix of dsm5 and dsm4 codes.
            dx_code = dx_latest[Diagnosis.DSM_CODE.value]
            # Compute time elapsed in days
            time_elapsed = (row[TemporalEvent.DATE.value] - dx_latest[Diagnosis.DATE.value]).days + 1
            return dx_code, time_elapsed

        latest_main, time_since_main = latest_of_type("main")
        latest_secondary, time_since_secondary = latest_of_type("secondary")

        return pd.Series({
            TemporalEvent.ACTIVE_DIAGNOSIS_MAIN.value: latest_main,
            TemporalEvent.TIME_SINCE_MAIN_DIAG.value: time_since_main,
            TemporalEvent.ACTIVE_DIAGNOSIS_SECONDARY.value: latest_secondary,
            TemporalEvent.TIME_SINCE_SECONDARY_DIAG.value: time_since_secondary
        })

    # return main and secondary diagnoses DSM-5 code.
    df_sorted = df_sorted.join(
        df_sorted.apply(
            lambda row: get_active_latest_diagnoses(row, diagnosis_data),
            axis=1
        )
    )

    # has_comorbid_diagnosis: 1 if both main and secondary diagnosis are active (non-UNKNOWN) at time of event
    df_sorted[TemporalEvent.HAS_COMORBID_DIAGNOSIS.value] = (
        (df_sorted[TemporalEvent.ACTIVE_DIAGNOSIS_MAIN.value] != "UNKNOWN") &
        (df_sorted[TemporalEvent.ACTIVE_DIAGNOSIS_SECONDARY.value] != "UNKNOWN")
    ).astype("float32")

    # has_history_of_comorbid_diagnosis: 1 if comorbidity was ever present at any prior event in this trajectory
    comorbid_int = df_sorted[TemporalEvent.HAS_COMORBID_DIAGNOSIS.value]

    df_sorted[TemporalEvent.HAS_HISTORY_OF_COMORBID_DIAGNOSIS.value] = (
        comorbid_int
        .groupby(df_sorted[Trajectory.ID.value])
        .cumsum()
        - comorbid_int
    ).clip(upper=1).astype("float32")

    ####################################################################################################################

    return df_sorted

def enrich_demographics(df_sorted: pd.DataFrame) -> pd.DataFrame:
    # TODO: SES from the postcode.
    #df_sorted[TemporalEvent.PATIENT_GENDER.value] = df_sorted[Patient.GENDER.value].map({'male': 0, 'female': 1}).astype(float)
    gender_dummies = pd.get_dummies(
        df_sorted[Patient.GENDER.value],
        prefix="gender",
        dtype=float
    )
    df_sorted = pd.concat([df_sorted, gender_dummies], axis=1)
    df_sorted[TemporalEvent.PATIENT_AGE_AT_START.value] = df_sorted[Trajectory.PATIENT_AGE_AT_START.value].astype(float)
    # fill missing ages as median from the dataset (there are no missing from UMCU).
    df_sorted[TemporalEvent.PATIENT_AGE_AT_START.value] = (
        df_sorted[TemporalEvent.PATIENT_AGE_AT_START.value]
        .fillna(df_sorted[TemporalEvent.PATIENT_AGE_AT_START.value].median())
        .astype("float32")
    )
    df_sorted[TemporalEvent.PATIENT_AGE.value] = (
            (pd.to_datetime(df_sorted[TemporalEvent.DATE.value]) - pd.to_datetime(df_sorted[Patient.BIRTH_DATE.value]))
            .dt.days / 365.25  # account for leap years
    )
    return df_sorted

def enrich_historical_med_changes(df_sorted: pd.DataFrame, medication_data: pd.DataFrame) -> pd.DataFrame:

    is_primary_med_change_event = (
            df_sorted[TemporalEvent.PRIMARY_ADDED.value] |
            df_sorted[TemporalEvent.PRIMARY_REMOVED.value]
    )
    is_modifier_med_change_event = (
            df_sorted[TemporalEvent.MODIFIER_ADDED.value] |
            df_sorted[TemporalEvent.MODIFIER_REMOVED.value]
    )
    is_other_med_change_event = (
            df_sorted[TemporalEvent.OTHER_ADDED.value] |
            df_sorted[TemporalEvent.OTHER_REMOVED.value]
    )

     
    # cumulative count of all previous events per trajectory
    df_sorted[TemporalEvent.NUM_PREV_EVENT.value] = (
            df_sorted.groupby(Trajectory.ID.value).cumcount()
        )
    primary_change_int = is_primary_med_change_event.astype(int)

    df_sorted[TemporalEvent.NUM_PREV_PRIMARY_MED_CHANGE.value] = (
            primary_change_int
            .groupby(df_sorted[Trajectory.ID.value])
            .cumsum()
            - primary_change_int
    )
    modifier_change_int = is_modifier_med_change_event.astype(int)
    other_change_int = is_other_med_change_event.astype(int)

    df_sorted[TemporalEvent.NUM_PREV_MODIFIER_MED_CHANGE.value] = (
            modifier_change_int
            .groupby(df_sorted[Trajectory.ID.value])
            .cumsum()
            - modifier_change_int
    )

    df_sorted[TemporalEvent.NUM_PREV_OTHER_MED_CHANGE.value] = (
            other_change_int
            .groupby(df_sorted[Trajectory.ID.value])
            .cumsum()
            - other_change_int
    )

    return df_sorted


def annotate_and_collapse_medication_events(df, medications,  dose_change_window_days=7,
    switch_window_days=14,):
    """
    1. Writes boolean count flags to each medication event row
    2. Computes active medication counts (main/modifier/other) from running state
    3. Collapses STOP+START pairs into single events:
       - Dose change (same base ATC, different dose): STOP dropped,
         START kept with dose flags and action_type
       - Switch (different base ATC, same type group, drug not continuing):
         STOP dropped, START becomes MEDICATION_SWITCH
  
    Active counts forward-filled to non-medication event rows.
    """
    DOSE_WINDOW   = pd.Timedelta(days=dose_change_window_days)
    SWITCH_WINDOW = pd.Timedelta(days=switch_window_days)
    MAX_WINDOW    = max(DOSE_WINDOW, SWITCH_WINDOW)
    
    FLAG_COLS = {
        "main":     (TemporalEvent.PRIMARY_ADDED.value,  TemporalEvent.PRIMARY_REMOVED.value,
                     TemporalEvent.PRIMARY_DOSE_INCREASED.value,  TemporalEvent.PRIMARY_DOSE_DECREASED.value),
        "modifier": (TemporalEvent.MODIFIER_ADDED.value, TemporalEvent.MODIFIER_REMOVED.value,
                     TemporalEvent.MODIFIER_DOSE_INCREASED.value, TemporalEvent.MODIFIER_DOSE_DECREASED.value),
        "other":    (TemporalEvent.OTHER_ADDED.value,    TemporalEvent.OTHER_REMOVED.value,
                     TemporalEvent.OTHER_DOSE_INCREASED.value,    TemporalEvent.OTHER_DOSE_DECREASED.value),
    }
    def get_cols(med_type):
        return FLAG_COLS.get(med_type, FLAG_COLS["other"])

    # ── setup ─────────────────────────────────────────────────────────
    df = df.copy()
    for col in med_count_cols:
        df[col] = 0
    df[TemporalEvent.COUNT_ACTIVE_MAIN_MED.value]     = 0
    df[TemporalEvent.COUNT_ACTIVE_MODIFIER_MED.value] = 0
    df[TemporalEvent.COUNT_ACTIVE_OTHER_MED.value]    = 0
    for col in [
        TemporalEvent.TRIGGER_MED_ATC_CODE.value,
        TemporalEvent.TRIGGER_MED_DRUG_CLASS.value,
        TemporalEvent.TRIGGER_MED_ACTION_TYPE.value,
        TemporalEvent.PREV_MED_ATC_CODE.value,
        TemporalEvent.PREV_MED_TYPE.value,
        TemporalEvent.PREV_DRUG_CLASS.value,
    ]:
        df[col] = ""

    medication_data = medications.copy()
    medication_data[Medication.START.value] = pd.to_datetime(medication_data[Medication.START.value])
    medication_data[Medication.STOP.value]  = pd.to_datetime(medication_data[Medication.STOP.value])

    stop_indices_to_drop = set()

    for traj_id, traj_df in df.groupby(TemporalEvent.TRAJECTORY_ID.value):

        # ── sort and index events ──────────────────────────────────────
        traj_df_sorted = (
            traj_df
            .sort_values(TemporalEvent.DATE.value)
            .reset_index()
        )
        traj_df_sorted["_event_order"] = traj_df_sorted.index

        meds_traj = medication_data[
            medication_data[Trajectory.ID.value] == traj_id
        ].copy()

        if meds_traj.empty:
            continue

        med_events = traj_df_sorted[
            traj_df_sorted[TemporalEvent.TYPE.value].isin([
                EventType.MEDICATION_START.value,
                EventType.MEDICATION_STOP.value,
            ])
        ][[Medication.ID.value, TemporalEvent.TYPE.value, "_event_order", TemporalEvent.DATE.value]]

        if med_events.empty:
            continue

        # ── map medication_id -> event order ───────────────────────────
        meds_traj["_start_order"] = meds_traj[Medication.ID.value].map(
            med_events[med_events[TemporalEvent.TYPE.value] == EventType.MEDICATION_START.value]
            .set_index(Medication.ID.value)["_event_order"]
        )
        meds_traj["_stop_order"] = meds_traj[Medication.ID.value].map(
            med_events[med_events[TemporalEvent.TYPE.value] == EventType.MEDICATION_STOP.value]
            .set_index(Medication.ID.value)["_event_order"]
        )

        first_order = med_events.iloc[0]["_event_order"]
        first_date  = med_events.iloc[0][TemporalEvent.DATE.value]

        # ── build initial state ────────────────────────────────────────
        current_state = {}
        for _, med in meds_traj[
            (
                (meds_traj["_start_order"].notna() & (meds_traj["_start_order"] < first_order)) |
                (meds_traj["_start_order"].isna()  & (meds_traj[Medication.START.value] < first_date))
            ) & (
                meds_traj[Medication.STOP.value].isna() |
                (meds_traj[Medication.STOP.value] > first_date)
            )
        ].iterrows():
            b = med[Medication.BASE_ATC_CODE.value]
            if b not in current_state:
                current_state[b] = {
                    "dose": 0.0,
                    "type": med[Medication.ANTIDEPRESSANT_TYPE.value],
                    "atc":  med[Medication.BASE_ATC_CODE.value],
                }
            current_state[b]["dose"] += med[Medication.DAY_DOSES.value]

        prev_state = {k: v.copy() for k, v in current_state.items()}

        # ── process events grouped by date ─────────────────────────────
        for event_date, day_events in med_events.groupby(
            pd.to_datetime(med_events[TemporalEvent.DATE.value]).dt.normalize(), sort=True
        ):
            day_indices = [traj_df_sorted.loc[i, "index"] for i in day_events.index]

            # ── apply day events to state ──────────────────────────────
            for _, row in day_events.iterrows():
                med_matches = meds_traj[meds_traj[Medication.ID.value] == row[Medication.ID.value]]
                if med_matches.empty:
                    continue
                med      = med_matches.iloc[0]
                b        = med[Medication.BASE_ATC_CODE.value]
                dose     = med[Medication.DAY_DOSES.value]
                med_type = med[Medication.ANTIDEPRESSANT_TYPE.value]
                base_atc = med[Medication.BASE_ATC_CODE.value]

                if row[TemporalEvent.TYPE.value] == EventType.MEDICATION_START.value:
                    if b not in current_state:
                        current_state[b] = {"dose": 0.0, "type": med_type, "atc": base_atc}
                    current_state[b]["dose"] += dose
                    current_state[b]["type"]  = med_type
                    current_state[b]["atc"]   = base_atc
                else:
                    if b in current_state:
                        current_state[b]["dose"] -= dose
                        if current_state[b]["dose"] <= 0:
                            del current_state[b]

            # ── write active counts from current state ─────────────────
            main_count     = sum(1 for v in current_state.values() if v["type"] == "main")
            modifier_count = sum(1 for v in current_state.values() if v["type"] == "modifier")
            other_count    = sum(1 for v in current_state.values() if v["type"] == "other")
            for idx in day_indices:
                df.at[idx, TemporalEvent.COUNT_ACTIVE_MAIN_MED.value]     = main_count
                df.at[idx, TemporalEvent.COUNT_ACTIVE_MODIFIER_MED.value] = modifier_count
                df.at[idx, TemporalEvent.COUNT_ACTIVE_OTHER_MED.value]    = other_count

            # ── skip if no real changes ────────────────────────────────
            prev_keys = set(prev_state)
            curr_keys = set(current_state)
            added     = curr_keys - prev_keys
            removed   = prev_keys - curr_keys
            common    = prev_keys & curr_keys

            if not added and not removed and all(
                current_state[k]["dose"] == prev_state[k]["dose"] and
                current_state[k]["type"] == prev_state[k]["type"]
                for k in common
            ):
                prev_state = {k: v.copy() for k, v in current_state.items()}
                continue

            # ── write boolean flags ────────────────────────────────────
            for b in added:
                col = get_cols(current_state[b]["type"])[0]
                for idx in day_indices:
                    df.at[idx, col] += 1

            for b in removed:
                col = get_cols(prev_state[b]["type"])[1]
                for idx in day_indices:
                    df.at[idx, col] += 1

            for b in common:
                p, c = prev_state[b], current_state[b]
                if c["dose"] == p["dose"] and c["type"] == p["type"]:
                    continue
                if c["dose"] > p["dose"] or (p["type"] == "modifier" and c["type"] == "main"):
                    col = get_cols(c["type"])[2]
                elif c["dose"] < p["dose"] or (p["type"] == "main" and c["type"] == "modifier"):
                    col = get_cols(c["type"])[3]
                else:
                    continue
                for idx in day_indices:
                    df.at[idx, col] += 1

            # ── collapse START+STOP pairs ──────────────────────────────
            day_starts = day_events[
                day_events[TemporalEvent.TYPE.value] == EventType.MEDICATION_START.value
            ]

            for _, start_row in day_starts.iterrows():
                start_orig_idx = traj_df_sorted.loc[start_row.name, "index"]
                if start_orig_idx in stop_indices_to_drop:
                    continue

                start_med_rows = meds_traj[
                    meds_traj[Medication.ID.value] == start_row[Medication.ID.value]
                ]
                if start_med_rows.empty:
                    continue

                start_med      = start_med_rows.iloc[0]
                start_base_atc = start_med[Medication.BASE_ATC_CODE.value]
                start_type     = start_med[Medication.ANTIDEPRESSANT_TYPE.value]
                start_dose     = start_med[Medication.DAY_DOSES.value]
                start_date     = pd.to_datetime(start_row[TemporalEvent.DATE.value])
                t              = start_type if start_type in ("main", "modifier") else "other"

                # candidate STOP events within window (bidirectional)
                traj_stop_events = traj_df_sorted[
                    traj_df_sorted[TemporalEvent.TYPE.value] == EventType.MEDICATION_STOP.value
                ]
                candidate_stops = traj_stop_events[
                    ~traj_stop_events["index"].isin(stop_indices_to_drop) &
                    (pd.to_datetime(traj_stop_events[TemporalEvent.DATE.value]) >= start_date - MAX_WINDOW) &
                    (pd.to_datetime(traj_stop_events[TemporalEvent.DATE.value]) <= start_date + MAX_WINDOW)
                ].copy()
                                
                
                if candidate_stops.empty:
                    continue

                # ── prioritise same base ATC (dose change) over switch ─────
                candidate_stops["_same_base_atc"] = candidate_stops[
                    Medication.ID.value  # medication_id column on the event row
                ].map(
                    meds_traj.set_index(Medication.ID.value)[Medication.BASE_ATC_CODE.value]
                ) == start_base_atc

                candidate_stops["_date_diff"] = (
                    pd.to_datetime(candidate_stops[TemporalEvent.DATE.value]) - start_date
                ).abs()

                candidate_stops = candidate_stops.sort_values(
                    ["_same_base_atc", "_date_diff"],
                    ascending=[False, True]  # same base ATC first, then closest date
                )
                # ─────────────────────────────────────────────────────────────
                for _, stop_row in candidate_stops.iterrows():
                    stop_orig_idx = stop_row["index"]
                    stop_med_rows = meds_traj[
                        meds_traj[Medication.ID.value] == stop_row[Medication.ID.value]
                    ]
                    if stop_med_rows.empty:
                        continue

                    stop_med      = stop_med_rows.iloc[0]
                    stop_base_atc = stop_med[Medication.BASE_ATC_CODE.value]
                    stop_type     = stop_med[Medication.ANTIDEPRESSANT_TYPE.value]
                    stop_dose     = stop_med[Medication.DAY_DOSES.value]

                    if stop_type != start_type:
                        continue

                    # DOSE CHANGE: same base ATC
                    if stop_base_atc == start_base_atc:
                        if stop_row["_date_diff"] > DOSE_WINDOW:
                            continue
                        dose_changed = (
                            not pd.isna(stop_dose) and
                            not pd.isna(start_dose) and
                            stop_dose != start_dose
                        )
                        if not dose_changed:
                            continue

                        if start_dose > stop_dose:
                            action = f"{t}_dose_increase"
                        elif start_dose < stop_dose:
                            action = f"{t}_dose_decrease"
                        else:
                            action = f"{t}_dose_change"

                        df.at[start_orig_idx, TemporalEvent.TRIGGER_MED_ACTION_TYPE.value] = action
                        stop_indices_to_drop.add(stop_orig_idx)
                        break

                    # SWITCH: different base ATC
                    else:
                        if stop_row["_date_diff"] > SWITCH_WINDOW:
                            continue
                        if pd.to_datetime(stop_med[Medication.STOP.value]) > start_date + SWITCH_WINDOW:
                            continue
                        # switch only makes sense for main depression medications..
                        if start_type == "other":
                            start_class = start_med[Medication.DRUG_CLASS.value]
                            stop_class  = stop_med[Medication.DRUG_CLASS.value]
                            if start_class == "other" or stop_class == "other":
                                continue
                            if start_class != stop_class:
                                continue

                        df.at[start_orig_idx, TemporalEvent.TYPE.value]                    = EventType.MEDICATION_SWITCH.value
                        df.at[start_orig_idx, TemporalEvent.PREV_MED_ATC_CODE.value]       = stop_med[Medication.BASE_ATC_CODE.value]
                        df.at[start_orig_idx, TemporalEvent.PREV_DRUG_CLASS.value]         = stop_med[Medication.DRUG_CLASS.value]
                        df.at[start_orig_idx, TemporalEvent.PREV_MED_TYPE.value]           = stop_med[Medication.ANTIDEPRESSANT_TYPE.value]
                        df.at[start_orig_idx, TemporalEvent.TRIGGER_MED_ACTION_TYPE.value] = f"{t}_switch"

                        stop_indices_to_drop.add(stop_orig_idx)
                        break

            prev_state = {k: v.copy() for k, v in current_state.items()}

    # ── drop redundant STOP rows ───────────────────────────────────────
    df = df.drop(index=list(stop_indices_to_drop)).reset_index(drop=True)

    # ── forward-fill active counts to non-medication event rows ────────
    for col in [
        TemporalEvent.COUNT_ACTIVE_MAIN_MED.value,
        TemporalEvent.COUNT_ACTIVE_MODIFIER_MED.value,
        TemporalEvent.COUNT_ACTIVE_OTHER_MED.value,
    ]:
        df[col] = (
            df.groupby(Trajectory.ID.value)[col]
            .transform(lambda x: x.replace(0, pd.NA).ffill().fillna(0))
            .astype("float32")
        )

    # ── label remaining uncollapsed medication events ──────────────────
    med_start_mask = df[TemporalEvent.TYPE.value] == EventType.MEDICATION_START.value
    med_stop_mask  = df[TemporalEvent.TYPE.value] == EventType.MEDICATION_STOP.value
    no_action      = df[TemporalEvent.TRIGGER_MED_ACTION_TYPE.value] == ""

    for med_type in ("main", "modifier", "other"):
        type_mask = df[Medication.ANTIDEPRESSANT_TYPE.value] == med_type
        df.loc[med_start_mask & no_action & type_mask,
               TemporalEvent.TRIGGER_MED_ACTION_TYPE.value] = f"{med_type}_added"
        df.loc[med_stop_mask & no_action & type_mask,
               TemporalEvent.TRIGGER_MED_ACTION_TYPE.value] = f"{med_type}_stopped"

    med_event_mask = df[TemporalEvent.TYPE.value].isin([
        EventType.MEDICATION_START.value,
        EventType.MEDICATION_STOP.value,
        EventType.MEDICATION_SWITCH.value,
    ])
    med_indexed = medication_data.set_index(Medication.ID.value)

    df.loc[med_event_mask, TemporalEvent.TRIGGER_MED_ATC_CODE.value] = (
        df.loc[med_event_mask, Medication.ID.value].map(med_indexed[Medication.BASE_ATC_CODE.value])
    )
  
    df.loc[med_event_mask, TemporalEvent.TRIGGER_MED_DRUG_CLASS.value] = (
        df.loc[med_event_mask, Medication.ID.value].map(med_indexed[Medication.DRUG_CLASS.value])
    )
    # ── fill string columns ────────────────────────────────────────────
    for col in [
        TemporalEvent.TRIGGER_MED_ACTION_TYPE.value,
        TemporalEvent.TRIGGER_MED_ATC_CODE.value,
        TemporalEvent.TRIGGER_MED_DRUG_CLASS.value,
        TemporalEvent.PREV_MED_ATC_CODE.value,
        TemporalEvent.PREV_MED_TYPE.value,
        TemporalEvent.PREV_DRUG_CLASS.value,
    ]:
        df[col] = df[col].fillna("UNKNOWN").astype("string")

    n_switches    = (df[TemporalEvent.TYPE.value] == EventType.MEDICATION_SWITCH.value).sum()
    n_dose_change = df[TemporalEvent.TRIGGER_MED_ACTION_TYPE.value].str.contains("dose", na=False).sum()
    print(f"Medication annotation and collapsing complete.")
    print(f"  Dropped redundant STOP rows : {len(stop_indices_to_drop)}")
    print(f"  Switch events               : {n_switches}")
    print(f"  Dose change events          : {n_dose_change}")
    print(f"  Full action_type distribution:")
    print(f"{df[TemporalEvent.TRIGGER_MED_ACTION_TYPE.value].value_counts().to_string()}")

    return df

def build_event_dataframe(text_data, admission_data, medication_data, diagnosis_data, patient_data, trajectory_data, appointment_data):
    """
       Returns a dataframe where each row corresponds to a new event.
       Keeps only important features for modeling:
           - time_since_start, time_since_last_event
           - admission_during_event (float)
           - medication counts (N05_count, N06_count, Other_count)
           - patient features (age, gender)
       Data is ordered by event_date within trajectory group.
       """
    text_cols = [
        e.value
        for e in ClinicalText
        if e != ClinicalText.TEXT
    ]
    text_cols = [c for c in text_cols if c in text_data.columns]

    event_specs = [
        (text_data[text_cols],
         ClinicalText.DATE.value, EventType.NOTE.value),
        (admission_data, Admission.START.value, EventType.ADMISSION_START.value),
        (admission_data, Admission.STOP.value, EventType.ADMISSION_STOP.value),
        (medication_data, Medication.START.value, EventType.MEDICATION_START.value),
        (medication_data, Medication.STOP.value, EventType.MEDICATION_STOP.value),
        (diagnosis_data, Diagnosis.DATE.value, EventType.DIAGNOSIS.value),
        (appointment_data, Appointment.DATE.value, EventType.APPOINTMENT.value),
        (pharmaco_results, PharmacoGenetics.DATE.value,EventType.PHARMACOGENETIC_TEST.value),
        (lab_results, LabResults.DATE.value, EventType.LAB_TEST.value),
    ]

    event_frames = []
    for df, date_col, event_type in event_specs:
        tmp = df.rename(columns={date_col: TemporalEvent.DATE.value}).copy(deep=True)
        tmp[TemporalEvent.TYPE.value] = event_type
        event_frames.append(tmp)

    # Concatenate all events
    df = pd.concat(event_frames, ignore_index=True)
    df = df.merge(trajectory_data, on=[Trajectory.ID.value, Patient.ID.value], how='inner')
    df = df.merge(patient_data, on=Patient.ID.value, how='inner')
    # sort events based on their date within each trajectory.
    df_sorted = df.sort_values([Trajectory.ID.value, TemporalEvent.DATE.value], ignore_index=True)

    # Keep only events that occur within the trajectory period.
    start_dt = pd.to_datetime(df_sorted[Trajectory.START.value])
    stop_dt = pd.to_datetime(df_sorted[Trajectory.STOP.value]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    event_dt = pd.to_datetime(df_sorted[TemporalEvent.DATE.value])

    df_sorted = df_sorted[
        (event_dt >= start_dt) &
        (event_dt <= stop_dt)
        ].reset_index(drop=True)

    df_sorted["_event_idx"] = df_sorted.groupby(Trajectory.ID.value).cumcount()
    df_sorted[TemporalEvent.ID.value] = df_sorted[Trajectory.ID.value].astype(str) + "_" + df_sorted["_event_idx"].astype(str)

    ########################## MEDICATIONS #############################################################################
    print("Enriching medication features...")
    print("Annotating and collapsing medication events...", df_sorted.shape)
    df_sorted = annotate_and_collapse_medication_events(
        df_sorted, medication_data,
        dose_change_window_days=7,
        switch_window_days=14,
    )
    print("Medication features enriched but collapsed.", df_sorted.shape)
    ######################### CLINICAL NOTE ENRICHMENT #################################################################
    print("Enriching clinical note features...")
    df_sorted = enrich_clinical_note_features(df_sorted)
    ######################### APPOINTMENT ENRICHMENT ###################################################################
    print("Enriching appointment features...")
    df_sorted = enrich_appointment_features(df_sorted, appointment_data)
    ####################################################################################################################
    # add ratio features for clinical note and appointment clinician interaction features. 
    print("Adding ratio features...")
    df_sorted = add_ratio_features(df_sorted)  # 
    ########################################## LAB RESULTS #############################################################
    print("Enriching lab result features...")
    df_sorted = enrich_lab_results(df_sorted, lab_results)  # adds lab result features to df_sorted for use in has_recent_abnormal_lab
    ######################### TEMPORAL CONTEXT #########################################################################
    print("Enriching temporal context features...")
    df_sorted = enrich_temporal_context(df_sorted)  # adds time_since_start, time_since_last_event, and event count features
    ####################################################################################################################
    print("Enriching admission features...")
    df_sorted = enrich_admission_features(df_sorted, admission_data)  # adds features related to current and past admissions
    ####################################################################################################################
    print("Enriching active diagnoses features...")
    df_sorted = enrich_active_diagnoses(df_sorted, diagnosis_data)  # adds features related to active diagnoses at time of event, and comorbidity history
    ########################################## DEMOGRAPHICS ############################################################
    print("Enriching demographic features...")
    df_sorted = enrich_demographics(df_sorted)  # adds patient demographic features such as age
    ####################################################################################################################
    print("Enriching historical medication features...")
    df_sorted = enrich_historical_med_changes(df_sorted, medication_data)  # adds features related to past medication changes within trajectory, and outcomes based on future medication changes within trajectory
    ####################################################################################################################
    print("Adding days since last medication event...")
    # add days since the last med_event as a feature
    df_sorted = add_days_since_med_event(df_sorted)
    ######################################################################################
  
    # Keep only important columns
    important_columns = [e.value for e in TemporalEvent]
    df_sorted = df_sorted[important_columns].copy()

    # TODO: for now using only tabular features for experiments..
    # df_sorted = df_sorted[df_sorted[TemporalEvent.TRIGGER_MED_ACTION_TYPE.value].isin([ "main_added", "main_switch","main_stopped",
    # "main_dose_increase", "main_dose_decrease"])]
    # print("filtered", df_sorted.shape)
   
    # for categoical string columns, fill NaN with empty string and convert to string type (or category if high cardinality).
    for col in CATEGORICAL_COLUMNS:
        if col in df_sorted.columns:
            df_sorted[col] = df_sorted[col].fillna("").astype(str)

    # create the other features files- better this way for efficiency and memory management, since the long format files can be quite large and we don't want to keep them all in memory at once. 
    create_list_based_event_feature_files(df_sorted, medication_data, diagnosis_data, trajectory_data)  # creates and saves long-format event-level feature files for active medications and pre-trajectory diagnoses.
    
    print("Constructing medication change outcomes...")
    df_sorted = create_outcomes(df_sorted)  # adds medication change outcome columns based on future medication changes within trajectory
    
    return df_sorted

def create_list_based_event_feature_files(
    df_sorted: pd.DataFrame,
    medication_data: pd.DataFrame,
    diagnosis_data: pd.DataFrame,
    trajectory_data: pd.DataFrame,   # added
) -> None:
    
     ################################# Active medications per event #####################################################
    print("Constructing active medications long format...")
    # save this as a separate file due to memory issue.
    active_meds_long = get_active_meds_long(df_sorted, medication_data)

    active_meds_long.to_parquet(
        data_path / "active_medications.parquet",
        engine="pyarrow",
        compression="zstd",
        index=False
    )
    print("Constructing historical medications long format...")
    historical_meds_long = get_historical_meds_long(df_sorted, medication_data)
    historical_meds_long.to_parquet(
        data_path / "historical_medications.parquet",
        engine="pyarrow",
        compression="zstd",
        index=False
    )
    
    ################# Pre-trajectory diagnoses per event ###############
    print("Constructing pre-event diagnoses history long format...")
  
    historical_main_dx = get_historical_diagnoses_long(df_sorted, diagnosis_data, diag_type="main")
    historical_main_dx.to_parquet(
        data_path / "historical_diagnoses_main.parquet",
        engine="pyarrow", compression="zstd", index=False
    )

    historical_secondary_dx = get_historical_diagnoses_long(df_sorted, diagnosis_data, diag_type="secondary")
    historical_secondary_dx.to_parquet(
        data_path / "historical_diagnoses_secondary.parquet",
        engine="pyarrow", compression="zstd", index=False
    )
    print("Constructing pre-trajectory main diagnoses long format...")
    pre_traj_main_dx = get_pre_trajectory_diagnoses_long(
        df_sorted, diagnosis_data, trajectory_data, diag_type="main"
    )
    pre_traj_main_dx.to_parquet(
        data_path / "pre_trajectory_diagnoses_main.parquet",
        engine="pyarrow", compression="zstd", index=False
    )

    print("Constructing pre-trajectory secondary diagnoses long format...")
    pre_traj_secondary_dx = get_pre_trajectory_diagnoses_long(
        df_sorted, diagnosis_data, trajectory_data, diag_type="secondary"
    )
    pre_traj_secondary_dx.to_parquet(
        data_path / "pre_trajectory_diagnoses_secondary.parquet",
        engine="pyarrow", compression="zstd", index=False
    )


def create_outcomes(
    df: pd.DataFrame,
    window_days: int = 42,
    titration_window_days: int = 28,
) -> pd.DataFrame:
    """
    Reads temporal_data.parquet, adds outcome columns, optionally saves back.
    """
    print(f"  Loaded {len(df)} rows, {df.shape[1]} columns")

    df = add_primary_medication_change_outcomes(
        df,
        window_days=window_days,
        titration_window_days=titration_window_days,
    )

    return df

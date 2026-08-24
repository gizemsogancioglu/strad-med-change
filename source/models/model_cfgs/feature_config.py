# /app/source/models/model_cfgs/feature_config.py

from source.data_processing.data_model import Features, ListEventFeatures, TemporalEvent, Medication, ClinicalText, Appointment

# ── Canonical "missing value" representation, shared by every preprocessing /
#    modeling entry point (sklearn baseline pipeline + LSTM vocab builders).
MISSING_CATEGORY_FILL = "UNKNOWN"   # value used by .fillna() calls
NONE_VALUES = {
    "", " ", "nan", "NaN", "None", "none",
    "UNKNOWN", "unknown", "<NONE>", "<PAD>", None,
}
# ── Categorical columns (for XGBoost OneHotEncoder) ──────────────────────────
CATEGORICAL_COLUMNS = [
    TemporalEvent.TYPE.value,
    TemporalEvent.ACTIVE_DIAGNOSIS_MAIN.value,
    TemporalEvent.ACTIVE_DIAGNOSIS_SECONDARY.value,
    TemporalEvent.PATIENT_GENDER.value,
    TemporalEvent.NOTE_CREATION_EMPLOYEE_ROLE.value,
    TemporalEvent.APPOINTMENT_ROLE.value,
    TemporalEvent.NOTE_MUTATION_EMPLOYEE_ROLE.value,
    TemporalEvent.LAB_RESULTS.value,
    TemporalEvent.NOTE_TYPE.value,
    TemporalEvent.RECENT_LAB_MEASURE.value,
    TemporalEvent.RECENT_LAB_TEST_TYPE.value,
    TemporalEvent.TRIGGER_MED_ATC_CODE.value, 
    TemporalEvent.PREV_MED_ATC_CODE.value,
    TemporalEvent.PREV_DRUG_CLASS.value,
    TemporalEvent.TRIGGER_MED_DRUG_CLASS.value,
    TemporalEvent.TRIGGER_MED_ACTION_TYPE.value,
]
TRIGGER_MED_COLUMNS = [
    #TemporalEvent.TRIGGER_MED_ATC_CODE.value,  # categorical → embeddings
    TemporalEvent.TRIGGER_MED_DRUG_CLASS.value,   # ssri, tca, maoi etc. : one-hot encodings 
    #TemporalEvent.TRIGGER_MED_ACTION_TYPE.value,  # categorical → OneHotEncoder
    #TemporalEvent.PREV_MED_ATC_CODE.value, # for switch action only 
    TemporalEvent.PREV_DRUG_CLASS.value, # for switch action only   
]

ALL_BOOLEAN_COLUMNS = [
    TemporalEvent.DURING_ADMISSION.value,
    TemporalEvent.HAS_RECENT_ABNORMAL_LAB.value,
    TemporalEvent.APPOINTMENT_IS_SAME_EMPLOYEE_AS_LAST.value,
    # med boolean flags
    TemporalEvent.PRIMARY_DOSE_INCREASED.value,
    TemporalEvent.PRIMARY_DOSE_DECREASED.value,
    TemporalEvent.MODIFIER_DOSE_INCREASED.value,
    TemporalEvent.MODIFIER_DOSE_DECREASED.value,
    TemporalEvent.OTHER_DOSE_INCREASED.value,
    TemporalEvent.OTHER_DOSE_DECREASED.value,
    TemporalEvent.PRIMARY_ADDED.value, 
    TemporalEvent.PRIMARY_REMOVED.value, 
    TemporalEvent.MODIFIER_ADDED.value, 
    TemporalEvent.MODIFIER_REMOVED.value, 
    TemporalEvent.OTHER_ADDED.value,
    TemporalEvent.HAS_RECENT_LAB_TEST.value,
]

NON_NUMERIC_COLUMNS = CATEGORICAL_COLUMNS + ALL_BOOLEAN_COLUMNS + [
    Medication.ATC_CODE.value, Medication.DURATION.value, ListEventFeatures.HISTORICAL_MEDICATIONS.value, 
    ListEventFeatures.HISTORICAL_MEDICATIONS_DURATION.value, ListEventFeatures.HISTORICAL_MEDICATIONS_RECENCY.value, 
    ListEventFeatures.HISTORICAL_DIAGNOSIS_MAIN.value,
    ListEventFeatures.HISTORICAL_DIAGNOSIS_SECONDARY.value,]

# =============================================================================
# MODALITY COLUMN DEFINITIONS
# Each modality owns its columns exclusively — no overlaps between modalities.
# Scalers are fitted per modality in PatientTrajectoryDataset.
# =============================================================================

# ── PATIENT modality (static, always present) ─────────────────────────────────
PATIENT_CONTINUOUS_COLUMNS = [
    TemporalEvent.PATIENT_AGE_AT_START.value,
   
]
PATIENT_COLUMNS = [
    TemporalEvent.PATIENT_GENDER.value,
    TemporalEvent.PATIENT_AGE_AT_START.value,
]

# ── TIMESTAMPS modality (always present) ──────────────────────────────────────
TIMESTAMPS_CONTINUOUS_COLUMNS = [
    TemporalEvent.TIME_SINCE_START.value,
    TemporalEvent.TIME_SINCE_LAST_EVENT.value,
]

TIMESTAMP_COLUMNS = TIMESTAMPS_CONTINUOUS_COLUMNS   

# ── ADMISSION modality (always present) ───────────────────────────────────────
ADMISSION_CONTINUOUS_COLUMNS = [
    TemporalEvent.TIME_SINCE_ADMISSION.value,
]
ADMISSION_BINARY_COLUMNS = [
    TemporalEvent.DURING_ADMISSION.value,
]
ADMISSION_COLUMNS = ADMISSION_BINARY_COLUMNS # + ADMISSION_CONTINUOUS_COLUMNS

# ── LAB modality ─────────────────────────────
LAB_BINARY_COLUMNS = [
    TemporalEvent.HAS_RECENT_ABNORMAL_LAB.value,
    TemporalEvent.HAS_RECENT_LAB_TEST.value,        #  boolean
]
LAB_CATEGORICAL_COLUMNS = [
    TemporalEvent.RECENT_LAB_MEASURE.value,         #  categorical
    TemporalEvent.RECENT_LAB_TEST_TYPE.value,
]
LAB_COLUMNS = LAB_BINARY_COLUMNS + LAB_CATEGORICAL_COLUMNS  

# ── DIAGNOSIS modality (always present as active diagnosis) ───────────────────
DIAGNOSIS_EMBEDDING_COLUMNS = [
    TemporalEvent.ACTIVE_DIAGNOSIS_MAIN.value,
    TemporalEvent.ACTIVE_DIAGNOSIS_SECONDARY.value,
]
DIAG_TIME_COLUMNS = [
    TemporalEvent.TIME_SINCE_MAIN_DIAG.value,
    TemporalEvent.TIME_SINCE_SECONDARY_DIAG.value,
]
DIAGNOSIS_COLUMNS = DIAGNOSIS_EMBEDDING_COLUMNS + DIAG_TIME_COLUMNS  # for XGBoost

# ── MED_BOOLEAN_FLAGS modality (medication events only) ───────────────────────
# Binary flags — no scaling needed. Non-zero only at medication events.
MED_BOOLEAN_FLAG_COLUMNS = [
    TemporalEvent.PRIMARY_DOSE_INCREASED.value,
    TemporalEvent.PRIMARY_DOSE_DECREASED.value,
    TemporalEvent.MODIFIER_DOSE_INCREASED.value,
    TemporalEvent.MODIFIER_DOSE_DECREASED.value,
    TemporalEvent.OTHER_DOSE_INCREASED.value,
    TemporalEvent.OTHER_DOSE_DECREASED.value,
]
MED_SUMMARY_COLUMNS = [TemporalEvent.COUNT_ACTIVE_MAIN_MED.value, 
                       TemporalEvent.COUNT_ACTIVE_MODIFIER_MED.value, 
                       TemporalEvent.COUNT_ACTIVE_OTHER_MED.value]

# ── APPOINTMENT modality ────────────────────────────
APPOINTMENT_CONTINUOUS_COLUMNS = [
    TemporalEvent.APPOINTMENT_DURATION.value,
    TemporalEvent.APPOINTMENT_EMPLOYEE_EXPERIENCE.value,
    TemporalEvent.APPOINTMENT_CLINICIAN_FAMILIARITY.value,   
    TemporalEvent.APPOINTMENT_CLINICIAN_DIVERSITY.value, 
]

APPOINTMENT_INTERACTION_COLUMNS = [TemporalEvent.APPOINTMENT_EMPLOYEE_EXPERIENCE.value, 
                                   TemporalEvent.APPOINTMENT_CLINICIAN_FAMILIARITY.value,   
                                   TemporalEvent.APPOINTMENT_CLINICIAN_DIVERSITY.value]

APPOINTMENT_ROLE_COLUMN = TemporalEvent.APPOINTMENT_ROLE.value  # one-hot encoded separately
# APPOINTMENT_COLUMNS = (                                          
#     APPOINTMENT_CONTINUOUS_COLUMNS +
#     [APPOINTMENT_ROLE_COLUMN]
# )
APPOINTMENT_COLUMNS = APPOINTMENT_INTERACTION_COLUMNS   

# ── NOTE modality — all text features fused internally before modality attention
TEXT_METADATA_COLUMNS = [
    TemporalEvent.NOTE_TYPE.value,
    # TemporalEvent.TEXT_TOKEN_NUMBER.value
]

# TEXT_INTERACTION: clinician context
TEXT_INTERACTION_CONTINUOUS_COLUMNS = [
    TemporalEvent.CREATION_EMPLOYEE_EXPERIENCE.value,  
    TemporalEvent.AUTHOR_FAMILIARITY.value,      
    TemporalEvent.AUTHOR_DIVERSITY.value, 
]
TEXT_INTERACTION_ROLE_COLUMN = TemporalEvent.NOTE_CREATION_EMPLOYEE_ROLE.value
TEXT_INTERACTION_COLUMNS = (                           
    [TEXT_INTERACTION_ROLE_COLUMN] +
    TEXT_INTERACTION_CONTINUOUS_COLUMNS
)

TEXT_CONTINUOUS_COLUMNS = TEXT_INTERACTION_CONTINUOUS_COLUMNS  

# ── HISTORICAL modality (XGBoost only, excluded from deep models) ─────────────
HISTORICAL_COLUMNS = [
    TemporalEvent.NUM_PREV_EVENT.value,
    TemporalEvent.NUM_PREV_OTHER_MED_CHANGE.value,
    TemporalEvent.NUM_PREV_MODIFIER_MED_CHANGE.value,
    TemporalEvent.NUM_PREV_PRIMARY_MED_CHANGE.value,
    TemporalEvent.NUM_ADMISSIONS_SO_FAR.value,
]

# ── STATIC patient columns (used in dataset for scaler fitting) ───────────────
STATIC_CONTINUOUS_COLUMNS = PATIENT_CONTINUOUS_COLUMNS

ALL_TIME_COLUMNS = [
    TemporalEvent.TIME_SINCE_START.value,
    TemporalEvent.TIME_SINCE_LAST_EVENT.value,
    TemporalEvent.TIME_SINCE_ADMISSION.value,
    TemporalEvent.TIME_SINCE_MAIN_DIAG.value,
    TemporalEvent.TIME_SINCE_SECONDARY_DIAG.value,
    TemporalEvent.APPOINTMENT_DURATION.value,
    TemporalEvent.CREATION_EMPLOYEE_EXPERIENCE.value,
    TemporalEvent.MUTATION_EMPLOYEE_EXPERIENCE.value,
    TemporalEvent.APPOINTMENT_EMPLOYEE_EXPERIENCE.value,
]


PRIMARY_TRIGGER_ACTIONS = [ "main_added", "main_stopped", "main_switch",
    "main_dose_increase", "main_dose_decrease"]
# =============================================================================
# FEATURE GROUP → COLUMN MAPPING 
# =============================================================================
FEATURE_GROUP_COLUMNS = {
    Features.TIMESTAMPS:        TIMESTAMP_COLUMNS,
    Features.PATIENT:           PATIENT_COLUMNS,
    Features.ADMISSION:         ADMISSION_COLUMNS,
    Features.TEXT_METADATA:     TEXT_METADATA_COLUMNS,
    Features.TEXT_INTERACTION:  TEXT_INTERACTION_COLUMNS,
    Features.APPOINTMENT:       APPOINTMENT_COLUMNS,
  
    Features.DIAGNOSIS:         DIAGNOSIS_COLUMNS,
    Features.PRE_TRAJ_MAIN_DIAGNOSIS   : [ListEventFeatures.PRE_TRAJ_MAIN_DIAGNOSIS.value],  # for sequential models.. 
    Features.PRE_TRAJ_SECONDARY_DIAGNOSIS: [ListEventFeatures.PRE_TRAJ_SECONDARY_DIAGNOSIS.value],  # for sequential models..
  
    Features.MED_BOOLEAN_FLAGS: MED_BOOLEAN_FLAG_COLUMNS,
    Features.TRIGGER_MEDICATIONS:  TRIGGER_MED_COLUMNS,
    Features.MED_SUMMARY: MED_SUMMARY_COLUMNS,
  
    Features.HISTORICAL:        HISTORICAL_COLUMNS,
    Features.LAB_RESULTS:       LAB_COLUMNS,
    Features.EVENT_TYPE:        [TemporalEvent.TRIGGER_MED_ACTION_TYPE.value],
    
}


def active(columns: list, df) -> list:
    """Filter to columns present in the dataframe."""
    return [c for c in columns if c in df.columns]

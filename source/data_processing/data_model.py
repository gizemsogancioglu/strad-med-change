from enum import Enum


class Dataset(Enum):
    TRAJECTORY = 'trajectories.csv'
    TEXT = 'clinical_texts.csv'
    ADMISSION = 'admissions.csv'
    MEDICATION = 'medications.csv'
    PATIENT = 'patients.csv'
    DIAGNOSIS = 'diagnoses.csv'
    EMPLOYEE = 'employees.csv'
    APPOINTMENT = 'appointments.csv'
    LAB = 'lab_med.csv'
    PHARMACOGENETICS = 'pharmacogenetics.csv'
    CURRENT_DATE = 'currentdate.csv'

class Patient(Enum):
    ID = 'patient_id'
    GENDER = 'patient_gender'
    BIRTH_DATE = 'patient_date_of_birth'
    POST_CODE = 'patient_postcode'


class Trajectory(Enum):
    ID = 'trajectory_id'
    PATIENT_ID = Patient.ID.value
    START = "trajectory_start_date"
    STOP = "trajectory_end_date"
    IS_CLOSED = 'trajectory_is_closed'
    COMPANY = 'trajectory_caregiver_company'
    # created later...
    DURATION = 'trajectory_duration'
    PATIENT_AGE_AT_START = 'patient_age_at_trajectory_start'


class Employee(Enum):
    ID = 'employee_id'
    ROLE = 'employee_role'  # nurse or clinician
    START = 'employee_start_date'  # start date at organization


class ClinicalText(Enum):
    NOTE_ID = 'text_id'
    PATIENT_ID = Patient.ID.value
    DATE = 'text_creation_date'
    DATE_LAST = 'text_mutation_date'
    TEXT = 'text'
    CREATION_PRACTITIONER_CODE = 'creation_employee_id'
    MUTATION_PRACTITIONER_CODE = 'mutation_employee_id'
    # created later...
    TRAJECTORY_ID = Trajectory.ID.value
    CREATION_EMPLOYEE_START = "creation_employee_start"
    CREATION_EMPLOYEE_ROLE = 'creation_employee_role'
    MUTATION_EMPLOYEE_START = "mutation_employee_start"
    MUTATION_EMPLOYEE_ROLE = 'mutation_employee_role'

    # metadata
    NOTE_TYPE = 'note_type'  # intake, progress, discharge, etc.
    TOKEN_NUMBER = 'token_number'  # number of tokens in the note



class Diagnosis(Enum):
    PATIENT_ID = Patient.ID.value
    DATE = 'diagnosis_date'  # date of diagnosis
    DSM5_CODE = 'diagnosis_DSM5'  # DSM5 code of diagnosis
    DSM4_CODE = 'diagnosis_DSM4'
    TYPE = 'diagnosis_type'  # main or secondary
    # created later...
    TRAJECTORY_ID = Trajectory.ID.value  # linked to trajectories if valid in the time of trajectory
    DESCRIPTION = 'description'
    DSM_CODE = 'diagnosis_DSM_code'


class Medication(Enum):
    # unique ID for this data.
    ID = 'medication_id'
    PATIENT_ID = Patient.ID.value
    ATC_CODE = 'ATC_code'
    NAME = 'medication_name'
    MED_TYPE = 'medication_type'  # op afspraak (prescribed) or clinic (administered in clinic)
    DAY_DOSES = 'day_dosage_amount'
    DOSE_UNIT = 'day_dosage_unit'
    START = 'medication_start_date'
    STOP = 'medication_end_date'
    IS_STOPPED = 'medication_is_stopped'  # True or False
    # created later...
    DURATION = 'medication_duration'
    TRAJECTORY_ID = Trajectory.ID.value
    ANTIDEPRESSANT_TYPE = 'is_antidepressant'  # either main or modifier medication or other
    BASE_ATC_CODE = 'base_atc_code'
    DRUG_CLASS    = 'drug_class'
    # using ATC code of patient, we represent it in 5 levels.
    # TODO: Implement medication ATC code levels if needed in the future.
    # MED_L1 = "med_l1" # Anatomical main group e.g. N
    # MED_L2 = "med_l2" # Therapeutic subgroup e.g. N06
    # MED_L3 = "med_l3" # Pharmacological subgroup e.g. N06A
    # MED_L4 = "med_l4" # Chemical subgroup e.g. N06AB
    # MED_L5 = "med_l5" # Chemical substance e.g. N06AB04


class Admission(Enum):
    ID = 'admission_id'  # unique identifier
    PATIENT_ID = Patient.ID.value
    START = 'admission_start_date'
    STOP = 'admission_end_date'
    IS_CLOSED = 'admission_is_closed'  # True or False
    COMPANY = 'admission_caregiver_company'
    # created later...
    DURATION = 'admission_duration'
    TRAJECTORY_ID = Trajectory.ID.value


class Appointment(Enum):
    ID = 'appointment_id'
    RECORD_ID = 'appointment_record_id'
    PATIENT_ID = Patient.ID.value
    DATE = 'appointment_date'
    START = 'appointment_start_time'
    STOP = 'appointment_end_time'
    INDIRECT_TIME = 'indirect_time'
    EMPLOYEE_ID = Employee.ID.value
    # created later...
    DURATION = 'appointment_duration'
    TRAJECTORY_ID = Trajectory.ID.value
    EMPLOYEE_START = Employee.START.value
    EMPLOYEE_ROLE = Employee.ROLE.value

class CurrentDate(Enum):
    CURRENT_TIME = 'current_time'

class LabResults(Enum):
    PATIENT_ID = Patient.ID.value
    DATE = 'lab_date'
    TEST = 'test'
    VALUE = 'value'
    COMMENT = 'comment'
    RANGE_LOW = 'reference_low'
    RANGE_HIGH = 'reference_high'
    # created later...
    TRAJECTORY_ID = Trajectory.ID.value
    IN_RANGE = 'in_range'  # low, high, in_range, unknown


class PharmacoGenetics(Enum):
    PATIENT_ID = Patient.ID.value
    DATE = 'lab_date'
    TEST = 'test'
    VALUE = 'value'
    COMMENT = 'comment'
    # created later...
    TRAJECTORY_ID = Trajectory.ID.value
    PHENOTYPE = 'phenotype'


class EventType(Enum):
    NOTE = 'note'
    ADMISSION_START = 'admission_start'
    ADMISSION_STOP = "admission_end"
    MEDICATION_START = 'medication_start'
    MEDICATION_STOP = "medication_end"
    MEDICATION_SWITCH = 'medication_switch'
    DIAGNOSIS = 'diagnosis'
    APPOINTMENT = 'appointment'
    PHARMACOGENETIC_TEST = 'pharmacogenetic_test'
    LAB_TEST = 'lab_test'


class Features(str, Enum):
    TIMESTAMPS = 'timestamps'
    EVENT_TYPE = 'event_type'
    PATIENT = 'patient'

    ADMISSION = 'admission'
    
    DIAGNOSIS = 'diagnosis'
    PAST_DIAGNOSIS = 'past_diagnosis'
    PRE_TRAJ_SECONDARY_DIAGNOSIS = 'pre_traj_secondary_diagnosis'
    PRE_TRAJ_MAIN_DIAGNOSIS = 'pre_traj_main_diagnosis'
    
    HISTORICAL = 'historical'
    
    APPOINTMENT = 'appointment'
    
    # medication features.
    PHARMACOGENETICS = 'pharmacogenetics'
    LAB_RESULTS = 'lab_results'

    ACTIVE_MEDICATIONS = 'active_medications'
    PAST_MEDICATIONS = 'past_medications'
    
    MED_SUMMARY = 'med_summary'
    MED_BOOLEAN_FLAGS = 'med_boolean_flags'
    TRIGGER_MEDICATIONS = 'trigger_medications' # only for event-level prediction, not trajectory-level.

    # text features.
    BERT = 'bert'
    TFIDF = 'tfidf'
    TEXT_METADATA = 'text_metadata'
    TEXT_INTERACTION = 'text_interaction'
    # symptom prediction features - extracted through llm.
    SYMPTOM_PREDICTIONS = 'symptom_predictions'

class TemporalEvent(Enum):
    # ID and patient demographics
    ID = '_event_id'
    TRAJECTORY_ID = Trajectory.ID.value  # trajectory id that event belongs to.
    PATIENT_ID = Patient.ID.value
    PATIENT_GENDER = Patient.GENDER.value # female, male, other [available for all patients]
    PATIENT_AGE_AT_START = Trajectory.PATIENT_AGE_AT_START.value
    PATIENT_AGE = "patient_age"
    #PATIENT_SES = 'patient_ses_value'

    # event and temporal information
    TYPE = 'event_type'  # type of an event.
    DATE = 'event_date'  # date of an event
    TIME_SINCE_START = 'time_since_start'  # time passed since the trajectory start
    TIME_SINCE_LAST_EVENT = 'time_since_last_event'  # time passed since the previous event
    TRAJECTORY_PHASE = 'trajectory_phase'  # early, middle, late - how to define? 

    # boolean event-level medication information
    TRIGGER_MED_ATC_CODE = 'trigger_med_atc_code'      # ATC code of the specific med that triggered this event
    #TRIGGER_MED_TYPE = 'trigger_med_type'               # primary / modifier / other of the triggered med
    TRIGGER_MED_DRUG_CLASS = 'trigger_med_drug_class' # drug class of the triggered med, e.g. SSRI, SNRI, TCA, etc.
    TRIGGER_MED_ACTION_TYPE = 'trigger_action_type' # addition or removal of the triggered med.
    PREV_MED_ATC_CODE  = 'prev_med_atc_code'
    PREV_MED_TYPE = 'prev_med_type' 
    PREV_DRUG_CLASS    = 'prev_drug_class'

    PRIMARY_ADDED = 'main_added'
    MODIFIER_ADDED = 'modifier_added'
    PRIMARY_REMOVED = 'main_removed'
    MODIFIER_REMOVED = 'modifier_removed'
    OTHER_ADDED = 'other_added'
    OTHER_REMOVED = 'other_removed'
    OTHER_DOSE_INCREASED = 'other_dose_increased'
    OTHER_DOSE_DECREASED = 'other_dose_decreased'
    PRIMARY_DOSE_INCREASED = 'main_dose_increased'
    MODIFIER_DOSE_INCREASED = 'modifier_dose_increased'
    PRIMARY_DOSE_DECREASED = 'main_dose_decreased'
    MODIFIER_DOSE_DECREASED = 'modifier_dose_decreased'
    # event-level medication summary features 
    COUNT_ACTIVE_MAIN_MED = 'count_main_med'
    COUNT_ACTIVE_MODIFIER_MED = 'count_modifier_med'
    COUNT_ACTIVE_OTHER_MED = 'count_other_med'
    DAYS_SINCE_MED_EVENT = 'days_since_med_event'

    # admission information
    DURING_ADMISSION = 'during_admission'  # a flag whether event happened during admission
    TIME_SINCE_ADMISSION = 'time_since_admission'
    NUM_ADMISSIONS_SO_FAR = 'num_admissions_so_far'

    # clinical text information
    TEXT_ID = ClinicalText.NOTE_ID.value  # only available for note events.
    NOTE_CREATION_EMPLOYEE_ROLE = ClinicalText.CREATION_EMPLOYEE_ROLE.value
    NOTE_MUTATION_EMPLOYEE_ROLE = ClinicalText.MUTATION_EMPLOYEE_ROLE.value
    CREATION_EMPLOYEE_EXPERIENCE = "creation_employee_experience"
    MUTATION_EMPLOYEE_EXPERIENCE = "mutation_employee_experience"
    TEXT_TOKEN_NUMBER = ClinicalText.TOKEN_NUMBER.value
    NOTE_TYPE = ClinicalText.NOTE_TYPE.value # categorical variable
    AUTHOR_NOTE_COUNT_SO_FAR = "author_note_count_so_far"  # number of notes created by the same author up to this note in the trajectory
    NUM_UNIQUE_AUTHORS_SO_FAR = "num_unique_authors_so_far"  # number of unique authors up to this note in the trajectory
    TIME_OF_DAY_BUCKET = "time_of_day_bucket"  # morning, afternoon, evening, night
    AUTHOR_FAMILIARITY = "author_familiarity"  # number of previous notes by the same author divided by total number of previous notes in the trajectory.
    AUTHOR_DIVERSITY = "author_diversity"  # number of unique authors divided by total number of previous notes in the trajectory.

    # diagnosis
    ACTIVE_DIAGNOSIS_MAIN = 'active_diagnosis_main'  # DSM-5 code later hierarchically coded for feature representation
    TIME_SINCE_MAIN_DIAG = 'time_since_main_diag'  # Time elapsed since main diagnosis up to the current event
    ACTIVE_DIAGNOSIS_SECONDARY = 'active_diagnosis_secondary'  # DSM-5 code later hierarchically coded for feature representation
    TIME_SINCE_SECONDARY_DIAG = 'time_since_secondary_diag'  # Time elapsed since secondary diagnosis up to the current event
    HAS_COMORBID_DIAGNOSIS = 'has_comorbid_diagnosis'  # flag indicating presence of both main and secondary diagnoses at the time of event
    HAS_HISTORY_OF_COMORBID_DIAGNOSIS = 'has_history_of_comorbid_diagnosis'  # flag indicating presence of comorbidity at any prior event in the trajectory
     
    # appointment
    APPOINTMENT_DURATION = Appointment.DURATION.value
    APPOINTMENT_ROLE = Appointment.EMPLOYEE_ROLE.value
    APPOINTMENT_EMPLOYEE_EXPERIENCE = "appointment_employee_experience"
    INDIRECT_TO_DURATION_RATIO = 'indirect_to_duration_ratio' # complexity signal for the case
    APPOINTMENT_TIME_OF_DAY_BUCKET = 'appointment_time_of_day_bucket' 
    APPOINTMENT_EMPLOYEE_COUNT_SO_FAR = 'appointment_employee_count_so_far' # employee familiarity for the given patient trajectory. 
    APPOINTMENT_NUM_UNIQUE_EMPLOYEES_SO_FAR = 'appointment_num_unique_employees_so_far' # proxy for complexity of the case, or change in the care team due to other reasons.
    APPOINTMENT_IS_SAME_EMPLOYEE_AS_LAST = 'appointment_is_same_employee_as_last'
    APPOINTMENT_CLINICIAN_FAMILIARITY = 'appointment_clinician_familiarity' # number of previous appointments with the same clinician divided by total number of previous appointments in the trajectory.
    APPOINTMENT_CLINICIAN_DIVERSITY = 'appointment_clinician_diversity' # number of unique clinicians divided by total number of previous appointments in the trajectory.
    
    #lab and pharmacogenetics
    LAB_RESULTS = LabResults.IN_RANGE.value
    HAS_RECENT_ABNORMAL_LAB = 'has_recent_abnormal_lab'  # flag indicating whether there has been an abnormal lab result in the recent past (e.g. 6 weeks)
    RECENT_LAB_MEASURE = "recent_lab_measure"
    HAS_RECENT_LAB_TEST = "has_recent_lab_test"
    RECENT_LAB_TEST_TYPE = "recent_lab_test_type"
    PHARMACOGENETICS_RESULTS = PharmacoGenetics.PHENOTYPE.value

    # additional history features for the non-sequential models. 
    NUM_PREV_EVENT = 'num_prev_event'
    NUM_PREV_PRIMARY_MED_CHANGE = 'num_prev_primary_med_change'
    NUM_PREV_MODIFIER_MED_CHANGE = 'num_prev_modifier_med_change'
    NUM_PREV_OTHER_MED_CHANGE = 'num_prev_other_med_change'
   
class ListEventFeatures(Enum):
    ACTIVE_MEDICATIONS = 'active_medications' # list of active medications at the time of event, with start timestamp for each medication.
    HISTORICAL_MEDICATIONS = 'historical_medications' # list of all medications that have been active in the past until this event, with start and stop timestamps for each medication.    
    HISTORICAL_MEDICATIONS_DURATION = 'historical_medications_duration' # list of durations of all medications that have been active in the past until this event.    
    HISTORICAL_MEDICATIONS_RECENCY = 'historical_medications_recency' #
   
    HISTORICAL_DIAGNOSIS_MAIN = 'historical_diagnosis_main'  # list of all main diagnoses that have been active in the past until this event, with start and stop timestamps for each diagnosis.
    HISTORICAL_DIAGNOSIS_MAIN_RECENCY = 'historical_diagnosis_main_recency' # recency of historical main diagnoses.
    HISTORICAL_DIAGNOSIS_MAIN_DURATION = 'historical_diagnosis_main_duration' # recency and duration of historical main diagnoses.
    
    HISTORICAL_DIAGNOSIS_SECONDARY = 'historical_diagnosis_secondary'  # list of all secondary diagnoses that have been active in the past until this event, with start and stop timestamps for each diagnosis.
    HISTORICAL_DIAGNOSIS_SECONDARY_RECENCY = 'historical_diagnosis_secondary_recency'
    HISTORICAL_DIAGNOSIS_SECONDARY_DURATION = 'historical_diagnosis_secondary_duration'

    PRE_TRAJ_MAIN_DIAGNOSIS      = "pre_traj_main_diagnosis" # list of all pre-trajectory diagnoses, not active now
    PRE_TRAJ_SECONDARY_DIAGNOSIS = "pre_traj_secondary_diagnosis" # list of all pre-trajectory diagnoses, not active now
   
class TrajectoryLevelOutcomes(Enum):
    TRAJECTORY_ID = Trajectory.ID.value
    MEDICATION_CHANGES = 'medication_changes'

class EventLevelOutcomes(Enum):
    EVENT_ID = TemporalEvent.ID.value
    ALL_MED_CHANGE_OUTCOME = 'all_med_change'
    ALL_MED_ADDITION_OUTCOME = 'all_med_addition'
    PRIMARY_MED_ADDITION_OUTCOME = 'primary_med_addition'
    PRIMARY_MED_REMOVAL_OUTCOME = 'primary_med_removal'
    
    PRIMARY_MED_CHANGE_OUTCOME = 'primary_med_change'
    PRIMARY_MED_CHANGE_TYPE_OUTCOME = 'primary_med_change_type' # addition, removal, switch, dose increase, dose decrease, or no change.
    PRIMARY_MED_DRUG_CHANGE_OUTCOME = 'primary_med_drug_change'
    PRIMARY_MED_DRUG_CHANGE_TYPE_OUTCOME = 'primary_med_drug_change_type' # switch to different drug class vs. switch within the same drug class vs. no drug change.
import os
import time
import pandas as pd
import yaml
from source.data_processing.data_model import *
from source.data_processing.data_reader import PROJECT_ROOT, trajectories, patients, admissions, appointments, \
    medications, diagnoses, clinical_texts, data_path, results_path, model_path
from source.models.base_model import baseline_training, persistence_baseline, set_seed, DatasetSplitter, \
    get_events, prepare_columns_to_preprocess, majority_baseline
from source.data_processing.event_level_data_preparation import build_event_dataframe
from source.models.embeddings import read_bert_features, read_tfidf_features
from source.models.medication_change_prediction import MedicationChangePrediction
from source.models.model_cfgs.feature_config import PRIMARY_TRIGGER_ACTIONS
import numpy as np
import joblib
from source.models.model_cfgs.feature_ablation_configs import (
    FULL_CONFIG, trigger_configs,
)
from source.models.symptom_progression_prediction import INT_TO_LABEL, PREDICTED_SYMPTOM_CATS, load_symptom_predictions, merge_symptom_predictions
from source.models.temporal_model import train_temporal_model, train_backbone_fixed_arch, train_temporal_model_multiseed

import pandas as pd
import numpy as np

SEEDS = (0, 13, 42, 123)

def load_filtered_active_medications(active_med_path):
    """
    Reads active_medications.parquet and applies, at read time:
        - Unique ATC codes per event (longest-duration row kept per code).
        - Ordered by medication start date, most recent first.
        - Top 10 most recent active medications kept per event, across
          ALL types (main + modifier + other) combined.
    Returns
    -------
    dict with:
        meds_dict : event_id -> list[str]   (ATC codes, recency-ordered)
        dur_dict  : event_id -> list[float] (durations, aligned with meds_dict)
        type_dict : event_id -> list[str]   ('main'/'modifier'/'other', aligned)
    """
    active_meds = pd.read_parquet(active_med_path)
    print(f"Loaded {active_med_path}. Shape: {active_meds.shape}")

    required_cols = {
        TemporalEvent.ID.value,
        Medication.BASE_ATC_CODE.value,
        Medication.DURATION.value,
        Medication.ANTIDEPRESSANT_TYPE.value,
        Medication.START.value,
    }
    missing = required_cols - set(active_meds.columns)
    if missing:
        raise ValueError(
            f"active_medications parquet is missing required columns: {missing}."
        )

    active_meds[Medication.START.value] = pd.to_datetime(active_meds[Medication.START.value])

    # ── Unique ATC per event (longest-duration row kept per code) ──────────
    n_before = len(active_meds)
    active_meds = (
        active_meds
        .sort_values(Medication.DURATION.value, ascending=False)
        .drop_duplicates(subset=[TemporalEvent.ID.value, Medication.BASE_ATC_CODE.value], keep='first')
    )
    n_after_dedup = len(active_meds)
    print(f"  Unique ATC per event: {n_before} -> {n_after_dedup} rows "
          f"({n_before - n_after_dedup} duplicate rows dropped)")

    # ── Per-event count before any capping ──────────────────────────────────
    active_meds["_n_total_available"] = (
        active_meds.groupby(TemporalEvent.ID.value)[Medication.BASE_ATC_CODE.value].transform("count")
    )
    max_total_pre_filter = active_meds.groupby(TemporalEvent.ID.value)["_n_total_available"].first().max()
    n_events_total = active_meds[TemporalEvent.ID.value].nunique()
    n_events_capped = (active_meds.groupby(TemporalEvent.ID.value)["_n_total_available"].first() > 10).sum()

    # ── Order by start date, most recent first, within each event ──────────
    active_meds = active_meds.sort_values(
        [TemporalEvent.ID.value, Medication.START.value],
        ascending=[True, False]
    )

    # combined recency rank across ALL medications (main + modifier + other)
    active_meds["_combined_rank"] = active_meds.groupby(TemporalEvent.ID.value).cumcount()

    is_main = active_meds[Medication.ANTIDEPRESSANT_TYPE.value] == "main"

    # recency rank among main meds only, most recent first
    active_meds["_main_rank"] = active_meds[is_main].groupby(TemporalEvent.ID.value).cumcount()

    keep_mask = (active_meds["_combined_rank"] < 10) 
   
    active_meds = active_meds[keep_mask]

    # ── Max meds per event AFTER filtering ──────────────────────────────────
    max_total_post_filter = (
        active_meds.groupby(TemporalEvent.ID.value)[Medication.BASE_ATC_CODE.value].count().max()
        if len(active_meds) else 0
    )

    print(f"  Events affected by the top-10 cap: {n_events_capped} / {n_events_total}")
    print(f"  Max total active meds per event, pre-filter: {max_total_pre_filter}")
    print(f"  Max total active meds per event, post-filter: {max_total_post_filter}")
   
    meds_dict = {}
    dur_dict = {}
    type_dict = {}
    for key, grp in active_meds.groupby(TemporalEvent.ID.value, sort=False):
        meds_dict[key] = grp[Medication.BASE_ATC_CODE.value].astype(str).tolist()
        dur_dict[key]  = grp[Medication.DURATION.value].tolist()
        type_dict[key] = grp[Medication.ANTIDEPRESSANT_TYPE.value].astype(str).tolist()

    n_events = active_meds[TemporalEvent.ID.value].nunique()
    n_main   = (active_meds[Medication.ANTIDEPRESSANT_TYPE.value] == "main").sum()
    n_other  = len(active_meds) - n_main
    print(f"Filtered active medications loaded for {n_events} events "
          f"({n_main} main rows, {n_other} modifier/other rows).")

    return {
        "meds_dict": meds_dict,
        "dur_dict": dur_dict,
        "type_dict": type_dict,
    }

def load_temporal_data(input_config):
    temp_parquet_path = os.path.join(data_path, "temporal_data_med_only.parquet")
    active_med_path = os.path.join(data_path, "active_medications.parquet")
    historical_med_path = os.path.join(data_path, "historical_medications.parquet")
    historical_dx_main_path = os.path.join(data_path, "historical_diagnoses_main.parquet")
    historical_dx_secondary_path = os.path.join(data_path, "historical_diagnoses_secondary.parquet")

    is_active_med = Features.ACTIVE_MEDICATIONS in input_config
    is_past_med = Features.PAST_MEDICATIONS in input_config
    is_past_dx = Features.PAST_DIAGNOSIS in input_config

    if (not os.path.exists(temp_parquet_path)) or \
            (is_active_med and not os.path.exists(active_med_path)) or \
            (is_past_med and not os.path.exists(historical_med_path)):
        print(f"File not found!: {temp_parquet_path} or {active_med_path} or {historical_med_path}")
        data = build_event_dataframe(clinical_texts, admissions, medications, diagnoses, patients,
                                     trajectories, appointments)
        data.to_parquet(temp_parquet_path, engine='pyarrow')
    else:
        print(f"File exists!: {temp_parquet_path}")

    print("Loading event-level dataset for preprocessing....")

    # ------------------------
    # Load main data
    # ------------------------
    data = pd.read_parquet(temp_parquet_path)
    print(f"Shape: {data.shape}")

    # Filter out notes written by nurses
    note_mask = data[TemporalEvent.TYPE.value] == EventType.NOTE.value
    nurse_notes_mask = note_mask & (data[ClinicalText.CREATION_EMPLOYEE_ROLE.value] == 'nurse')
    data = data[~nurse_notes_mask].reset_index(drop=True)

    # ------------------------
    # Load active medications
    # ------------------------
    if is_active_med:
        filtered = load_filtered_active_medications(active_med_path)
        meds_dict, dur_dict = filtered["meds_dict"], filtered["dur_dict"]

        data[Medication.BASE_ATC_CODE.value] = data[TemporalEvent.ID.value].map(
            lambda k: meds_dict.get(k, []))
        data[Medication.DURATION.value] = data[TemporalEvent.ID.value].map(
            lambda k: dur_dict.get(k, []))
        print(f"processed active medications data.")

    # ------------------------
    # Load historical (past, non-active) medications
    # ------------------------
    print("is past med:", is_past_med)
    if is_past_med:
        historical_meds = pd.read_parquet(historical_med_path)
        print(f"Loaded historical_medications.parquet. Shape: {historical_meds.shape}")

        past_meds_dict = {}
        past_dur_dict = {}
        past_recency_dict = {}
        for _, row in historical_meds.iterrows():
            key = row[TemporalEvent.ID.value]
            past_meds_dict.setdefault(key, []).append(row[Medication.BASE_ATC_CODE.value])
            past_dur_dict.setdefault(key, []).append(row.get(Medication.DURATION.value, 1.0))
            past_recency_dict.setdefault(key, []).append(row.get("time_since_stopped", 0.0))

        past_meds_list, past_dur_list, past_recency_list = [], [], []
        for _, row in data.iterrows():
            key = row[TemporalEvent.ID.value]
            past_meds_list.append(past_meds_dict.get(key, []))
            past_dur_list.append(past_dur_dict.get(key, []))
            past_recency_list.append(past_recency_dict.get(key, []))

        data[ListEventFeatures.HISTORICAL_MEDICATIONS.value] = past_meds_list
        data[ListEventFeatures.HISTORICAL_MEDICATIONS_DURATION.value] = past_dur_list
        data[ListEventFeatures.HISTORICAL_MEDICATIONS_RECENCY.value] = past_recency_list
        print(f"processed past medications data.")

    # ------------------------
    # Load historical main diagnoses
    # ------------------------
    if is_past_dx:
        hist_main_dx = pd.read_parquet(historical_dx_main_path)
        print(f"Loaded historical_diagnoses_main.parquet. Shape: {hist_main_dx.shape}")
        hist_sec_dx = pd.read_parquet(historical_dx_secondary_path)
        print(f"Loaded historical_diagnoses_secondary.parquet. Shape: {hist_sec_dx.shape}")

        grp = hist_main_dx.groupby(TemporalEvent.ID.value)
        past_main_dx_dict = grp[Diagnosis.DSM_CODE.value].apply(list).to_dict()
        past_main_dx_recency = grp["time_since_diagnosis"].apply(list).to_dict()
        past_main_dx_duration = grp["diagnosis_duration"].apply(list).to_dict()

        data[ListEventFeatures.HISTORICAL_DIAGNOSIS_MAIN.value] = data[TemporalEvent.ID.value].map(
            lambda k: past_main_dx_dict.get(k, []))
        data[ListEventFeatures.HISTORICAL_DIAGNOSIS_MAIN_RECENCY.value] = data[TemporalEvent.ID.value].map(
            lambda k: past_main_dx_recency.get(k, []))
        data[ListEventFeatures.HISTORICAL_DIAGNOSIS_MAIN_DURATION.value] = data[TemporalEvent.ID.value].map(
            lambda k: past_main_dx_duration.get(k, []))

        # ------------------------
        # Load historical secondary diagnoses
        # ------------------------

        grp = hist_sec_dx.groupby(TemporalEvent.ID.value)
        past_sec_dx_dict = grp[Diagnosis.DSM_CODE.value].apply(list).to_dict()
        past_sec_dx_recency = grp["time_since_diagnosis"].apply(list).to_dict()
        past_sec_dx_duration = grp["diagnosis_duration"].apply(list).to_dict()

        data[ListEventFeatures.HISTORICAL_DIAGNOSIS_SECONDARY.value] = data[TemporalEvent.ID.value].map(
            lambda k: past_sec_dx_dict.get(k, []))
        data[ListEventFeatures.HISTORICAL_DIAGNOSIS_SECONDARY_RECENCY.value] = data[TemporalEvent.ID.value].map(
            lambda k: past_sec_dx_recency.get(k, []))
        data[ListEventFeatures.HISTORICAL_DIAGNOSIS_SECONDARY_DURATION.value] = data[TemporalEvent.ID.value].map(
            lambda k: past_sec_dx_duration.get(k, []))

    print(f"Prepared {len(data)} rows with medications as lists and durations aligned.")

    return data


def save_or_append_features(new_df, path, filename):
    file_path = path / filename

    if file_path.exists():
        # Read existing features and concatenate
        existing_df = pd.read_csv(file_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        print(f"File exists. Appended {len(new_df)} rows. Total rows: {len(combined_df)}")
    else:
        combined_df = new_df
        print(f"File not found. Saving {len(new_df)} rows.")

    # Save updated features
    print(f"Results are saved into {file_path}")
    combined_df.to_csv(file_path, index=False)
    return combined_df

def filter_events(data_full: pd.DataFrame) -> pd.DataFrame:
    valid_events = [e.value for e in EventType]

    excluded_events =     [EventType.PHARMACOGENETIC_TEST.value,
                           EventType.ADMISSION_START.value,
                           EventType.ADMISSION_STOP.value, 
                           EventType.LAB_TEST.value,
                           EventType.DIAGNOSIS.value,
                           EventType.APPOINTMENT.value,
                           EventType.NOTE.value
                           ]
    valid_events = [e for e in valid_events if e not in excluded_events]
    data_full = data_full[data_full[TemporalEvent.TYPE.value].isin(valid_events)].reset_index(drop=True)
    return data_full

def merge_text_features(data, input_config, finetune_bert, bert_finetuned=False):
    features = pd.DataFrame()
    columns = []

    if Features.BERT in input_config and not finetune_bert:
        features = read_bert_features(finetuned=bert_finetuned)
        columns = [col for col in features.columns if col.startswith("bert_feature")]
        print("bert shape: ", features.shape)

    elif Features.TFIDF in input_config:
        features = read_tfidf_features(dim=500)
        columns = [col for col in features.columns if col.startswith("tfidf_feature")]
        print("tfidf shape: ", features.shape)

    if not features.empty:
        note_ids = data.loc[data[TemporalEvent.TYPE.value] == EventType.NOTE.value, ClinicalText.NOTE_ID.value].unique()
        features = features[features[ClinicalText.NOTE_ID.value].isin(note_ids)]
        features[columns] = features[columns].astype('float32')
        features = features.drop_duplicates(subset=ClinicalText.NOTE_ID.value)

        text_map_df = features.set_index(ClinicalText.NOTE_ID.value)[columns]

        data = data.join(text_map_df, on=ClinicalText.NOTE_ID.value)

        # ensure data is sorted by trajectory and date before forward-filling
        traj_col = Trajectory.ID.value
        date_col = TemporalEvent.DATE.value
        data = data.sort_values([traj_col, date_col])

        # forward-fill text embeddings from most recent note within each trajectory
        # non-note rows get NaN from the join — ffill propagates the last note's embeddings
        # note rows with no matching embedding (e.g. missing note) also get ffilled
        data[columns] = (
            data.groupby(traj_col, sort=False)[columns]
            .transform(lambda x: x.ffill())
            .astype('float32')
        )

        # any remaining NaN means no prior note exists in trajectory — fill with zeros
        data[columns] = data[columns].fillna(0).astype('float32')

        non_zero_counts = (data[columns] != 0).sum(axis=0)
        print(f"Non-zero text feature counts (min/max): {non_zero_counts.min()} / {non_zero_counts.max()}")

    if Features.BERT in input_config and finetune_bert:
        print("raw text is added to the event-level dataframe")
        data = data.merge(clinical_texts[[ClinicalText.TEXT.value, ClinicalText.NOTE_ID.value]],
                          on=ClinicalText.NOTE_ID.value, how='left')

    return data


def load_textual_features(data_full, FULL_CONFIG, finetune_bert=False, bert_finetuned=False):
    data_full = merge_text_features(data_full, FULL_CONFIG,
                                        finetune_bert=False, bert_finetuned=False)
     # Symptom predictions — load and merge once
    symptom_predictions_df = load_symptom_predictions(
            PROJECT_ROOT / "Qwen3-30B-A3B_remapped.jsonl")
    data_full = merge_symptom_predictions(data_full, symptom_predictions_df)
    #print example values for symptom categories
    symptom_cols = [f"symptom_pred_cat_{cat}" for cat in PREDICTED_SYMPTOM_CATS]
    print("\nExample symptom category values (first 5 rows with non-no_mention signal):")
    mask = (data_full[symptom_cols] != "no_mention").any(axis=1)
    print(data_full.loc[mask, [TemporalEvent.TYPE.value] + symptom_cols].head(5).to_string())
    print("\nValue counts per symptom column:")
    for col in symptom_cols:
         print(f"\n{col}:\n{data_full[col].value_counts()}")

    data_full = forward_fill_symptom_features(
        data_full, PREDICTED_SYMPTOM_CATS)
        
    data_full = forward_fill_note_features(data_full)
    print(f"Note metadata features forward-filled — shape: {data_full.shape}")
    
    return data_full

def forward_fill_symptom_features(data: pd.DataFrame, symptom_cats: list) -> pd.DataFrame:
    """
    Forward-fill symptom_pred_cat_* within each trajectory.
    For every event (note or non-note), if a symptom category is 'no_mention',
    it inherits the most recent meaningful signal (worsened/improved/established)
    from any prior event in the same trajectory.
    If no prior meaningful signal exists, the value remains 'no_mention'.
    """
    date_col = TemporalEvent.DATE.value
    traj_col = Trajectory.ID.value
    symptom_cols = [f"symptom_pred_cat_{cat}" for cat in symptom_cats]
    symptom_cols = [c for c in data.columns if c in symptom_cols]

    if not symptom_cols:
        return data

    data = data.sort_values([traj_col, date_col]).copy()

    for col in symptom_cols:
        data[col] = (
            data[col]
            .replace("no_mention", np.nan)
            .groupby(data[traj_col])
            .transform(lambda x: x.ffill())
            .fillna("no_mention")
        )
    return data

def forward_fill_note_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill note metadata and interaction features from note events
    to all subsequent events within the same trajectory.
    Non-note events inherit the most recent note's values.
    If no prior note exists in the trajectory, fills with a sensible default.
    """
    date_col = TemporalEvent.DATE.value
    traj_col = Trajectory.ID.value

    # columns that are only populated for note events
    # categorical → fill with "unknown", continuous → fill with 0.0
    note_feature_defaults = {
        # text metadata
        TemporalEvent.NOTE_TYPE.value:                      "unknown",
        TemporalEvent.TEXT_TOKEN_NUMBER.value:              0.0,
        # text interaction
        TemporalEvent.NOTE_CREATION_EMPLOYEE_ROLE.value:    "unknown",
        TemporalEvent.CREATION_EMPLOYEE_EXPERIENCE.value:   0.0,
        TemporalEvent.AUTHOR_FAMILIARITY.value:      0.0,
        TemporalEvent.AUTHOR_DIVERSITY.value:        0.0,
        TemporalEvent.AUTHOR_NOTE_COUNT_SO_FAR.value:       0.0,
        TemporalEvent.NUM_UNIQUE_AUTHORS_SO_FAR.value:      0.0,
    }

    # only keep columns that actually exist in data
    note_feature_defaults = {
        col: default for col, default in note_feature_defaults.items()
        if col in data.columns
    }

    if not note_feature_defaults:
        print("No note feature columns found — skipping forward fill.")
        return data

    data = data.sort_values([traj_col, date_col]).copy()
    note_mask = data[TemporalEvent.TYPE.value] == EventType.NOTE.value

    for col, default in note_feature_defaults.items():
        # non-note rows have no note metadata — set to NaN so ffill can propagate
        data.loc[~note_mask, col] = np.nan

        data[col] = (
            data.groupby(traj_col, sort=False)[col]
            .transform(lambda x: x.ffill())
            .fillna(default)
        )

    filled_cols = list(note_feature_defaults.keys())
    print(f"Forward-filled {len(filled_cols)} note feature columns: {filled_cols}")
    return data

def filter_split(split_df, trigger_cfg):
    mask = pd.Series(True, index=split_df.index)
    if trigger_cfg["combined_mask"] is not None:
        mask &= trigger_cfg["combined_mask"](split_df).values  # ← .values strips the index
    return split_df[mask]
 
def main(config_folder="model_cfgs"):
    baseline_models = ['majority', 'xgboost', 'svm', 'persistence']
    deep_models = ['lstm', 'gru', 'transformer', 'time-lstm', "mlp"]


    is_available = all([df is not None and not df.empty for df in
                        [trajectories, admissions, medications, diagnoses, clinical_texts]])

    if not is_available:
        print("Datasets are not available, ending the program..")
    else:
        print("=== Loading full data once ===")
        start_time = time.time()
        # ── 1. Load structured data ───────────────────────────────────────────────
        data_full = load_temporal_data(FULL_CONFIG)
        print(f"Temporal data loaded in {time.time()-start_time:.2f}s — shape: {data_full.shape}")
        end_time = time.time()
        print(f"Temporal data loaded in {end_time - start_time:.2f}s — shape: {data_full.shape}")
        
        # 2.  load and merge with textual features - including forward-filling of symptom and note features
        #data_full = load_textual_features(data_full, FULL_CONFIG)
    
        # ── 2. Filter events ─────────────────
        data_full = filter_events(data_full)
        print(f"Data shape after filtering: {data_full.shape}")
        
        # ── 3. Load model config ──────────────────────────────────────────────────
        print(f"Config folder path: {config_folder}, exists: {os.path.exists(config_folder)}, files: {os.listdir(config_folder) if os.path.exists(config_folder) else 'N/A'}", flush=True)
        config_path = os.path.join(config_folder, "model_config.yml")
        with open(config_path, "r") as f:
            model_cfg = yaml.safe_load(f)
        
        if 'architecture_configs' in model_cfg:
            arch_runs = [
            {**model_cfg, **cfg}
            for cfg in model_cfg['architecture_configs']
            ]
        else:
            arch_runs = [model_cfg]
        for run_cfg in arch_runs:
            validation_type = run_cfg['validation_type']
            model = run_cfg['model']
            is_baseline = model in baseline_models
            print(f"experimentation is starting with given configs {run_cfg}")
            
            for trigger_cfg in trigger_configs:
                trigger_name = trigger_cfg["name"]
                input_config = trigger_cfg["feature_config"]
                ablation_name = trigger_name
                 # ── 4. Build data_for_run — select correct text/slope variant ─────
                data_for_run = data_full.copy()

                # ── 5. Prepare columns ────────────────────────────────────────────
                print(f"\n{'='*60}")
                print(f"=== Trigger: {trigger_name} ===")

                task = MedicationChangePrediction(outcome=trigger_cfg["outcome"])
              
                splitter = DatasetSplitter(
                    outcome_col=task.outcome,
                    trajectory_col=Trajectory.ID.value,
                    gender_col=Patient.GENDER.value,
                    patient_col=Patient.ID.value,
                )
                data_for_run, final_columns, numeric_columns = prepare_columns_to_preprocess(
                data_for_run.copy(), input_config, task.outcome,
                is_baseline=is_baseline, finetune_bert=False,
                atc_chars=trigger_cfg.get('atc_chars', 5),
                dsm_level=trigger_cfg.get('dsm_level', 'disorder'),
                )
                
                print(f"\n=== Ablation step: {ablation_name} ===")
                print(f"Feature config: {input_config}")
                print(f"Final columns: {(final_columns)}")
                

                filename = f"{model}_{validation_type}_{task.outcome}_{ablation_name}" 
                med_mask = data_for_run[TemporalEvent.TYPE.value].isin([
                        EventType.MEDICATION_START.value,
                        EventType.MEDICATION_STOP.value,
                        EventType.MEDICATION_SWITCH.value,
                    ])
                # for medication events, use the trigger_med_action_type which is more detailed, including also dosage changes etc. 
                data_for_run.loc[med_mask, TemporalEvent.TYPE.value] = \
                    data_for_run.loc[med_mask, TemporalEvent.TRIGGER_MED_ACTION_TYPE.value]

                
                seed =42 
                all_dfs      = []
                train_df, val_df, test_df = splitter.create_hs_sets(
                                                filter_split(data_for_run, trigger_cfg),                        # ← filtered med events only
                                                trigger_action_types=PRIMARY_TRIGGER_ACTIONS, 
                                                random_state=seed
                                            )
                print(f"  train={len(train_df)}, val={len(val_df)}, test={len(test_df) if test_df is not None else '—'}")
                print(f"  Outcome — train:\n{train_df[task.outcome].value_counts().to_string()}")
                print(f"  Outcome — val:\n{val_df[task.outcome].value_counts().to_string()}")                                               
                print(test_df.head(10))    
            
                if model in deep_models:
                    if model_cfg['ablation_mode'] == False:
                        save_dir = model_path / f"{filename}_multiseed"
                        save_dir.mkdir(parents=True, exist_ok=True)
                        
                        df, predictions, summary_df, best_architecture, best_optim, best_batch_size = \
                                train_backbone_fixed_arch(
                                    target_task=task, config=input_config, model_cfg=run_cfg,
                                    train_df=train_df, val_df=val_df, test_df=test_df,
                                    seeds=SEEDS,
                                    target_event_types=PRIMARY_TRIGGER_ACTIONS,
                                    save_dir=save_dir,
                                )
                    
                        print(f"[{model}] Summary across {len(SEEDS)} seeds:\n{best_optim}, {best_batch_size}, {best_architecture}")
                    else:  
                        #set_seed(0)    
                        # df, predictions, summary = train_temporal_model_multiseed(target_task=task, config=input_config,
                        #                                         model_cfg=run_cfg, train_df=train_df,
                        #                                         val_df=val_df, test_df=test_df,
                        #                                         target_event_types=PRIMARY_TRIGGER_ACTIONS)
                        df, predictions = train_temporal_model(target_task=task, config=input_config,
                                        model_cfg=run_cfg, train_df=train_df,
                                        val_df=val_df, test_df=test_df,
                                        target_event_types=PRIMARY_TRIGGER_ACTIONS)
                        #print(summary)    
                    
        
                elif model in baseline_models:
                    # filter only triggering events.. 
                    # if model in ['svm', 'xgboost']:    
                    #     df, predictions = baseline_training(task, input_config, train_df, val_df,
                    #                                                 final_columns, numeric_columns, 
                    #                                                 full_model_path=model_path/f'{filename}_{ablation_name}_pipeline.joblib',
                    #                                                 test_df=test_df, learner=model)

                    if model == 'xgboost':
                        
                        df, predictions = baseline_training(task, input_config, train_df, val_df,
                                                                    final_columns, numeric_columns,
                                                                    full_model_path=model_path/f'{filename}_{ablation_name}_pipeline.joblib',
                                                                    test_df=test_df, learner=model,
                                                                    splitter=splitter,
                                                                    use_cv_ablation=False,
                                                                    n_cv_folds=5,
                                                                    trigger_action_types=PRIMARY_TRIGGER_ACTIONS)
                            
                    elif model == 'majority':
                        df, predictions = majority_baseline(task, train_df, val_df, test_df=test_df)
                    
                    elif model == 'persistence':
                        df, predictions = persistence_baseline(task, train_df, val_df, test_df)
                        
               
                all_dfs.append(df)
                predictions.to_csv(results_path / f'{filename}_predictions_{ablation_name}.csv',
                                        index=False)
            
                if all_dfs:
                    final_df = pd.concat(all_dfs, ignore_index=True)
                    print(
                        f"writing the results to the given folder: /{results_path}/{filename}.csv")

                    save_or_append_features(final_df, results_path, f'{filename}.csv')

            
if __name__ == "__main__":
    main(config_folder=PROJECT_ROOT / "source/models/model_cfgs/")

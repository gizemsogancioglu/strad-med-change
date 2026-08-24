import torch
from source.models.base_model import Task
from datetime import timedelta

from source.data_processing.data_model import EventLevelOutcomes, Trajectory, TemporalEvent, TrajectoryLevelOutcomes
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import pandas as pd
import numpy as np
# todo: work on LLMs for medication change reason extraction from notes.


def add_primary_medication_change_outcomes(
    df: pd.DataFrame,
    window_days: int = 42,
    titration_window_days: int = 28,
) -> pd.DataFrame:
    """
    Adds four outcome columns based on future primary medication changes
    within a rolling prediction window.

    Binary outcome (PRIMARY_MED_CHANGE):
        1 if any meaningful primary medication change occurs within window_days
        0 otherwise. Includes switches, additions, removals, and late dose changes.

    Multi-class outcome (PRIMARY_MED_CHANGE_TYPE):
        'escalation'    : main_switch, late dose increase, augmentation/restart
        'de_escalation' : main_stopped, main_dose_decrease
        'no_change'     : no meaningful change within window

    Drug change outcome (PRIMARY_MED_DRUG_CHANGE):
        1 if a drug addition, removal, or switch occurs within window_days
        0 otherwise. Excludes dose changes — only captures actual drug changes.

    Drug change multi-class outcome (PRIMARY_MED_DRUG_CHANGE_TYPE):  ← NEW
        'escalation'    : main_switch or main_added (augmentation)
        'de_escalation' : main_stopped
        'no_change'     : no drug addition/removal/switch within window
        Dose changes (main_dose_increase, main_dose_decrease) are excluded
        entirely 
    """
    ESCALATION_ACTIONS    = {"main_switch"}
    DE_ESCALATION_ACTIONS = {"main_stopped", "main_dose_decrease"}
    ALL_PRIMARY_ACTIONS   = (
        ESCALATION_ACTIONS | DE_ESCALATION_ACTIONS |
        {"main_added", "main_dose_increase"}
    )

    # drug change: additions, removals, switches only — no dose changes
    DRUG_CHANGE_ACTIONS = {"main_switch", "main_stopped", "main_added"}

    # drug change multiclass: same set, direction-labelled
    DRUG_ESCALATION_ACTIONS    = {"main_switch", "main_added"}
    DRUG_DE_ESCALATION_ACTIONS = {"main_stopped"}

    WINDOW = pd.Timedelta(days=window_days)

    df = df.copy()
    df[TemporalEvent.DATE.value] = pd.to_datetime(df[TemporalEvent.DATE.value])

    df[EventLevelOutcomes.PRIMARY_MED_CHANGE_OUTCOME.value]           = 0
    df[EventLevelOutcomes.PRIMARY_MED_CHANGE_TYPE_OUTCOME.value]      = "no_change"
    df[EventLevelOutcomes.PRIMARY_MED_DRUG_CHANGE_OUTCOME.value]      = 0
    df[EventLevelOutcomes.PRIMARY_MED_DRUG_CHANGE_TYPE_OUTCOME.value] = "no_change"  

    for traj_id, traj_df in df.groupby(Trajectory.ID.value):
        traj_df = traj_df.sort_values(TemporalEvent.DATE.value)

        # track when each ATC was first started — for titration window
        drug_start_dates = {}
        for _, row in traj_df[
            traj_df[TemporalEvent.TRIGGER_MED_ACTION_TYPE.value].isin(
                {"main_added", "main_switch"}
            )
        ].iterrows():
            atc  = row[TemporalEvent.TRIGGER_MED_ATC_CODE.value]
            date = row[TemporalEvent.DATE.value]
            if atc and atc not in drug_start_dates:
                drug_start_dates[atc] = date

        primary_changes = traj_df[
            traj_df[TemporalEvent.TRIGGER_MED_ACTION_TYPE.value].isin(ALL_PRIMARY_ACTIONS)
        ][[
            TemporalEvent.DATE.value,
            TemporalEvent.TRIGGER_MED_ACTION_TYPE.value,
            TemporalEvent.TRIGGER_MED_ATC_CODE.value,
            TemporalEvent.COUNT_ACTIVE_MAIN_MED.value,
        ]].copy()

        if primary_changes.empty:
            continue

        for idx, row in traj_df.iterrows():
            event_date    = row[TemporalEvent.DATE.value]
            window_end    = event_date + WINDOW
            current_count = row[TemporalEvent.COUNT_ACTIVE_MAIN_MED.value]

            future_changes = primary_changes[
                (primary_changes[TemporalEvent.DATE.value] > event_date) &
                (primary_changes[TemporalEvent.DATE.value] <= window_end)
            ]

            if future_changes.empty:
                continue

            is_escalation      = False
            is_de_escalation   = False
            is_drug_change     = False
            is_drug_escalation    = False  
            is_drug_de_escalation = False  

            for _, change in future_changes.iterrows():
                action    = change[TemporalEvent.TRIGGER_MED_ACTION_TYPE.value]
                atc       = change[TemporalEvent.TRIGGER_MED_ATC_CODE.value]
                date      = change[TemporalEvent.DATE.value]
                new_count = change[TemporalEvent.COUNT_ACTIVE_MAIN_MED.value]

                # ── drug change binary outcome ─────────────────────────
                if action in DRUG_CHANGE_ACTIONS:
                    if action == "main_added":
                        if new_count > current_count:
                            is_drug_change     = True
                            is_drug_escalation = True      
                    else:
                        is_drug_change = True
                        if action in DRUG_ESCALATION_ACTIONS:      # main_switch
                            is_drug_escalation = True              
                        elif action in DRUG_DE_ESCALATION_ACTIONS: # main_stopped
                            is_drug_de_escalation = True           

                # ── binary + full multiclass outcome ───────────────────
                if action == "main_switch":
                    is_escalation = True

                elif action == "main_dose_increase":
                    drug_start = drug_start_dates.get(atc)
                    if drug_start is not None:
                        if (date - drug_start).days > titration_window_days:
                            is_escalation = True
                    else:
                        is_escalation = True

                elif action == "main_added":
                    if new_count > current_count:
                        is_escalation = True

                elif action in DE_ESCALATION_ACTIONS:
                    is_de_escalation = True

            # ── write binary outcome ───────────────────────────────────
            if is_escalation or is_de_escalation:
                df.at[idx, EventLevelOutcomes.PRIMARY_MED_CHANGE_OUTCOME.value] = 1

            # ── write full multiclass (escalation takes priority) ──────
            if is_escalation:
                df.at[idx, EventLevelOutcomes.PRIMARY_MED_CHANGE_TYPE_OUTCOME.value] = "escalation"
            elif is_de_escalation:
                df.at[idx, EventLevelOutcomes.PRIMARY_MED_CHANGE_TYPE_OUTCOME.value] = "de_escalation"

            # ── write drug change binary ───────────────────────────────
            if is_drug_change:
                df.at[idx, EventLevelOutcomes.PRIMARY_MED_DRUG_CHANGE_OUTCOME.value] = 1

            # ── write drug change multiclass ─────────────────────
            # escalation takes priority over de_escalation
            # (e.g. same window has both a switch and a stop → escalation)
            if is_drug_escalation:
                df.at[idx, EventLevelOutcomes.PRIMARY_MED_DRUG_CHANGE_TYPE_OUTCOME.value] = "escalation"
            elif is_drug_de_escalation:
                df.at[idx, EventLevelOutcomes.PRIMARY_MED_DRUG_CHANGE_TYPE_OUTCOME.value] = "de_escalation"

    # ── cast dtypes ────────────────────────────────────────────────────
    df[EventLevelOutcomes.PRIMARY_MED_CHANGE_OUTCOME.value] = (
        df[EventLevelOutcomes.PRIMARY_MED_CHANGE_OUTCOME.value].astype("int8")
    )
    df[EventLevelOutcomes.PRIMARY_MED_CHANGE_TYPE_OUTCOME.value] = (
        df[EventLevelOutcomes.PRIMARY_MED_CHANGE_TYPE_OUTCOME.value].astype("string")
    )
    df[EventLevelOutcomes.PRIMARY_MED_DRUG_CHANGE_OUTCOME.value] = (
        df[EventLevelOutcomes.PRIMARY_MED_DRUG_CHANGE_OUTCOME.value].astype("int8")
    )
    df[EventLevelOutcomes.PRIMARY_MED_DRUG_CHANGE_TYPE_OUTCOME.value] = (    
        df[EventLevelOutcomes.PRIMARY_MED_DRUG_CHANGE_TYPE_OUTCOME.value].astype("string")
    )

    print(f"\nPrimary medication change outcomes added.")
    print(f"  Window days           : {window_days}")
    print(f"  Titration window days : {titration_window_days}")

    print_outcome_distributions(df)

    return df

def print_outcome_distributions(df):
     # ── print distributions ────────────────────────────────────────────
    binary_col      = EventLevelOutcomes.PRIMARY_MED_CHANGE_OUTCOME.value
    type_col        = EventLevelOutcomes.PRIMARY_MED_CHANGE_TYPE_OUTCOME.value
    drug_col        = EventLevelOutcomes.PRIMARY_MED_DRUG_CHANGE_OUTCOME.value
    drug_type_col   = EventLevelOutcomes.PRIMARY_MED_DRUG_CHANGE_TYPE_OUTCOME.value   
    event_col       = TemporalEvent.TYPE.value
    admission_col   = TemporalEvent.DURING_ADMISSION.value

    # ── global ─────────────────────────────────────────────────────────
    print(f"\n── Global ──────────────────────────────────────────────────")
    print(f"  Total events              : {len(df)}")
    print(f"  Binary positive rate      : {df[binary_col].mean():.3f}")
    print(f"  Drug change positive rate : {df[drug_col].mean():.3f}")
    print(f"  Multi-class distribution:")
    print(f"{df[type_col].value_counts().to_string()}")
    print(f"  Drug change multi-class distribution:")       
    print(f"{df[drug_type_col].value_counts().to_string()}")

    # ── per event type ─────────────────────────────────────────────────
    print(f"\n── Per Event Type ──────────────────────────────────────────")
    for event_type, group in df.groupby(event_col):
        n         = len(group)
        pos_rate  = group[binary_col].mean()
        drug_rate = group[drug_col].mean()
        type_dist      = group[type_col].value_counts()
        drug_type_dist = group[drug_type_col].value_counts()   
        print(f"\n  {event_type} (n={n}, binary={pos_rate:.3f}, drug_change={drug_rate:.3f})")
        for outcome, count in type_dist.items():
            print(f"    {outcome:<20} : {count:>6} ({count/n:.1%})")
        print(f"  drug_change_type:")                          
        for outcome, count in drug_type_dist.items():          
            print(f"    {outcome:<20} : {count:>6} ({count/n:.1%})")

    # ── primary medication events only ─────────────────────────────────
    print(f"\n── Primary Medication Events Only ──────────────────────────")
    primary_med_action_types = {
        "main_added", "main_stopped", "main_switch",
        "main_dose_increase", "main_dose_decrease",
    }
    primary_med_mask = df[
        TemporalEvent.TRIGGER_MED_ACTION_TYPE.value
    ].isin(primary_med_action_types)

    primary_med_df = df[primary_med_mask]
    n_primary = len(primary_med_df)

    if n_primary == 0:
        print("  No primary medication events found.")
    else:
        print(f"  Total primary med events           : {n_primary}")
        print(f"  Binary positive rate               : {primary_med_df[binary_col].mean():.3f}")
        print(f"  Drug change positive rate          : {primary_med_df[drug_col].mean():.3f}")
        print(f"  Multi-class distribution:")
        print(f"{primary_med_df[type_col].value_counts().to_string()}")
        print(f"  Drug change multi-class distribution:")     # ← NEW
        print(f"{primary_med_df[drug_type_col].value_counts().to_string()}")
        print(f"\n  Per action type:")
        for action, group in primary_med_df.groupby(
            TemporalEvent.TRIGGER_MED_ACTION_TYPE.value
        ):
            n         = len(group)
            pos_rate  = group[binary_col].mean()
            drug_rate = group[drug_col].mean()
            type_dist      = group[type_col].value_counts()
            drug_type_dist = group[drug_type_col].value_counts()   # ← NEW
            print(f"\n    {action} (n={n}, binary={pos_rate:.3f}, drug_change={drug_rate:.3f})")
            for outcome, count in type_dist.items():
                print(f"      {outcome:<20} : {count:>6} ({count/n:.1%})")
            print(f"    drug_change_type:")                        # ← NEW
            for outcome, count in drug_type_dist.items():          # ← NEW
                print(f"      {outcome:<20} : {count:>6} ({count/n:.1%})")

    # ── during admission vs outside ────────────────────────────────────
    print(f"\n── During Admission vs Outside ─────────────────────────────")
    for admission_flag, label in [(1.0, "During admission"), (0.0, "Outside admission")]:
        group    = df[df[admission_col] == admission_flag]
        n        = len(group)
        if n == 0:
            print(f"\n  {label} : no events")
            continue
        pos_rate  = group[binary_col].mean()
        drug_rate = group[drug_col].mean()
        type_dist      = group[type_col].value_counts()
        drug_type_dist = group[drug_type_col].value_counts()   
        print(f"\n  {label} (n={n}, binary={pos_rate:.3f}, drug_change={drug_rate:.3f})")
        for outcome, count in type_dist.items():
            print(f"    {outcome:<20} : {count:>6} ({count/n:.1%})")
        print(f"  drug_change_type:")                          
        for outcome, count in drug_type_dist.items():          
            print(f"    {outcome:<20} : {count:>6} ({count/n:.1%})")

    # ── per trajectory summary ─────────────────────────────────────────
    print(f"\n── Per Trajectory (summary) ────────────────────────────────")
    traj_stats = (
        df.groupby(Trajectory.ID.value)[binary_col]
        .agg(["mean", "count"])
        .rename(columns={"mean": "positive_rate", "count": "n_events"})
    )
    n_escalation = (
        df.groupby(Trajectory.ID.value)[type_col]
        .apply(lambda x: (x == "escalation").any())
        .sum()
    )
    n_de_escalation = (
        df.groupby(Trajectory.ID.value)[type_col]
        .apply(lambda x: (x == "de_escalation").any())
        .sum()
    )
    n_drug_escalation = (                                      
        df.groupby(Trajectory.ID.value)[drug_type_col]
        .apply(lambda x: (x == "escalation").any())
        .sum()
    )
    n_drug_de_escalation = (                                   
        df.groupby(Trajectory.ID.value)[drug_type_col]
        .apply(lambda x: (x == "de_escalation").any())
        .sum()
    )
    print(f"  Trajectories                             : {len(traj_stats)}")
    print(f"  Avg events / trajectory                  : {traj_stats['n_events'].mean():.1f}")
    print(f"  Avg positive rate (binary)               : {traj_stats['positive_rate'].mean():.3f}")
    print(f"  Drug change positive rate                : {df[drug_col].mean():.3f}")
    print(f"  Trajectories with any escalation         : {n_escalation}")
    print(f"  Trajectories with any de_escalation      : {n_de_escalation}")
    print(f"  Trajectories with any drug escalation    : {n_drug_escalation}")     # ← NEW
    print(f"  Trajectories with any drug de_escalation : {n_drug_de_escalation}")  # ← NEW

    return

def precompute_change_dates(outcome_data: pd.DataFrame) -> dict:
    """
    Build a dict: trajectory_id -> sorted np.array of change dates (as np.datetime64).
    Done once, reused across all window sizes and temporal analysis.
    Only includes main_added / main_removed events (primary med changes).
    """
    change_dates = {}
    for _, row in outcome_data.iterrows():
        traj_id     = row[Trajectory.ID.value]
        med_changes = row[TrajectoryLevelOutcomes.MEDICATION_CHANGES.value]

        if not isinstance(med_changes, dict):
            change_dates[traj_id] = np.array([], dtype='datetime64[ns]')
            continue

        dates = np.array([
            np.datetime64(k)
            for k, v in med_changes.items()
            if any(
                evt.endswith("main_added") or evt.endswith("main_removed")
                for evt in v
            )
        ], dtype='datetime64[ns]')

        change_dates[traj_id] = np.sort(dates)

    return change_dates


class MedicationChangePrediction(Task):
    def __init__(self, name="medication_change_prediction", task="classification", outcome=EventLevelOutcomes.PRIMARY_MED_DRUG_CHANGE_OUTCOME.value, num_labels=2):
        super().__init__(name=name, task=task, outcome=outcome, num_labels=num_labels)

    def evaluate_prob(self, labels, preds_prob):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if isinstance(preds_prob, torch.Tensor):
            preds_prob = preds_prob.detach().cpu().numpy()

        labels    = np.array(labels).ravel()
        preds_prob = np.asarray(preds_prob)

        n_classes = len(np.unique(labels))

        if preds_prob.ndim > 1 and preds_prob.shape[1] == 2:
            # binary with 2-column proba — take positive class
            preds_prob = preds_prob[:, 1]

        if n_classes == 2:
            return roc_auc_score(labels, preds_prob.ravel())
        else:
            # multi-class: preds_prob must be (n_samples, n_classes)
            return roc_auc_score(
                labels, preds_prob,
                multi_class='ovr',
                average='macro',
            )

    def evaluate(self, labels, preds):
        # Convert to NumPy only if they are PyTorch tensors
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()

        # Flatten arrays
        labels = np.array(labels).ravel()
        preds = np.array(preds).ravel()
        #return f1_score(labels, pred, average='macro')
        return balanced_accuracy_score(labels, preds)

    def str_measure(self):
        #return "f1_score"
        return "balanced_accuracy"


import random

from abc import abstractmethod

import joblib
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from source.data_processing.data_model import Trajectory, TemporalEvent, ClinicalText, Medication, ListEventFeatures, Features
from source.data_processing.data_reader import model_path
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, brier_score_loss

from source.models.model_cfgs.feature_config import (
    ADMISSION_BINARY_COLUMNS, ADMISSION_CONTINUOUS_COLUMNS, ALL_TIME_COLUMNS, FEATURE_GROUP_COLUMNS, LAB_BINARY_COLUMNS, 
    STATIC_CONTINUOUS_COLUMNS, MED_SUMMARY_COLUMNS,
    DIAG_TIME_COLUMNS, CATEGORICAL_COLUMNS, ALL_BOOLEAN_COLUMNS,
    MED_BOOLEAN_FLAG_COLUMNS, NON_NUMERIC_COLUMNS, TEXT_INTERACTION_CONTINUOUS_COLUMNS, TIMESTAMPS_CONTINUOUS_COLUMNS,
    active, APPOINTMENT_CONTINUOUS_COLUMNS, NONE_VALUES, MISSING_CATEGORY_FILL
)

class Task:

    def __init__(self, name, task, outcome, num_labels):
        self.name = name
        self.task = task
        self.outcome = outcome
        self.num_labels = num_labels

    @abstractmethod
    def set_labels(self, data, outcome_data=None, window_days=None):
        """For every task, we set the outcomes."""
        pass

    @abstractmethod
    def str_measure(self):
        pass

    def evaluate(self, labels, predictions):
        pass
    
class Evaluator:
    def __init__(self):
        self.results = pd.DataFrame()
        self.predictions = pd.DataFrame()
        self.target_event_types = []

    @staticmethod
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)

    def add_results(self, learner, input_config, split_scores, task, event_types):
    
        def _is_binary(labels):
            return len(np.unique(labels)) == 2

        def _to_1d_prob(prob_arr):
            """For binary: return 1-D positive class probs. For multi-class: return as-is."""
            prob_arr = np.asarray(prob_arr)
            if prob_arr.ndim == 2 and prob_arr.shape[1] == 2:
                return prob_arr[:, 1]
            return prob_arr

        def _find_best_threshold(labels, probs):
            """Youden's J statistic (maximizes sensitivity + specificity - 1)."""
            labels = np.asarray(labels)
            probs  = np.asarray(probs)
            if len(np.unique(labels)) < 2 or np.all(np.isnan(probs)):
                return 0.5
            fpr, tpr, thresholds = roc_curve(labels, probs)
            j_scores = tpr - fpr
            return float(thresholds[np.argmax(j_scores)])

        # ── Determine best threshold from the VALIDATION split only ────────
        # This threshold is fixed here and reused, unchanged, across all splits
        # (train/val/test) below — it must never be re-derived on test data.
        best_threshold = 0.5
        if "val" in split_scores:
            val_probs_raw  = self.to_numpy(split_scores["val"][1])
            val_labels_raw = self.to_numpy(split_scores["val"][2])
            if _is_binary(val_labels_raw):
                val_probs_1d   = _to_1d_prob(val_probs_raw)
                best_threshold = _find_best_threshold(val_labels_raw, val_probs_1d)

        def _make_row(row_labels, row_preds, row_probs, row_event_type):
            binary = _is_binary(row_labels)

            r = {
                "learner":      learner,
                "event_types":  event_types,
                "split":        split,
                "input_config": input_config,
                "event_type":   row_event_type,
                task.str_measure(): task.evaluate(row_labels, row_preds),
            }

            # ── ROC-AUC ───────────────────────────────────────────────
            if hasattr(task, "evaluate_prob") and callable(task.evaluate_prob):
                try:
                    prob_input = _to_1d_prob(row_probs) if binary else row_probs
                    r["roc_auc"] = task.evaluate_prob(row_labels, prob_input)
                except Exception:
                    r["roc_auc"] = np.nan

            # ── Brier score (binary only) ─────────────────────────────
            if binary and row_probs is not None:
                prob_1d = _to_1d_prob(row_probs)
                if not np.all(np.isnan(prob_1d)):
                    try:
                        r["brier"] = brier_score_loss(row_labels, prob_1d)
                    except Exception:
                        r["brier"] = np.nan

            # ── Threshold-dependent metrics (default / model's own decision) ──
            if binary:
                sens = recall_score(
                    row_labels, row_preds, pos_label=1, zero_division=np.nan
                )
                spec = recall_score(
                    row_labels, row_preds, pos_label=0, zero_division=np.nan
                )
                r["sensitivity"]  = sens
                r["specificity"]  = spec
                r["ppv"]          = precision_score(
                    row_labels, row_preds, zero_division=np.nan
                )
                r["ppv_default"]  = r["ppv"]
                r["macro_recall"] = (
                    (sens + spec) / 2
                    if not (np.isnan(sens) or np.isnan(spec))
                    else np.nan
                )

                # ── Best-threshold metrics (threshold fixed from val split) ──
                prob_1d = _to_1d_prob(row_probs) if row_probs is not None else None
                if prob_1d is not None and not np.all(np.isnan(prob_1d)):
                    preds_bestthr = (prob_1d >= best_threshold).astype(int)
                    sens_bt = recall_score(
                        row_labels, preds_bestthr, pos_label=1, zero_division=np.nan
                    )
                    spec_bt = recall_score(
                        row_labels, preds_bestthr, pos_label=0, zero_division=np.nan
                    )
                    r["threshold_best"]                  = best_threshold
                    r["sensitivity_thresholdBEST"]        = sens_bt
                    r["specificity_thresholdBEST"]        = spec_bt
                    r["ppv_thresholdBEST"]                = precision_score(
                        row_labels, preds_bestthr, zero_division=np.nan
                    )
                    r["balanced_accuracy_thresholdBEST"]  = (
                        (sens_bt + spec_bt) / 2
                        if not (np.isnan(sens_bt) or np.isnan(spec_bt))
                        else np.nan
                    )
            else:
                # macro-averaged metrics for multi-class
                r["sensitivity"]  = recall_score(
                    row_labels, row_preds, average='macro', zero_division=np.nan
                )
                r["specificity"]  = np.nan
                r["ppv"]          = precision_score(
                    row_labels, row_preds, average='macro', zero_division=np.nan
                )
                r["ppv_default"]  = r["ppv"]
                r["macro_recall"] = r["sensitivity"]

                # per-class precision and recall
                classes = sorted(np.unique(row_labels))
                prec_per = precision_score(
                    row_labels, row_preds,
                    labels=classes, average=None, zero_division=np.nan
                )
                rec_per = recall_score(
                    row_labels, row_preds,
                    labels=classes, average=None, zero_division=np.nan
                )
                for cls, p, rec in zip(classes, prec_per, rec_per):
                    r[f"ppv_{cls}"]         = p
                    r[f"sensitivity_{cls}"] = rec

            return r

        # ── Main split loop ───────────────────────────────────────────
        for split, (preds, preds_prob, labels, df_split) in split_scores.items():
            labels     = self.to_numpy(labels)
            preds      = self.to_numpy(preds)
            preds_prob = self.to_numpy(preds_prob)
            binary     = _is_binary(labels)

            if event_types is not None:
                type_mask = df_split[TemporalEvent.TYPE.value].isin(event_types)
                df_split = df_split[type_mask].copy()
                labels = labels[type_mask.values]
                preds = preds[type_mask.values]
                preds_prob = preds_prob[type_mask.values]


            df_split = df_split.reset_index(drop=True)
            # for binary use 1-D probs; for multi-class keep (n, n_classes)
            preds_prob_1d = _to_1d_prob(preds_prob)
            traj_ids = self.to_numpy(df_split[Trajectory.ID.value])

            # ── 1. Global row ─────────────────────────────────────────
            row = _make_row(labels, preds, preds_prob, "global")

            # trajectory-level score
            df_results = pd.DataFrame({
                "traj":  traj_ids,
                "label": labels,
                "pred":  preds,
            })
            traj_scores = []
            for traj, group in df_results.groupby("traj"):
                tl = group["label"].values
                tp = group["pred"].values
                if len(np.unique(tl)) > 1:
                    traj_scores.append(task.evaluate(tl, tp))
            row[f"{task.str_measure()}_traj"] = (
                np.mean(traj_scores) if traj_scores else np.nan
            )

            # trajectory-level ROC-AUC
            if hasattr(task, "evaluate_prob"):
                traj_roc_auc_scores = []
                for traj, group in df_results.groupby("traj"):
                    traj_labels = group["label"].values
                    traj_idx    = group.index.values
                    traj_probs  = (
                        preds_prob_1d[traj_idx]
                        if binary
                        else preds_prob[traj_idx]
                    )
                    if len(np.unique(traj_labels)) > 1:
                        try:
                            traj_roc_auc_scores.append(
                                task.evaluate_prob(traj_labels, traj_probs)
                            )
                        except Exception:
                            pass
                if traj_roc_auc_scores:
                    row["roc_auc_traj"] = np.mean(traj_roc_auc_scores)

            # ── 2. Per-event-type rows ─────────────────────────────────
            per_event_rows = []
            df_indexed = pd.DataFrame({
                "traj":                   traj_ids,
                "label":                  labels,
                "pred":                   preds,
                TemporalEvent.TYPE.value: df_split[TemporalEvent.TYPE.value].values,
            })

            for evt_type, group in df_indexed.groupby(TemporalEvent.TYPE.value):
                group_labels = group["label"].values
                group_preds  = group["pred"].values
                group_idx    = group.index.values
                group_probs  = (
                    preds_prob_1d[group_idx]
                    if binary
                    else preds_prob[group_idx]
                )

                if len(np.unique(group_labels)) < 2:
                    continue

                evt_row = _make_row(group_labels, group_preds, group_probs, evt_type)

                traj_scores = []
                for traj, traj_group in group.groupby("traj"):
                    tl = traj_group["label"].values
                    tp = traj_group["pred"].values
                    if len(np.unique(tl)) > 1:
                        traj_scores.append(task.evaluate(tl, tp))
                if traj_scores:
                    evt_row[f"{task.str_measure()}_traj"] = np.mean(traj_scores)

                per_event_rows.append(evt_row)

            # ── 2b. Per-trigger-action-type rows ──────────────────────────────
            per_action_rows = []
            action_col = TemporalEvent.TRIGGER_MED_ACTION_TYPE.value

            if action_col in df_split.columns:
                df_action = pd.DataFrame({
                    "traj":       traj_ids,
                    "label":      labels,
                    "pred":       preds,
                    action_col:   df_split[action_col].values,
                    TemporalEvent.TYPE.value: df_split[TemporalEvent.TYPE.value].values,
                })

                for action_type, action_group in df_action.groupby(action_col):
                    action_labels = action_group["label"].values
                    action_preds  = action_group["pred"].values
                    action_idx    = action_group.index.values
                    action_probs  = (
                        preds_prob_1d[action_idx]
                        if binary
                        else preds_prob[action_idx]
                    )

                    if len(np.unique(action_labels)) < 2:
                        continue

                    action_row = _make_row(
                        action_labels, action_preds, action_probs,
                        f"action_{action_type}"   # e.g. action_main_added, action_main_stopped
                    )

                    # trajectory-level score per action type
                    traj_scores = []
                    for traj, traj_group in action_group.groupby("traj"):
                        tl = traj_group["label"].values
                        tp = traj_group["pred"].values
                        if len(np.unique(tl)) > 1:
                            traj_scores.append(task.evaluate(tl, tp))
                    if traj_scores:
                        action_row[f"{task.str_measure()}_traj"] = np.mean(traj_scores)

                    per_action_rows.append(action_row)
                      
                    admission_col      = TemporalEvent.DURING_ADMISSION.value
                    # ── also per action × admission status ────────────────────────
                    if admission_col in df_split.columns:
                        df_action_adm = action_group.copy()
                        df_action_adm["admission"] = df_split.loc[
                            action_group.index, admission_col
                        ].fillna(0).astype(float).values

                        for adm_status, adm_label in [(1.0, 'inpatient'), (0.0, 'outpatient')]:
                            adm_sub = df_action_adm[df_action_adm["admission"] == adm_status]
                            if len(adm_sub) == 0 or len(np.unique(adm_sub["label"].values)) < 2:
                                continue

                            adm_sub_labels = adm_sub["label"].values
                            adm_sub_preds  = adm_sub["pred"].values
                            adm_sub_idx    = adm_sub.index.values
                            adm_sub_probs  = (
                                preds_prob_1d[adm_sub_idx]
                                if binary
                                else preds_prob[adm_sub_idx]
                            )

                            per_action_rows.append(_make_row(
                                adm_sub_labels, adm_sub_preds, adm_sub_probs,
                                f"{adm_label}_action_{action_type}"
                            ))

            # ── 3. Per-admission-status rows ───────────────────────────
            admission_col      = TemporalEvent.DURING_ADMISSION.value
            per_admission_rows = []

            if admission_col in df_split.columns:
                df_adm = pd.DataFrame({
                    "traj":                   traj_ids,
                    "label":                  labels,
                    "pred":                   preds,
                    "admission":              df_split[admission_col].fillna(0).astype(float).values,
                    TemporalEvent.TYPE.value: df_split[TemporalEvent.TYPE.value].values,
                })

                for adm_status, adm_label in [(1.0, 'inpatient'), (0.0, 'outpatient')]:
                    adm_group = df_adm[df_adm['admission'] == adm_status]
                    if len(adm_group) == 0 or len(np.unique(adm_group['label'].values)) < 2:
                        continue

                    adm_labels = adm_group['label'].values
                    adm_preds  = adm_group['pred'].values
                    adm_idx    = adm_group.index.values
                    adm_probs  = (
                        preds_prob_1d[adm_idx]
                        if binary
                        else preds_prob[adm_idx]
                    )

                    adm_row = _make_row(adm_labels, adm_preds, adm_probs, adm_label)
                    per_admission_rows.append(adm_row)

                    for evt_type, evt_group in adm_group.groupby(TemporalEvent.TYPE.value):
                        evt_labels = evt_group['label'].values
                        evt_preds  = evt_group['pred'].values
                        evt_idx    = evt_group.index.values
                        evt_probs  = (
                            preds_prob_1d[evt_idx]
                            if binary
                            else preds_prob[evt_idx]
                        )

                        if len(np.unique(evt_labels)) < 2:
                            continue

                        evt_adm_row = _make_row(
                            evt_labels, evt_preds, evt_probs,
                            f"{adm_label}_{evt_type}",
                        )
                        per_admission_rows.append(evt_adm_row)

            # ── 4. Concat all rows ─────────────────────────────────────
            self.results = pd.concat(
                [self.results,
                 pd.DataFrame([row] + per_event_rows + per_admission_rows + per_action_rows)],
                ignore_index=True
            )

            # ── 5. Event-level predictions ─────────────────────────────
            # for multi-class store the full proba vector as string
            if binary:
                prob_col = preds_prob_1d
                preds_out = (preds_prob_1d >= best_threshold).astype(int)
            else:
                prob_col = [str(list(p)) for p in preds_prob]
                preds_out = preds

            admission_col = TemporalEvent.DURING_ADMISSION.value
            pred_df = pd.DataFrame({
                TemporalEvent.ID.value:            df_split[TemporalEvent.ID.value],
                TemporalEvent.TRAJECTORY_ID.value: df_split[Trajectory.ID.value],
                TemporalEvent.TYPE.value:          df_split[TemporalEvent.TYPE.value],
                TemporalEvent.DATE.value:          df_split[TemporalEvent.DATE.value],
                "prediction":                      preds_out, # for binary class, this is the prediction using the best threshold determined on the validation.
                "probability":                     prob_col,
                "label":                           labels,
                "split":                           split,
                "during_admission":                df_split[admission_col].fillna(0).astype(int).values
                                                    if admission_col in df_split.columns else np.nan,
            })
            self.predictions = pd.concat(
                [self.predictions, pred_df], ignore_index=True
            )


class DatasetSplitter:
    def __init__(self, outcome_col, trajectory_col, gender_col, patient_col,
                 event_type_col: str = 'event_type'):
        self.outcome        = outcome_col
        self.trajectory     = trajectory_col
        self.gender         = gender_col
        self.patient        = patient_col
        self.event_type_col = event_type_col

    # ── helpers ───────────────────────────────────────────────────────────────

    def _is_binary(self, full_df):
        return full_df[self.outcome].dropna().nunique() == 2

    def _outcome_rate(self, df):
        if self._is_binary(df):
            return pd.Categorical(df[self.outcome]).codes.astype(float).mean()
        return None

    @staticmethod
    def _median_bin(series):
        median = series.median()
        return (series >= median).astype(int).astype(str)

    @staticmethod
    def _quartile_bin(series):
        try:
            return pd.qcut(
                series, q=4, labels=['q1', 'q2', 'q3', 'q4'],
                duplicates='drop'
            ).astype(str)
        except ValueError:
            return DatasetSplitter._median_bin(series)

    # ── trajectory aggregation ────────────────────────────────────────────────

    def _aggregate_trajectories(self, full_df,
                            trigger_action_types: list = None):
   
        is_binary = self._is_binary(full_df)

        # ── filter to trigger events for outcome rate ─────────────────────────
        action_col = TemporalEvent.TRIGGER_MED_ACTION_TYPE.value
        if trigger_action_types and action_col in full_df.columns:
            strat_df = full_df[full_df[action_col].isin(trigger_action_types)]
            print(f"  Stratifying on {len(strat_df)} trigger events | "
                f"action distribution: {strat_df[action_col].value_counts().to_dict()}")
        else:
            strat_df = full_df

        # ── patient-level outcome rate ────────────────────────────────────────
        if is_binary:
            outcome_numeric = pd.Categorical(
                strat_df[self.outcome]).codes.astype(float)
            patient_df = (
                strat_df.assign(_o=outcome_numeric)
                .groupby(self.patient)
                .agg(outcome_rate=('_o', 'mean'))
                .reset_index()
            )
        else:
            dominant = (
                strat_df.groupby(self.patient)[self.outcome]
                .agg(lambda x: x.value_counts().index[0])
                .rename('dominant_class')
            )
            patient_df = dominant.reset_index()
            patient_df['outcome_rate'] = (
                pd.Categorical(patient_df['dominant_class'])
                .codes.astype(float)
            )

        # ── ensure all patients in full_df are present ────────────────────────
        # patients with no trigger events (e.g. note-only) still need a split
        all_patients = full_df[self.patient].unique()
        missing = set(all_patients) - set(patient_df[self.patient])
        if missing:
            print(f"  {len(missing)} patients have no trigger events — "
                f"assigned neutral stratum")
            missing_df = pd.DataFrame({
                self.patient:      list(missing),
                'outcome_rate':    0.5,
            })
            patient_df = pd.concat(
-                [patient_df, missing_df], ignore_index=True)   

        # ── single binary stratum ───────────────
        outcome_bin   = self._median_bin(patient_df['outcome_rate'])
        patient_df['stratum'] = outcome_bin

       
        dist = patient_df['stratum'].value_counts().to_dict()
        print(f"  Stratum distribution: {dist}")

        stratify_labels = patient_df.set_index(self.patient)['stratum']
        return patient_df, stratify_labels

    # ── split creation ────────────────────────────────────────────────────────

    def _get_indices(self, full_df, ids):
        return full_df[full_df[self.trajectory].isin(ids)].index.tolist()

    def create_hs_sets(self, full_df, val_ratio=0.15, test_ratio=0.15,
                       random_state=42,
                       trigger_action_types: list = None):
        """
        Hard split: train / val / test stratified by outcome rate of
        trigger_action_types events. All events (notes + medication) go
        into splits; only trigger events drive stratification.
        """
        print(f"\nOutcome distribution (all events):\n"
              f"{full_df[self.outcome].value_counts().to_string()}\n")

        patient_df, stratify_labels = self._aggregate_trajectories(
            full_df,
            trigger_action_types=trigger_action_types,
        )
        patient_ids = patient_df[self.patient].tolist()

        traj_df = (
            full_df[[self.patient, self.trajectory]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        # split 1: train+test vs val
        train_test_patients, val_patients = train_test_split(
            patient_ids,
            test_size    = val_ratio,
            stratify     = stratify_labels.loc[patient_ids].values,
            random_state = random_state,
        )

        # split 2: train vs test
        tv_df     = patient_df[
            patient_df[self.patient].isin(train_test_patients)].copy()
        tv_strata = tv_df['stratum'].copy()
        # collapse any rare strata within train+test subset
        tv_rare   = tv_strata.value_counts()[lambda x: x < 3].index
        tv_strata[tv_strata.isin(tv_rare)] = 'other'

        test_fraction = test_ratio / (1.0 - val_ratio)
        train_patients, test_patients = train_test_split(
            tv_df[self.patient].tolist(),
            test_size    = test_fraction,
            stratify     = tv_strata.values,
            random_state = random_state,
        )

        def get_trajs(pts):
            return traj_df[
                traj_df[self.patient].isin(pts)
            ][self.trajectory].tolist()

        train_df = full_df.loc[
            self._get_indices(full_df, get_trajs(train_patients))].copy()
        val_df   = full_df.loc[
            self._get_indices(full_df, get_trajs(val_patients))].copy()
        test_df  = full_df.loc[
            self._get_indices(full_df, get_trajs(test_patients))].copy()

        # no-leakage assertions
        assert not (set(train_df[self.patient].unique()) &
                    set(val_df[self.patient].unique())),  "Overlap: train ∩ val"
        assert not (set(train_df[self.patient].unique()) &
                    set(test_df[self.patient].unique())), "Overlap: train ∩ test"
        assert not (set(val_df[self.patient].unique()) &
                    set(test_df[self.patient].unique())), "Overlap: val ∩ test"

        self._print_stats(train_df, val_df, test_df)
        return train_df, val_df, test_df

    # ── diagnostics ───────────────────────────────────────────────────────────

    def _print_stats(self, train_df, val_df, test_df):
        is_binary = self._is_binary(train_df)
        W = 72
        splits = [('train', train_df), ('val', val_df), ('test', test_df)]

        print("\n" + "=" * W)
        print("SPLIT VERIFICATION")
        print("=" * W)
        print(f"{'Metric':<34} {'Train':>10} {'Val':>10} {'Test':>10}")
        print("-" * W)

        for name, fmt in [('n_events', 'd'), ('n_patients', 'd'),
                          ('events_per_patient', '.1f')]:
            vals = {s: (len(df) if name == 'n_events'
                        else df[self.patient].nunique() if name == 'n_patients'
                        else df.groupby(self.patient).size().mean())
                    for s, df in splits}
            row = f"{name:<34}"
            for s in ['train', 'val', 'test']:
                row += f" {vals[s]:>10{fmt}}"
            print(row)

        # outcome rates
        print()
        if is_binary:
            vals = {s: pd.Categorical(df[self.outcome]).codes.astype(float).mean()
                    for s, df in splits}
            row  = f"{'outcome_rate':<34}"
            for s in ['train', 'val', 'test']:
                row += f" {vals[s]:>10.3f}"
            gap  = abs(vals['val'] - vals['test'])
            row += f"  {'⚠️' if gap > 0.05 else '✓ '} val/test Δ={gap:.3f}"
            print(row)
        else:
            for cls in sorted(set(train_df[self.outcome].dropna().unique())):
                vals = {s: (df[self.outcome] == cls).mean() for s, df in splits}
                row  = f"  {cls:<32}"
                for s in ['train', 'val', 'test']:
                    row += f" {vals[s]:>10.3f}"
                gap  = abs(vals['val'] - vals['test'])
                row += f"  {'⚠️' if gap > 0.05 else '✓ '} Δ={gap:.3f}"
                print(row)

        # event type rates
        if self.event_type_col in val_df.columns:
            print("\nEvent type rates (val vs test):")
            for evt in sorted(set(val_df[self.event_type_col].dropna()) |
                               set(test_df[self.event_type_col].dropna())):
                vr   = (val_df[self.event_type_col]  == evt).mean()
                tr   = (test_df[self.event_type_col] == evt).mean()
                gap  = abs(vr - tr)
                flag = '  ⚠️' if gap > 0.04 else ''
                print(f"  {evt:<32} val={vr:.3f}  test={tr:.3f}"
                      f"  Δ={gap:.3f}{flag}")

        # admission rate
        adm_col = TemporalEvent.DURING_ADMISSION.value
        if adm_col in val_df.columns:
            vr   = (val_df[adm_col]  == 1.0).mean()
            tr   = (test_df[adm_col] == 1.0).mean()
            gap  = abs(vr - tr)
            flag = '  ⚠️' if gap > 0.04 else ''
            print(f"\n  {'admission_rate':<32} val={vr:.3f}  test={tr:.3f}"
                  f"  Δ={gap:.3f}{flag}")

        print("=" * W + "\n")


    # ── k-fold CV ─────────────────────────────────────────────────────────────

    def create_stratified_folds(self, full_df, n_splits=5, random_state=42,
                                trigger_action_types: list = None):
        """Stratified k-fold CV at patient level."""
        patient_df, stratify_labels = self._aggregate_trajectories(
            full_df, trigger_action_types=trigger_action_types)
        patients = patient_df[self.patient].values
        traj_df  = (full_df[[self.patient, self.trajectory]]
                    .drop_duplicates().reset_index(drop=True))

        skf   = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                random_state=random_state)
        folds = []
        for tr_idx, vl_idx in skf.split(
                patients, stratify_labels.loc[patients].values):
            tr_pts  = patients[tr_idx].tolist()
            vl_pts  = patients[vl_idx].tolist()
            tr_traj = traj_df[traj_df[self.patient].isin(tr_pts)][self.trajectory].tolist()
            vl_traj = traj_df[traj_df[self.patient].isin(vl_pts)][self.trajectory].tolist()
            folds.append({
                'train_idx': self._get_indices(full_df, tr_traj),
                'val_idx':   self._get_indices(full_df, vl_traj),
            })
        return folds

class PatientTrajectoryDataset(Dataset):

    def __init__(
            self,
            df,
            target_task,
            event2id,
            gender2id,
            diagnosis2id,
            symptom2id,
            med_vocab,
            bert_tokenizer=None,
            max_len=512,
            stride=256,
            scaler=None,
            fit_scaler=False,
            diag_group2id=None,
            diag_subgroup2id=None,
            role2id=None,
            note_type2id=None,
            appointment_role2id=None,
            trigger_med_type2id=None,
            prev_drug_class2id=None,
            lab_measure2id=None,
            lab_test_type2id=None,
    ):
        self.df = df.reset_index(drop=True)
        self.trajectory_groups = self.df.groupby(Trajectory.ID.value)
        self.trajectory_ids = list(self.trajectory_groups.groups.keys())

        self.outcome = target_task.outcome
        self.task = target_task.task

        self.bert_tokenizer = bert_tokenizer
        self.max_len = max_len
        self.stride = stride
        self.text_column = ClinicalText.TEXT.value

        self.event_column = TemporalEvent.TYPE.value
        self.med_ATC_list_column = Medication.ATC_CODE.value
        self.med_duration_column = Medication.DURATION.value

        self.bert_columns = [col for col in self.df.columns if col.startswith("bert_feature")]
        self.tfidf_columns = [col for col in self.df.columns if col.startswith("tfidf_feature")]
        self.symptom_pred_columns = [col for col in self.df.columns if col.startswith("symptom_pred_cat_")]

        
        # ── TRIGGER MEDICATIONS ───────────────────────────────────────────────
        self.trigger_med_type_col  = TemporalEvent.TRIGGER_MED_DRUG_CLASS.value
        self.prev_drug_class_col   = TemporalEvent.PREV_DRUG_CLASS.value
        
        self.has_trigger_meds = all(
            c in self.df.columns for c in [
                self.trigger_med_type_col,
            ]
        )
        self.trigger_med_type2id     = trigger_med_type2id
        self.prev_drug_class2id      = prev_drug_class2id
        
        self.has_trigger_med_type = (
            trigger_med_type2id is not None and
            self.trigger_med_type_col in self.df.columns
        )
       
        self.has_prev_drug_class = (
            prev_drug_class2id is not None and
            self.prev_drug_class_col in self.df.columns
        )
        
        # ── Static patient features ───────────────────────────────────────────
        self.static_continuous_columns = active(STATIC_CONTINUOUS_COLUMNS, df)

        # ── TIMESTAMPS modality ────────────────────────────────────────────────
        self.dynamic_columns_continuous = active(TIMESTAMPS_CONTINUOUS_COLUMNS, df)

        # ── ADMISSION modality ─────────────────────────────────────────────────
        self.admission_continuous_columns = active(ADMISSION_CONTINUOUS_COLUMNS, df)
        self.admission_binary_columns     = active(ADMISSION_BINARY_COLUMNS, df)

        # ── LAB modality ───────────────────────────────────────────────────────
        self.lab_binary_columns = active(LAB_BINARY_COLUMNS, df)

        self.lab_measure2id    = lab_measure2id
        self.lab_test_type2id  = lab_test_type2id
        self.lab_measure_col   = TemporalEvent.RECENT_LAB_MEASURE.value
        self.lab_test_type_col = TemporalEvent.RECENT_LAB_TEST_TYPE.value
        self.has_lab_measure = (
            lab_measure2id is not None and self.lab_measure_col in self.df.columns
        )
        self.has_lab_test_type = (
            lab_test_type2id is not None and self.lab_test_type_col in self.df.columns
        )
      
        # ── MED_BOOLEAN_FLAGS modality ─────────────────────────────────────────
        self.med_boolean_columns = active(MED_BOOLEAN_FLAG_COLUMNS, df)

        # ── MED_SUMMARY ────────────────────────────────────────────────────────
        self.med_summary_columns = active(MED_SUMMARY_COLUMNS, df)

        # ── DIAGNOSIS time columns ─────────────────────────────────────────────
        self.diag_time_columns = active(DIAG_TIME_COLUMNS, df)

        # ── APPOINTMENT modality ───────────────────────────────────────────────
        self.appointment_continuous_columns = active(APPOINTMENT_CONTINUOUS_COLUMNS, df)

        # ══════════════════════════════════════════════════════════════════════
        # TEXT-DERIVED metadata  
        # note_meta_feats = role one-hot + text-interaction continuous columns
        # (experience, author familiarity, author diversity).
        # ══════════════════════════════════════════════════════════════════════
        self.text_interaction_continuous_columns = active(TEXT_INTERACTION_CONTINUOUS_COLUMNS, df)

        self.gender_column = TemporalEvent.PATIENT_GENDER.value
        self.gender2id = gender2id
        self.event2id = event2id
        self.diagnosis2id = diagnosis2id
        self.symptom2id = symptom2id
        self.med_vocab = med_vocab
        self.diag_group2id = diag_group2id
        self.diag_subgroup2id = diag_subgroup2id
        self.role2id = role2id
        self.note_type2id = note_type2id          # kept for compatibility, unused in note_meta now
        self.appointment_role2id = appointment_role2id


        self.has_atc = (
            Medication.ATC_CODE.value in self.df.columns
        )

        # ── Note author role (CHANGED: now drives note_meta directly) ──────────
        self.note_role_column = TemporalEvent.NOTE_CREATION_EMPLOYEE_ROLE.value
        self.has_note_role = (
            self.role2id is not None and
            self.note_role_column in self.df.columns
        )

        self.appointment_role_column = TemporalEvent.APPOINTMENT_ROLE.value
        self.has_appointment_role = (
            self.appointment_role2id is not None and
            self.appointment_role_column in self.df.columns
        )
        self.has_note_type = (
            self.note_type2id is not None and
            TemporalEvent.NOTE_TYPE.value in self.df.columns
        )

        self.diag_main_column = TemporalEvent.ACTIVE_DIAGNOSIS_MAIN.value \
            if TemporalEvent.ACTIVE_DIAGNOSIS_MAIN.value in self.df.columns else None
        self.diag_secondary_column = TemporalEvent.ACTIVE_DIAGNOSIS_SECONDARY.value \
            if TemporalEvent.ACTIVE_DIAGNOSIS_SECONDARY.value in self.df.columns else None

        self.has_diag_hierarchy = (
            self.diag_group2id is not None and
            self.diag_subgroup2id is not None
        )

        # ── Vocabulary sizes ──────────────────────────────────────────────────
        self.event_vocab_size              = len(event2id) if event2id else 0
        self.diagnosis_vocab_size          = len(diagnosis2id) if diagnosis2id else 0
        self.symptom_vocab_size            = len(symptom2id) if symptom2id else 0
        self.diagnosis_group_vocab_size    = len(diag_group2id) if diag_group2id else 0
        self.diagnosis_subgroup_vocab_size = len(diag_subgroup2id) if diag_subgroup2id else 0

        # ── Modality dimensions ───────────────────────────────────────────────
        self.dynamic_dim = len(self.dynamic_columns_continuous)

        self.admission_dim = (
            len(self.admission_continuous_columns) +
            len(self.admission_binary_columns)
        )

        self.lab_dim = len(self.lab_binary_columns)
        if self.has_lab_measure:
            self.lab_dim += len(self.lab_measure2id)
        if self.has_lab_test_type:
            self.lab_dim += len(self.lab_test_type2id)
            
        self.med_boolean_dim = len(self.med_boolean_columns)
        self.med_summary_dim = len(self.med_summary_columns)

        appt_role_dim = len(self.appointment_role2id) if self.has_appointment_role else 0
        self.appointment_dim = (
            len(self.appointment_continuous_columns) + appt_role_dim
        )

        # TRIGGER_MEDICATIONS dim
        self.trigger_med_dim = 0
        if self.has_trigger_med_type:
            self.trigger_med_dim += len(trigger_med_type2id)
        if self.has_prev_drug_class:
            self.trigger_med_dim += len(prev_drug_class2id)

        # ── NOTE metadata dim (CHANGED) ────────────────────────────────────────
        # role one-hot + text-interaction continuous columns
        role_dim = len(self.role2id) if self.has_note_role else 0
        note_type_dim = len(self.note_type2id) if self.has_note_type else 0
        self.note_meta_dim = role_dim + note_type_dim + len(self.text_interaction_continuous_columns)
        print(f"DEBUG: note meta dim {self.note_meta_dim}")
        print(f"DEBUG role={self.has_note_role} "
        f"int_cols={self.text_interaction_continuous_columns} "
        f"role_col_present={self.note_role_column in self.df.columns}")
        
        # Other dims
        self.diag_time_dim    = len(self.diag_time_columns)
        self.bert_dim         = len(self.bert_columns)
        self.tfidf_dim        = len(self.tfidf_columns)
        self.symptom_pred_dim = len(self.symptom_pred_columns)
        self.static_dim       = len(self.static_continuous_columns) + len(self.gender2id)

        self.fit_scaler = fit_scaler
        self.preprocessors = scaler if scaler is not None else self._init_preprocessors()
        self.NONE_VALUES = NONE_VALUES  

        if self.fit_scaler:
            self._fit_preprocessors()

        self._tokenized_cache = {}
      

    def __len__(self):
        return len(self.trajectory_ids)

    def _init_preprocessors(self):
        return {
            "tfidf":            StandardScaler(),
            "static":           StandardScaler(),
            "timestamps":       StandardScaler(),
            "admission":        StandardScaler(),
            "diag_time":        StandardScaler(),
            "med_summary":      StandardScaler(),
            "text_interaction": StandardScaler(),   # role-continuous note features
            "appointment":      StandardScaler(),
        }

    def _map_cat(self, val, vocab):
        s = str(val)
        if s in self.NONE_VALUES:
            return vocab["<NONE>"]
        return vocab.get(s, vocab["<NONE>"])

    def _map_diag(self, d):
        s = str(d)
        if s in self.NONE_VALUES:
            return self.diagnosis2id["<NONE>"]
        return self.diagnosis2id.get(s, self.diagnosis2id["<NONE>"])

    def _map_symptom_row(self, row):
        return [
            self.symptom2id.get(
                "<NONE>" if str(v) in self.NONE_VALUES else str(v),
                self.symptom2id["<NONE>"]
            )
            for v in row
        ]

    def _map_diag_group(self, d):
        if self.diag_group2id is None:
            return 0
        s = str(d)
        if s in self.NONE_VALUES:
            return self.diag_group2id["<NONE>"]
        group = s.split(".")[0] if "." in s else s
        return self.diag_group2id.get(group, self.diag_group2id["<NONE>"])

    def _map_diag_subgroup(self, d):
        if self.diag_subgroup2id is None:
            return 0
        s = str(d)
        if s in self.NONE_VALUES:
            return self.diag_subgroup2id["<NONE>"]
        parts = s.split(".")
        subgroup = ".".join(parts[:2]) if len(parts) >= 2 else s
        return self.diag_subgroup2id.get(subgroup, self.diag_subgroup2id["<NONE>"])

    def _map_meds(self, meds_val):
        if isinstance(meds_val, str):
            if meds_val.strip() == "" or meds_val.strip() in self.NONE_VALUES:
                return []
            meds_list = meds_val.split()
        elif isinstance(meds_val, list):
            meds_list = meds_val
        elif isinstance(meds_val, float) and np.isnan(meds_val):
            return []
        elif meds_val is None or len(meds_val) == 0:
            return []
        else:
            return []

        med_ids = []
        for atc_code in meds_list:
            atc_code = atc_code.strip()
            if not atc_code or atc_code in self.NONE_VALUES:
                continue
            
            med_ids.append({
                "l2": self.med_vocab["l2"].get(atc_code[:3], self.med_vocab["l2"]["<NONE>"]),
                "l3": self.med_vocab["l3"].get(atc_code[:4], self.med_vocab["l3"]["<NONE>"]),
                "l4": self.med_vocab["l4"].get(atc_code[:5], self.med_vocab["l4"]["<NONE>"]),
                "l5": self.med_vocab["l5"].get(atc_code,     self.med_vocab["l5"]["<NONE>"]),
            })
      
        return med_ids

    def _fit_preprocessors(self):
        tfidf_vals, static_vals = [], []
        timestamps_vals, admission_vals = [], []
        diag_time_vals, med_summary_vals = [], []
        text_int_vals, appointment_vals = [], []   # CHANGED: text_int_vals

        for _, traj in self.trajectory_groups:
            if traj.empty:
                continue

            if self.dynamic_columns_continuous:
                timestamps_vals.append(
                    traj[self.dynamic_columns_continuous].apply(
                                    pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32))

            if self.admission_continuous_columns:
                admission_vals.append(
                    traj[self.admission_continuous_columns].apply(
                        pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32))

            if self.diag_time_columns:
                diag_time_vals.append(
                    traj[self.diag_time_columns].apply(
                        pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32))

            if self.med_summary_columns:
                med_summary_vals.append(
                    np.nan_to_num(traj[self.med_summary_columns].values.astype(np.float32), nan=0.0))

            if self.static_continuous_columns:
                static_vals.append(
                    traj[self.static_continuous_columns].apply(
                        pd.to_numeric, errors="coerce").fillna(0).values[:1].astype(np.float32))

            if self.text_interaction_continuous_columns:
                text_int_vals.append(
                    traj[self.text_interaction_continuous_columns].apply(
                        pd.to_numeric, errors="coerce"
                    ).fillna(0).values.astype(np.float32))

            if self.tfidf_columns:
                tfidf_vals.append(
                    np.nan_to_num(traj[self.tfidf_columns].values.astype(np.float32), nan=0.0))

            if self.appointment_continuous_columns:
                appointment_vals.append(
                    traj[self.appointment_continuous_columns].apply(
                        pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32))

        if timestamps_vals:
            self.preprocessors["timestamps"].fit(np.vstack(timestamps_vals))
        if admission_vals:
            self.preprocessors["admission"].fit(np.vstack(admission_vals))
        if diag_time_vals:
            self.preprocessors["diag_time"].fit(np.vstack(diag_time_vals))
        if med_summary_vals:
            self.preprocessors["med_summary"].fit(np.vstack(med_summary_vals))
        if static_vals:
            self.preprocessors["static"].fit(np.vstack(static_vals))
        if text_int_vals:                                       # CHANGED
            self.preprocessors["text_interaction"].fit(np.vstack(text_int_vals))
        if tfidf_vals:
            self.preprocessors["tfidf"].fit(np.vstack(tfidf_vals))
        if appointment_vals:
            self.preprocessors["appointment"].fit(np.vstack(appointment_vals))

    def __getitem__(self, idx):
        traj_id = self.trajectory_ids[idx]
        traj = self.trajectory_groups.get_group(traj_id)
        n_steps = len(traj)

        # ── BERT tokenization ─────────────────────────────────────────────────
        input_ids_list, attention_mask_list = [], []
        if self.bert_tokenizer is not None and self.text_column in traj.columns:
            for text in traj[self.text_column]:
                if not isinstance(text, str) or text.strip() == "":
                    input_ids_list.append([])
                    attention_mask_list.append([])
                    continue
                encoding = self.bert_tokenizer(
                    text, truncation=True, max_length=self.max_len,
                    stride=self.stride, return_overflowing_tokens=True,
                    padding="max_length", return_tensors="pt",
                )
                n_chunks = encoding["input_ids"].size(0)
                input_ids_list.append([encoding["input_ids"][i] for i in range(n_chunks)])
                attention_mask_list.append([encoding["attention_mask"][i] for i in range(n_chunks)])

        # ── Pretrained text features (BERT or TF-IDF) ─────────────────────────
        if len(self.bert_columns) > 0:
            text_feats = np.nan_to_num(traj[self.bert_columns].values.astype(np.float32), nan=0.0)
        elif len(self.tfidf_columns) > 0:
            text_feats = np.nan_to_num(traj[self.tfidf_columns].values.astype(np.float32), nan=0.0)
            if self.preprocessors["tfidf"] is not None:
                text_feats = self.preprocessors["tfidf"].transform(text_feats)
        else:
            text_feats = np.zeros((n_steps, 0), dtype=np.float32)

        # ── TIMESTAMPS ─────────────────────────────────────────────────────────
        if self.dynamic_columns_continuous:
            ts_cont = traj[self.dynamic_columns_continuous].apply(
                pd.to_numeric, errors="coerce").values.astype(np.float32)
            ts_cont = self.preprocessors["timestamps"].transform(ts_cont)
        else:
            ts_cont = np.zeros((n_steps, 0), dtype=np.float32)
        dynamic_feats = ts_cont

        # ── ADMISSION ──────────────────────────────────────────────────────────
        admission_parts = []
        if self.admission_continuous_columns:
            adm_cont = traj[self.admission_continuous_columns].apply(
                pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
            adm_cont = self.preprocessors["admission"].transform(adm_cont)
            admission_parts.append(adm_cont)
        if self.admission_binary_columns:
            adm_bin = np.nan_to_num(
                traj[self.admission_binary_columns].values.astype(np.float32), nan=0.0)
            admission_parts.append(adm_bin)
        admission_feats = np.concatenate(admission_parts, axis=-1) if admission_parts \
            else np.zeros((n_steps, 0), dtype=np.float32)

        # ── LAB ────────────────────────────────────────────────────────────────
        lab_parts = []
        if self.lab_binary_columns:
            lab_parts.append(np.nan_to_num(
                traj[self.lab_binary_columns].values.astype(np.float32), nan=0.0))

        if self.has_lab_measure:
            measure_ids = [
                self._map_cat(v, self.lab_measure2id)
                for v in traj[self.lab_measure_col].fillna("<NONE>").astype(str)
            ]
            lab_parts.append(
                np.eye(len(self.lab_measure2id), dtype=np.float32)[measure_ids])

        if self.has_lab_test_type:
            test_type_ids = [
                self._map_cat(v, self.lab_test_type2id)
                for v in traj[self.lab_test_type_col].fillna("<NONE>").astype(str)
            ]
            lab_parts.append(
                np.eye(len(self.lab_test_type2id), dtype=np.float32)[test_type_ids])

        lab_feats = np.concatenate(lab_parts, axis=-1) if lab_parts \
            else np.zeros((n_steps, 0), dtype=np.float32)
        
      
        # ── MED_BOOLEAN_FLAGS ──────────────────────────────────────────────────
        if self.med_boolean_columns:
            med_boolean_feats = np.nan_to_num(
                traj[self.med_boolean_columns].values.astype(np.float32), nan=0.0)
        else:
            med_boolean_feats = np.zeros((n_steps, 0), dtype=np.float32)

        # ── TRIGGER MEDICATIONS ────────────────────────────────────────────────
        # Low-cardinality categoricals → one-hot, concatenated into trigger_med_feats.
        # High-cardinality ATC codes → integer ids, embedded later via shared embedding.
        trigger_onehot_parts = []

        if self.has_trigger_med_type:
            type_ids = [
                self._map_cat(v, self.trigger_med_type2id)
                for v in traj[self.trigger_med_type_col].fillna('<NONE>').astype(str)
            ]
            trigger_onehot_parts.append(
                np.eye(len(self.trigger_med_type2id), dtype=np.float32)[type_ids])

        if self.has_prev_drug_class:
            prev_class_ids = [
                self._map_cat(v, self.prev_drug_class2id)
                for v in traj[self.prev_drug_class_col].fillna('<NONE>').astype(str)
            ]
            trigger_onehot_parts.append(
                np.eye(len(self.prev_drug_class2id), dtype=np.float32)[prev_class_ids])

        trigger_med_feats = np.concatenate(trigger_onehot_parts, axis=-1) \
            if trigger_onehot_parts else np.zeros((n_steps, 0), dtype=np.float32)

        # ── DIAGNOSIS time features ────────────────────────────────────────────
        if self.diag_time_columns:
            diag_time_feats = traj[self.diag_time_columns].apply(
                pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
            diag_time_feats = self.preprocessors["diag_time"].transform(diag_time_feats)

        else:
            diag_time_feats = np.zeros((n_steps, 0), dtype=np.float32)

        # ── MED_SUMMARY ────────────────────────────────────────────────────────
        if self.med_summary_columns:
            med_summary_feats = np.nan_to_num(
                traj[self.med_summary_columns].values.astype(np.float32), nan=0.0)
            if self.preprocessors["med_summary"] is not None:
                med_summary_feats = self.preprocessors["med_summary"].transform(med_summary_feats)
        else:
            med_summary_feats = np.zeros((n_steps, 0), dtype=np.float32)

        # ── PATIENT (static) ───────────────────────────────────────────────────
        if self.static_continuous_columns:
            static_continuous_feats = self.preprocessors["static"].transform(
                traj[self.static_continuous_columns].values.astype(np.float32))[0]
            gender_val    = str(traj[self.gender_column].iloc[0])
            gender_id     = self.gender2id.get(gender_val, self.gender2id.get("<NONE>", 0))
            gender_one_hot = np.eye(len(self.gender2id), dtype=np.float32)[gender_id]
            static_feats  = np.concatenate([static_continuous_feats, gender_one_hot], axis=0)
        else:
            static_feats = np.zeros(self.static_dim, dtype=np.float32)

        # ══════════════════════════════════════════════════════════════════════
        # NOTE metadata: role one-hot + text-interaction continuous
        # ══════════════════════════════════════════════════════════════════════
        parts = []
        if self.has_note_role:
            role_vals = traj[self.note_role_column].fillna("<NONE>").astype(str)
            role_ids  = [self.role2id.get(r, self.role2id["<NONE>"]) for r in role_vals]
            role_onehot = np.eye(len(self.role2id), dtype=np.float32)[role_ids]
            parts.append(role_onehot)
            
        if self.has_note_type:
            note_type_vals = traj[TemporalEvent.NOTE_TYPE.value].fillna("<NONE>").astype(str)
            note_type_ids  = [self.note_type2id.get(nt, self.note_type2id["<NONE>"]) for nt in note_type_vals]
            note_type_onehot = np.eye(len(self.note_type2id), dtype=np.float32)[note_type_ids]
            parts.append(note_type_onehot)

        if self.text_interaction_continuous_columns:
            text_int = traj[self.text_interaction_continuous_columns].apply(
                pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
            text_int = self.preprocessors["text_interaction"].transform(text_int)
            parts.append(text_int)

        note_meta_feats = np.concatenate(parts, axis=-1) if parts \
            else np.zeros((n_steps, 0), dtype=np.float32)

        # ── APPOINTMENT ────────────────────────────────────────────────────────
        appt_parts = []
        if self.appointment_continuous_columns:
            appt_cont = traj[self.appointment_continuous_columns].apply(
                pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
            appt_cont = self.preprocessors["appointment"].transform(appt_cont)
            appt_parts.append(appt_cont)
        if self.has_appointment_role:
            appt_role_vals = traj[self.appointment_role_column].fillna("<NONE>").astype(str)
            appt_role_ids  = [
                self.appointment_role2id.get(v, self.appointment_role2id["<NONE>"])
                for v in appt_role_vals
            ]
            appt_parts.append(
                np.eye(len(self.appointment_role2id), dtype=np.float32)[appt_role_ids])
        appointment_feats = np.concatenate(appt_parts, axis=-1) if appt_parts \
            else np.zeros((n_steps, 0), dtype=np.float32)

        # ── Categorical IDs ────────────────────────────────────────────────────
        event_ids = torch.tensor(
            [self.event2id.get(e, 0) for e in traj[self.event_column]], dtype=torch.long)
        event_one_hot = torch.nn.functional.one_hot(
            event_ids, num_classes=len(self.event2id)).float()

        if self.symptom_pred_columns:
            symptom_ids = torch.tensor(
                [self._map_symptom_row(row)
                 for row in traj[self.symptom_pred_columns].itertuples(index=False)],
                dtype=torch.long)
        else:
            symptom_ids = torch.zeros((n_steps, 0), dtype=torch.long)

        if self.diag_main_column:
            diag_main_ids = torch.tensor(
                [self._map_diag(d) for d in traj[self.diag_main_column]], dtype=torch.long)
            diag_secondary_ids = torch.tensor(
                [self._map_diag(d) for d in traj[self.diag_secondary_column]], dtype=torch.long)
        else:
            diag_main_ids      = torch.zeros(n_steps, dtype=torch.long)
            diag_secondary_ids = torch.zeros(n_steps, dtype=torch.long)

        if self.has_diag_hierarchy:
            diag_group_ids = torch.tensor(
                [self._map_diag_group(d) for d in traj[self.diag_main_column]], dtype=torch.long)
            diag_subgroup_ids = torch.tensor(
                [self._map_diag_subgroup(d) for d in traj[self.diag_main_column]], dtype=torch.long)
        else:
            diag_group_ids    = torch.zeros(n_steps, dtype=torch.long)
            diag_subgroup_ids = torch.zeros(n_steps, dtype=torch.long)

        # ── Medications ────────────────────────────────────────────────────────
        if self.has_atc:
            ATC_data_per_event = []
            for t in range(n_steps):
                meds_val = traj[self.med_ATC_list_column].iloc[t]
                med_ids_hier = self._map_meds(meds_val)
                if not med_ids_hier:
                    med_ids_hier = [{
                        "l2": self.med_vocab["l2"]["<NONE>"],
                        "l3": self.med_vocab["l3"]["<NONE>"],
                        "l4": self.med_vocab["l4"]["<NONE>"],
                        "l5": self.med_vocab["l5"]["<NONE>"],
                    }]
                ATC_data_per_event.append(med_ids_hier)

        else:
            ATC_data_per_event = []
        # ── Labels ─────────────────────────────────────────────────────────────
        label_tensor = torch.tensor(
            traj[self.outcome].values,
            dtype=torch.float32 if self.task == "regression" else torch.long)

        return (
            input_ids_list,                                               # 0
            attention_mask_list,                                          # 1
            torch.tensor(static_feats, dtype=torch.float32),             # 2
            torch.tensor(dynamic_feats, dtype=torch.float32),            # 3
            torch.tensor(diag_time_feats, dtype=torch.float32),          # 4
            torch.tensor(text_feats, dtype=torch.float32),               # 5
            event_one_hot,                                                # 6
            diag_main_ids,                                                # 7
            diag_secondary_ids,                                           # 8
            torch.tensor(med_summary_feats, dtype=torch.float32),        # 9
            ATC_data_per_event,                                           # 10
            label_tensor,                                                 # 11
            torch.tensor(note_meta_feats, dtype=torch.float32),          # 12
            diag_group_ids,                                               # 13
            diag_subgroup_ids,                                            # 14
            symptom_ids,                                                  # 15
            torch.tensor(appointment_feats, dtype=torch.float32),        # 16
            torch.tensor(admission_feats, dtype=torch.float32),          # 17
            torch.tensor(lab_feats, dtype=torch.float32),                # 18
            torch.tensor(med_boolean_feats, dtype=torch.float32),        # 19
            torch.tensor(trigger_med_feats, dtype=torch.float32),        # 20
        )
                

    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_events(df, event_type):
    df = df[df[TemporalEvent.TYPE.value] == event_type]
    return df

def extract_dsm_level(code, levels):
    """
    Extract a DSM-5 code truncated to the given hierarchy depth.
 
    levels : 1    -> category only        ("D5_4.02.01" -> "D5_4")
             2    -> category + disorder  ("D5_4.02.01" -> "D5_4.02")
             None -> no truncation, return code unchanged (dsm_level='full')
    """
    def _single(c):
        if pd.isna(c) or str(c).strip() == '':
            return None
        c = str(c).strip()
        if levels is None:
            return c
        parts = c.split('.')
        return '.'.join(parts[:levels]) if len(parts) >= levels else parts[0]
 
    if isinstance(code, (list, np.ndarray)):
        return [_single(c) for c in code]
 
    if pd.isna(code) or str(code).strip() == '':
        return None
 
    code = str(code).strip()
    if ' ' in code:
        tokens = []
        for c in code.split():
            single = _single(c)
            tokens.append(single if single is not None else '')
        return ' '.join(tokens)
 
    return _single(code)

class MultiHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer(sparse_output=True)

    def fit(self, x, y=None):
        # X is 2D array-like from ColumnTransformer
        self.mlb.fit(x.ravel())
        return self

    def transform(self, x):
        return self.mlb.transform(x.ravel())


def split_tokens(s):
    return s.split()

def preprocess_time_columns(df, time_columns):
    df = df.copy()

    for col in time_columns:
        if df[col].isna().any():
            print(f"column {col} has NaN values")

        vals = df[col].fillna(0).astype("float32")
        # we apply log transformation for time features..
        if col in time_columns:
            # optional safety guard if negative values are possible
            if (vals < 0).any():
                print(f"WARNING: {col} has negative values before log1p")
            vals = np.log1p(vals)

        df[col] = vals

    return df


def majority_baseline(task, train_df, val_df, test_df=None):
    target_label = task.outcome
    print(f"Majority baseline experiment started for target label: {target_label}")
    # labels
    y_train, y_val = [(df[target_label].to_numpy().ravel()) for df in [train_df, val_df]]

    # Find the majority class in the training set
    majority_class = train_df[target_label].mode()[0]
    print(f"Majority class: {majority_class}")
    class_counts = train_df[target_label].value_counts(normalize=True)
    majority_prob = class_counts[majority_class]
    print(f"Majority class: {majority_class} with probability: {majority_prob:.4f}")

    # Predict majority class for validation set
    val_preds = [majority_class] * len(val_df)
    # If test set is provided, predict for test set too
    val_probs = [majority_prob for _ in range(len(val_df))]

    split_scores = {
        "val": (val_preds,
                val_probs,
                y_val, val_df)
    }

    if test_df is not None:
        test_preds = [majority_class] * len(test_df)
        test_probs = [majority_prob for _ in range(len(test_df))]
        y_test_np = test_df[target_label].to_numpy().ravel()
        split_scores["test"] = (test_preds,
                                test_probs,
                                y_test_np, test_df)

    evaluator = Evaluator()

    # Add all available splits to the evaluator/results
    evaluator.add_results(
        learner="majority",
        input_config=None,
        split_scores=split_scores,
        task=task,
        event_types=train_df[TemporalEvent.TYPE.value].unique()
    )
    return evaluator.results, evaluator.predictions


def truncate_atc(x, atc_chars):
    """
    Truncate ATC medication code(s) to atc_chars length.
    """
    def _truncate_token(tok):
        return tok if tok in NONE_VALUES else tok[:atc_chars]
 
    if isinstance(x, str):
        return _truncate_token(x)
    else:  # list/tuple of codes
        return " ".join(_truncate_token(m) for m in x)


def prepare_columns_to_preprocess(data, input_config, outcome_col, is_baseline=True, finetune_bert=False, atc_chars=4, dsm_level='disorder'):
        result = data.copy()
        final_columns = []
        symptom_columns = [] 
        
        for feature in input_config:
            if feature in FEATURE_GROUP_COLUMNS:
                final_columns.extend(FEATURE_GROUP_COLUMNS[feature])

        # ── Special cases that need extra handling beyond column selection ────────

        if Features.BERT in input_config:
            if finetune_bert:
                final_columns.extend([ClinicalText.TEXT.value])
            else:
                final_columns.extend([c for c in data.columns if c.startswith('bert_feature')])

        if Features.TFIDF in input_config:
            final_columns.extend([c for c in data.columns if c.startswith('tfidf_feature')])

        if Features.SYMPTOM_PREDICTIONS in input_config:
            symptom_cat_columns = [
                c for c in data.columns
                if c.startswith('symptom_pred_cat_') and not c.endswith('_slope')
            ]
            # numeric: slope columns
            symptom_slope_columns = [
                c for c in data.columns
                if c.startswith('symptom_pred_cat_') and c.endswith('_slope')
            ]

            final_columns.extend(symptom_cat_columns)
            final_columns.extend(symptom_slope_columns)

            symptom_columns = symptom_cat_columns
        
        if Features.ACTIVE_MEDICATIONS in input_config:
            result[Medication.ATC_CODE.value] = result[Medication.BASE_ATC_CODE.value].apply(
                lambda x: truncate_atc(x, atc_chars)
            ).astype(str)

            final_columns.extend([Medication.ATC_CODE.value])
           
        if Features.PAST_MEDICATIONS in input_config:
            if is_baseline:
                 result[ListEventFeatures.HISTORICAL_MEDICATIONS.value] = result[ListEventFeatures.HISTORICAL_MEDICATIONS.value].apply(
                    lambda x: truncate_atc(x, atc_chars)).astype(str)
            final_columns.extend([ListEventFeatures.HISTORICAL_MEDICATIONS.value])
        
        if Features.DIAGNOSIS in input_config:
            dsm_levels = {'category': 1, 'disorder': 2, 'full': None}
            for col in [TemporalEvent.ACTIVE_DIAGNOSIS_MAIN.value,
                        TemporalEvent.ACTIVE_DIAGNOSIS_SECONDARY.value]:
                levels = dsm_levels[dsm_level]
                if levels is not None:
                    result[col] = result[col].apply(lambda x: extract_dsm_level(x, levels))
            

        if (Features.PAST_DIAGNOSIS in input_config) :
            dsm_levels = {'category': 1, 'disorder': 2, 'full': None}
            levels = dsm_levels[dsm_level]
            if is_baseline:
                for col in [ListEventFeatures.HISTORICAL_DIAGNOSIS_MAIN.value,
                        ListEventFeatures.HISTORICAL_DIAGNOSIS_SECONDARY.value]:
                    if levels is not None:
                        result[col] = result[col].apply(lambda x: extract_dsm_level(x, levels))
 
                    result[col] = result[col].apply(
                        lambda x: " ".join(t or "" for t in x) if isinstance(x, list) else (x or "")
                    )
                final_columns.extend([ListEventFeatures.HISTORICAL_DIAGNOSIS_MAIN.value,
                                   ListEventFeatures.HISTORICAL_DIAGNOSIS_SECONDARY.value])
            
          
        if Features.LAB_RESULTS in input_config:
            result[TemporalEvent.LAB_RESULTS.value] = result[TemporalEvent.LAB_RESULTS.value].fillna('unknown')
            result[TemporalEvent.RECENT_LAB_MEASURE.value] = result[TemporalEvent.RECENT_LAB_MEASURE.value].fillna('unknown')

        if Features.APPOINTMENT in input_config:
            result[TemporalEvent.APPOINTMENT_DURATION.value] = (
                pd.to_timedelta(data[TemporalEvent.APPOINTMENT_DURATION.value], errors='coerce')
                .fillna(pd.Timedelta(0))
                .dt.total_seconds()
                .astype(float)
            )

        non_numeric_columns = [c for c in NON_NUMERIC_COLUMNS if c in final_columns] + symptom_columns
        numeric_columns = [c for c in final_columns if c not in non_numeric_columns]
        result = preprocess_time_columns(result, ALL_TIME_COLUMNS)
        
        columns_to_keep= [outcome_col, TemporalEvent.TRAJECTORY_ID.value, TemporalEvent.TEXT_ID.value,
                                    TemporalEvent.PATIENT_ID.value, TemporalEvent.DATE.value,
                                     TemporalEvent.ID.value, 
                                    TemporalEvent.DAYS_SINCE_MED_EVENT.value]
        for col in [
            TemporalEvent.TYPE.value,
            TemporalEvent.TRIGGER_MED_ACTION_TYPE.value,
            TemporalEvent.DURING_ADMISSION.value,
            TemporalEvent.TIME_SINCE_LAST_EVENT.value,
        ]:
            if col not in final_columns:
                columns_to_keep.append(col)

        categorical_columns_to_fill = [c for c in (CATEGORICAL_COLUMNS + symptom_columns) if c in final_columns]
        if categorical_columns_to_fill:
            result[categorical_columns_to_fill] = result[categorical_columns_to_fill].fillna(MISSING_CATEGORY_FILL)
            result[categorical_columns_to_fill] = result[categorical_columns_to_fill].apply(
                lambda s: s.mask(s.astype(str).str.strip() == "", MISSING_CATEGORY_FILL)
            )

        return result[final_columns + columns_to_keep], final_columns, numeric_columns

def persistence_baseline(task, train_df, val_df, test_df=None):
    """
    Predicts a primary-med drug change if at least one drug change
    (switch, stop, or add) occurred within the patient's 42-day historical
    lookback window. 
    """
    target_label = task.outcome
    print(f"Persistence Baseline started for: {target_label}")

    # Mirror the annotation: drug changes = switch / stop / add (no dose changes)
    DRUG_CHANGE_ACTIONS = {"main_switch", "main_stopped", "main_added"}

    def calculate_persistence(df):
        # 1. Ensure Date is datetime and sort chronologically
        df_work = df.copy()
        df_work[TemporalEvent.DATE.value] = pd.to_datetime(df_work[TemporalEvent.DATE.value])
        df_work = df_work.sort_values([TemporalEvent.TRAJECTORY_ID.value, TemporalEvent.DATE.value])

        preds = pd.Series(0, index=df_work.index)
        window_delta = pd.Timedelta(days=42)

        # 2. Iterate by trajectory to maintain patient silos
        for traj_id, group in df_work.groupby(TemporalEvent.TRAJECTORY_ID.value):
            dates   = group[TemporalEvent.DATE.value].values
            actions = group[TemporalEvent.TRIGGER_MED_ACTION_TYPE.value].values

            # A historical "change" = any drug-change action, matching the label
            changes = np.isin(actions, list(DRUG_CHANGE_ACTIONS)).astype(int)

            indices = group.index

            for i in range(len(group)):
                current_date = dates[i]

                # Lookback window: [current_date - 42 days, current_date)
                # strict '<' on current_date to avoid leaking the current event
                mask = (dates < current_date) & (dates >= (current_date - window_delta))

                if np.any(changes[mask] == 1):
                    preds.loc[indices[i]] = 1

        return preds.reindex(df.index).values

    # Generate predictions for each split
    train_preds = calculate_persistence(train_df)
    val_preds = calculate_persistence(val_df)

    train_preds_prob = train_preds.astype(float)
    val_preds_prob = val_preds.astype(float)

    y_train = train_df[target_label].to_numpy().ravel()
    y_val = val_df[target_label].to_numpy().ravel()

    split_scores = {
        "train": (train_preds, train_preds_prob, y_train, train_df),
        "val": (val_preds, val_preds_prob, y_val, val_df)
    }

    if test_df is not None and len(test_df) > 0:
        test_preds = calculate_persistence(test_df)
        test_preds_prob = test_preds.astype(float)
        y_test = test_df[target_label].to_numpy().ravel()
        split_scores["test"] = (test_preds, test_preds_prob, y_test, test_df)

    evaluator = Evaluator()
    evaluator.add_results(
        learner='persistence-baseline',
        input_config=None,
        split_scores=split_scores,
        task=task,
        event_types=train_df[TemporalEvent.TYPE.value].unique()
    )

    return evaluator.results, evaluator.predictions

def baseline_training(task, input_config, train_df, val_df,
                       final_columns, numeric_columns, full_model_path=None,
                       learner='linear-regression', test_df=None,
                       splitter=None, use_cv_ablation=False, n_cv_folds=5,
                       trigger_action_types=None):
    target_label = task.outcome
    print(f"Baseline experiment started for target label: {target_label}")

    y_train, y_val = [(df[target_label].to_numpy().ravel()) for df in [train_df, val_df]]
    x_train = train_df[final_columns]
    x_val = val_df[final_columns]
    x_test = test_df[final_columns] if test_df is not None else None

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)

    X = pd.concat([x_train, x_val], axis=0)
    y = np.concatenate([y_train, y_val])

    # ── build cv strategy: patient-grouped stratified k-fold (ablation),
    #    or predefined single train/val split (default, unchanged) ──────────
    if use_cv_ablation:
        if splitter is None:
            raise ValueError("splitter must be provided when use_cv_ablation=True")
        print(f"[XGB] Using {n_cv_folds}-fold patient-level stratified CV "
              f"on train+val pool for ablation (leakage-safe)")

        combined_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
        X = X.reset_index(drop=True)  # keep X/y positionally aligned with combined_df

        folds = splitter.create_stratified_folds(
            combined_df, n_splits=n_cv_folds,
            trigger_action_types=trigger_action_types,
        )
        # folds give index *labels* into combined_df; since we reset_index above,
        # labels == positions, so this is directly usable as GridSearchCV's cv
        cv = [(f['train_idx'], f['val_idx']) for f in folds]

        assert len(X) == len(combined_df), \
            "X and combined_df length mismatch — index alignment broken"
        max_idx = max(max(tr.max(initial=-1), va.max(initial=-1)) for tr, va in
                       [(np.array(f['train_idx']), np.array(f['val_idx'])) for f in folds])
        assert max_idx < len(X), \
            f"fold index {max_idx} out of bounds for X of length {len(X)}"
    else:
        test_fold = np.r_[
            -np.ones(len(x_train), dtype=int),
            np.zeros(len(x_val), dtype=int)
        ]
        cv = PredefinedSplit(test_fold)

    event_types = train_df[TemporalEvent.TYPE.value].unique()
    symptom_columns = [
        col for col in final_columns
        if col.startswith('symptom_pred_cat_') and not col.endswith('_slope')
    ]

    active_numeric_columns     = [c for c in numeric_columns if c in final_columns]
    active_categorical_columns = active(CATEGORICAL_COLUMNS + symptom_columns, x_train)
    active_boolean_columns     = active(ALL_BOOLEAN_COLUMNS, x_train)

    already_handled = (
        active_numeric_columns +
        active_categorical_columns +
        active_boolean_columns +
        ([Medication.ATC_CODE.value] if Medication.ATC_CODE.value in final_columns else []) +
        ([ListEventFeatures.HISTORICAL_MEDICATIONS.value] if ListEventFeatures.HISTORICAL_MEDICATIONS.value in final_columns else []) +
        ([ListEventFeatures.HISTORICAL_DIAGNOSIS_MAIN.value] if ListEventFeatures.HISTORICAL_DIAGNOSIS_MAIN.value in final_columns else []) +
        ([ListEventFeatures.HISTORICAL_DIAGNOSIS_SECONDARY.value] if ListEventFeatures.HISTORICAL_DIAGNOSIS_SECONDARY.value in final_columns else [])
    )

    # everything else in final_columns passes through as-is
    active_passthrough_columns = [c for c in final_columns if c not in already_handled]

    transformers = []
    if active_numeric_columns:
        transformers.append(("num", StandardScaler(), active_numeric_columns))
    if active_boolean_columns:
        transformers.append((
            "bool",
            "passthrough",  # already 0/1, no scaling needed
            active_boolean_columns
        ))

    diag_main_col = TemporalEvent.ACTIVE_DIAGNOSIS_MAIN.value
    diag_sec_col  = TemporalEvent.ACTIVE_DIAGNOSIS_SECONDARY.value
    diag_pair_present = (
        diag_main_col in active_categorical_columns and
        diag_sec_col in active_categorical_columns
    )

    if diag_pair_present:
        shared_diag_categories = sorted(
            set(x_train[diag_main_col].astype(str).unique()) |
            set(x_train[diag_sec_col].astype(str).unique())
        )
        print(f"[XGB] shared diagnosis vocab size (main ∪ secondary): {len(shared_diag_categories)}")
        active_categorical_columns = [
            c for c in active_categorical_columns
            if c not in (diag_main_col, diag_sec_col)
        ]

    if diag_pair_present:
        transformers.append((
            "cat_diag_shared",
            OneHotEncoder(
                categories=[shared_diag_categories, shared_diag_categories],
                handle_unknown="ignore",
                sparse_output=True,
            ),
            [diag_main_col, diag_sec_col]
        ))

    print("baseline experiment: active categorical columns:", active_categorical_columns)
    if active_categorical_columns:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            active_categorical_columns
        ))

    if active_passthrough_columns:
        transformers.append(("passthrough", "passthrough", active_passthrough_columns))

    if Medication.ATC_CODE.value in final_columns:
        med_vectorizer = CountVectorizer(
            tokenizer=split_tokens,
            token_pattern=None,
            min_df=50,
            binary=True
        )
        transformers.append(("med", med_vectorizer, Medication.ATC_CODE.value))

    if ListEventFeatures.HISTORICAL_MEDICATIONS.value in final_columns:
        transformers.append(("past_med", CountVectorizer(
            tokenizer=split_tokens,
            token_pattern=None,
            min_df=50
        ), ListEventFeatures.HISTORICAL_MEDICATIONS.value))

    if ListEventFeatures.HISTORICAL_DIAGNOSIS_MAIN.value in final_columns:
        transformers.append(("past_dx_main", CountVectorizer(
            tokenizer=split_tokens,
            token_pattern=None,
            min_df=20
        ), ListEventFeatures.HISTORICAL_DIAGNOSIS_MAIN.value))

    if ListEventFeatures.HISTORICAL_DIAGNOSIS_SECONDARY.value in final_columns:
        transformers.append(("past_dx_sec", CountVectorizer(
            tokenizer=split_tokens,
            token_pattern=None,
            min_df=20
        ), ListEventFeatures.HISTORICAL_DIAGNOSIS_SECONDARY.value))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    param_grid = {
        'model__n_estimators':  [100, 200, 300],
        'model__max_depth':     [2, 3, 5, 7],
        'model__learning_rate': [0.01, 0.1, 0.2],
    }

    is_binary = pd.Series(y_train).dropna().nunique() == 2
    if is_binary:
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        objective   = 'binary:logistic'
        eval_metric = 'logloss'
    else:
        scale_pos_weight = 1.0
        objective   = 'multi:softprob'
        eval_metric = 'mlogloss'

    print(f"training will start now: {x_val.columns}")
    nan_cols = x_val.columns[x_val.isna().any()]
    print(f"Columns with NaNs: {list(nan_cols)}")
    if learner == 'svm':
        pipeline = Pipeline([
            ("scaler", preprocessor),
            ("model", SVC(probability=True))
        ])
        param_grid = {
            'model__C': [0.01, 0.1, 1],
            'model__kernel': ['linear', 'rbf'],
        }
    else:
        pipeline = Pipeline([
            ("scaler", preprocessor),
            ("model", XGBClassifier(
                objective=objective,
                eval_metric=eval_metric,
                num_class=pd.Series(y_train).nunique() if not is_binary else None
            ))
        ])

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1
    )

    print("Model pipeline is set, grid.fit() is started..", x_train.shape)
    grid.fit(X, y)
    print("Model training is completed...")
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV score: {grid.best_score_:.4f}"
          + (f" (mean over {n_cv_folds} folds)" if use_cv_ablation else " (val score)"))

    cv_mean_score, cv_std_score = np.nan, np.nan
    if use_cv_ablation:
        cv_mean_score = grid.cv_results_['mean_test_score'][grid.best_index_]  # same as grid.best_score_
        cv_std_score  = grid.cv_results_['std_test_score'][grid.best_index_]
        print(f"[XGB] CV mean AUC: {cv_mean_score:.4f}  CV std: {cv_std_score:.4f} "
              f"(over {n_cv_folds} folds, winning hyperparams)")

    best_pipeline = clone(pipeline).set_params(**grid.best_params_)

    # ── final fit: ALWAYS train-only, val held out 
    best_pipeline.fit(x_train, y_train)

    fitted_preprocessor = best_pipeline.named_steps["scaler"]
    X_train_transformed = fitted_preprocessor.transform(x_train)
    print(f"[XGB] transformed feature matrix shape: {X_train_transformed.shape}")

    feature_names = fitted_preprocessor.get_feature_names_out()
    print(f"[XGB] total features after encoding: {len(feature_names)}")

    #################### EVALUATION #################################
    evaluator = Evaluator()

    val_preds  = best_pipeline.predict(x_val)
    val_proba  = best_pipeline.predict_proba(x_val)
    val_preds_prob = val_proba[:, 1] if val_proba.shape[1] == 2 else val_proba

    train_preds = best_pipeline.predict(x_train)
    train_proba = best_pipeline.predict_proba(x_train)
    train_preds_prob = train_proba[:, 1] if train_proba.shape[1] == 2 else train_proba

    split_scores = {
        "train": (train_preds, train_preds_prob, y_train, train_df),
        "val":   (val_preds,   val_preds_prob,   y_val,   val_df),
    }

    if test_df is not None and len(test_df) > 0:
        y_test_np = le.transform(test_df[target_label].to_numpy().ravel())
        test_preds = best_pipeline.predict(x_test)
        test_proba = best_pipeline.predict_proba(x_test)
        test_preds_prob = test_proba[:, 1] if test_proba.shape[1] == 2 else test_proba

        split_scores["test"] = (test_preds, test_preds_prob, y_test_np, test_df)

    evaluator.add_results(
        learner=learner,
        input_config=input_config,
        split_scores=split_scores,
        task=task,
        event_types=event_types,
    )

    # attach CV summary (NaN if use_cv_ablation=False) so ablation tables can
    # report CV robustness alongside fixed val/test performance in one row
    evaluator.results['cv_mean_auc'] = cv_mean_score
    evaluator.results['cv_std_auc']  = cv_std_score
    evaluator.results['n_cv_folds']  = n_cv_folds if use_cv_ablation else np.nan

    if full_model_path is None:
        full_model_path = model_path / f'best_{learner}_{target_label}_pipeline.joblib'
    joblib.dump(best_pipeline, full_model_path)

    return evaluator.results, evaluator.predictions

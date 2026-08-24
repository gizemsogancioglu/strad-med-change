import collections
from datetime import timedelta
import json

from pathlib import Path
import numpy as np
import pandas as pd
from source.data_processing.data_reader import PROJECT_ROOT
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from source.data_processing.data_model import ClinicalText, EventType, Trajectory, TemporalEvent
from source.models.base_model import Task


# Only categories 0, 2, 3, 4, 7, 8 are predicted (6 total)
SYMPTOM_CATEGORIES = {
    0: ("depressed mood", "depressed mood, gloominess, despondence, or hopelessness"),
    2: ("suicidal tendency", "feeling life not worth living, desire to die, consideration/plans for suicide"),
    3: ("insomnia", "issues with falling asleep, staying asleep, or waking up prematurely"),
    4: ("interests and activities", "anhedonia, lack of interests or motivation, reduced activity level"),
    7: ("anxiety (psychological)", "anxiety, being scared, feeling unsafe, panic, worry, irritability"),
    8: ("somatic", "gastro-intestinal issues, appetite changes, trembling, aches, tiredness"),
}

PREDICTED_SYMPTOM_CATS = [0, 2, 3, 4, 7, 8]

# Canonical string labels used throughout — no magic numbers
VALID_LABELS = {"established", "worsened", "improved", "no_mention"}

# Mapping from raw Qwen integer output to canonical string label
INT_TO_LABEL = {
    0: "established",
    -1: "worsened",
    1: "improved",
    100: "no_mention",
    13: "no_mention",  # noise/unknown value from Qwen
}

# Kept for backward compatibility / external references only
SYMPTOM_PROGRESSION_LABELS = {
    "established": 0,
    "worsened": -1,
    "improved": 1,
    "no_mention": 100,
}

SYMPTOM_PROGRESSION_LABELS_INVERSE = {
    value: key for key, value in SYMPTOM_PROGRESSION_LABELS.items()
}

# Mapping from canonical string label to model training index
MODEL_LABEL_MAPPING = {
    "established": 0,
    "worsened": 1,
    "improved": 2,
    "no_mention": 3,
}

MODEL_LABEL_INVERSE = {v: k for k, v in MODEL_LABEL_MAPPING.items()}


def merge_symptom_predictions(data, symptom_predictions_df: pd.DataFrame) -> pd.DataFrame:
    if symptom_predictions_df is None or symptom_predictions_df.empty:
        return data

    data = data.copy()
    text_id_col = ClinicalText.NOTE_ID.value
    if text_id_col not in data.columns:
        return data

    note_mask = data[TemporalEvent.TYPE.value] == EventType.NOTE.value
    if not note_mask.any():
        return data

    data[text_id_col] = data[text_id_col].astype('string')
    symptom_predictions_df = symptom_predictions_df.copy()
    symptom_predictions_df[text_id_col] = symptom_predictions_df[text_id_col].astype('string')

    note_ids = data.loc[note_mask, text_id_col].dropna().unique()
    symptom_predictions_df = symptom_predictions_df[symptom_predictions_df[text_id_col].isin(note_ids)]

    data = data.merge(symptom_predictions_df, on=text_id_col, how='left')
    symptom_columns = [col for col in data.columns if col.startswith('symptom_pred_cat_')]
    if symptom_columns:
        # Non-note rows get no_mention, NaN notes (no prediction available) also get no_mention
        for col in symptom_columns:
            data[col] = (
                pd.to_numeric(data[col], errors='coerce')
                .map(lambda x: INT_TO_LABEL.get(int(x), "no_mention") if pd.notna(x) else "no_mention")
            )
        data.loc[~note_mask, symptom_columns] = "no_mention"

    return data
 
def parse_symptom_prediction_jsonl(jsonl_path: str | Path) -> pd.DataFrame:
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Symptom prediction JSONL file not found: {jsonl_path}")

    rows = []
    with jsonl_path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            category = record.get("category")
            prediction = record.get("prediction", {})

            for note_id, score in prediction.items():
                rows.append({
                    ClinicalText.NOTE_ID.value: str(note_id),
                    "category": int(category) if category is not None else None,
                    "symptom_prediction": score,
                })

    return pd.DataFrame(rows)

def load_symptom_predictions(jsonl_path: str | Path | None = None) -> pd.DataFrame:
    if jsonl_path is None:
        jsonl_path = PROJECT_ROOT / "Qwen3-30B-A3B_remapped.jsonl"
    else:
        jsonl_path = Path(jsonl_path)

    df = parse_symptom_prediction_jsonl(jsonl_path)
    if df.empty:
        print(f"No symptom predictions found in {jsonl_path}")
        return df

    df = df.pivot_table(
        index=ClinicalText.NOTE_ID.value,
        columns="category",
        values="symptom_prediction",
        aggfunc="first",
    )
    df.columns = [f"symptom_pred_cat_{int(c)}" for c in df.columns]
    df = df.reset_index()
    return df

def _normalize_label_cell(cell) -> list:
    """
    Convert a raw prediction cell into a list of canonical string labels.

    Handles:
    - str:   "100.0", "no_mention", "-1.0", "worsened", "" etc.
    - int/float: raw Qwen output values
    - list:  list of the above
    - dict:  {note_id: raw_value, ...}
    """
    if isinstance(cell, str):
        if cell == '':
            return []
        if cell in VALID_LABELS:
            return [cell]
        try:
            return [INT_TO_LABEL.get(int(float(cell)), "no_mention")]
        except (ValueError, TypeError):
            return []

    if isinstance(cell, dict):
        return [INT_TO_LABEL.get(int(v), "no_mention")
                for v in cell.values()
                if not pd.isna(v) and int(v) in INT_TO_LABEL]

    if isinstance(cell, list):
        result = []
        for item in cell:
            result.extend(_normalize_label_cell(item))
        return result

    if isinstance(cell, (int, float)) and not pd.isna(cell):
        return [INT_TO_LABEL.get(int(cell), "no_mention")]

    return []


def _resolve_counts(worsened_count: int, improved_count: int, established_count: int) -> str:
    """Resolve window label counts using clinical dominance rule."""
    if worsened_count or improved_count:
        if worsened_count and improved_count:
            return "worsened" if worsened_count >= improved_count else "improved"
        return "worsened" if worsened_count else "improved"
    if established_count:
        return "established"
    return "no_mention"


def _resolve_window_labels(labels: list) -> str:
    """
    Resolve multiple progression labels within a time window using a dominance rule.
    This dominance rule is a CLINICAL DECISION RULE that prioritizes more actionable
    findings over others. Priority reflects clinical significance:
    - Worsened/Improved indicate change (more clinically relevant than no change)
    - Among changes, worsened indicates treatment inefficacy (highest priority)
    - Established indicates status quo
    - No mention is lowest priority (absence of information)
    """
    if not labels:
        return "no_mention"

    counts = collections.Counter(labels)
    return _resolve_counts(counts["worsened"], counts["improved"], counts["established"])


def _collect_labels_in_window(event_date, window_end, cell) -> list:
    """
    Extract all valid canonical string labels from a cell within the time window.
    Only used by label_symptom_progression (legacy row-wise path).
    """
    labels = []

    if isinstance(cell, dict):
        for key, value in cell.items():
            try:
                candidate_date = pd.to_datetime(key)
            except (ValueError, TypeError):
                continue
            if event_date < candidate_date <= window_end:
                normalized = _normalize_label_cell(value)
                labels.extend(normalized)

    elif isinstance(cell, (list, str, int, float)):
        labels.extend(_normalize_label_cell(cell))

    return labels


def compute_symptom_progression_targets(data: pd.DataFrame, symptom_categories: list, window_days: int = 42) -> pd.DataFrame:
    """
    For each event, compute the dominant symptom progression label over the
    next `window_days` days within the same trajectory.

    Outcome columns are stored as canonical string labels (e.g. "worsened", "no_mention").
    """
    if data is None or data.empty:
        return data

    date_col = TemporalEvent.DATE.value
    trajectory_col = Trajectory.ID.value

    if date_col not in data.columns or trajectory_col not in data.columns:
        return data

    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce")

    for symptom_cat in symptom_categories:
        pred_col = f"symptom_pred_cat_{symptom_cat}"
        outcome_col = f"symptom_progression_cat_{symptom_cat}"

        if pred_col not in data.columns:
            data[outcome_col] = "no_mention"
            continue

        targets = np.full(len(data), "no_mention", dtype=object)

        for _, group in data.groupby(trajectory_col, sort=False):
            if group.empty:
                continue

            group = group.sort_values(date_col)
            dates = group[date_col].to_numpy(dtype="datetime64[ns]")
            raw_cells = group[pred_col].tolist()
            original_positions = data.index.get_indexer(group.index.to_numpy())

            if len(dates) <= 1:
                continue

            label_lists = [_normalize_label_cell(cell) for cell in raw_cells]

            worsened_counts = np.fromiter(
                (labels.count("worsened") for labels in label_lists),
                dtype=np.int32, count=len(label_lists))
            improved_counts = np.fromiter(
                (labels.count("improved") for labels in label_lists),
                dtype=np.int32, count=len(label_lists))
            established_counts = np.fromiter(
                (labels.count("established") for labels in label_lists),
                dtype=np.int32, count=len(label_lists))

            prefix_worsened = np.concatenate([[0], np.cumsum(worsened_counts)])
            prefix_improved = np.concatenate([[0], np.cumsum(improved_counts)])
            prefix_established = np.concatenate([[0], np.cumsum(established_counts)])

            for idx, event_date in enumerate(dates):
                window_end = event_date + np.timedelta64(window_days, "D")
                end_idx = np.searchsorted(dates, window_end, side="right") - 1
                if end_idx <= idx:
                    continue

                worsened_sum = int(prefix_worsened[end_idx + 1] - prefix_worsened[idx + 1])
                improved_sum = int(prefix_improved[end_idx + 1] - prefix_improved[idx + 1])
                established_sum = int(prefix_established[end_idx + 1] - prefix_established[idx + 1])
                targets[original_positions[idx]] = _resolve_counts(worsened_sum, improved_sum, established_sum)

        data[outcome_col] = targets

    return data


def label_symptom_progression(event_date, trajectory_id, outcome_data, window_days=42, outcome_column="symptom_progression") -> str:
    """
    Extract symptom progression label for a given event within a time window.
    Returns a canonical string label: "worsened", "improved", "established", or "no_mention".
    """
    event_date = pd.to_datetime(event_date)
    window_end = event_date + timedelta(days=window_days)

    trajectory_rows = outcome_data[outcome_data[Trajectory.ID.value] == trajectory_id]
    if trajectory_rows.empty or outcome_column not in trajectory_rows.columns:
        return "no_mention"

    labels = []
    for cell in trajectory_rows[outcome_column].tolist():
        labels.extend(_collect_labels_in_window(event_date, window_end, cell))

    return _resolve_window_labels(labels)


class SymptomProgressionPrediction(Task):
    def __init__(self, name="symptom_progression_prediction", task="classification",
                 outcome='symptom_progression', num_labels=4):
        super().__init__(name=name, task=task, outcome=outcome, num_labels=num_labels)

    def set_labels(self, data, outcome_data=None, window_days=42):
        data[self.outcome] = data.apply(
            lambda row: MODEL_LABEL_MAPPING.get(
                label_symptom_progression(
                    row[TemporalEvent.DATE.value],
                    row[Trajectory.ID.value],
                    outcome_data,
                    window_days=window_days,
                ),
                3  # default to no_mention index
            ),
            axis=1,
        )
        return data

    def evaluate_prob(self, labels, preds_prob):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if isinstance(preds_prob, torch.Tensor):
            preds_prob = preds_prob.detach().cpu().numpy()
        labels = np.array(labels).ravel()
        preds_prob = np.array(preds_prob)
        if preds_prob.ndim == 1:
            return roc_auc_score(labels, preds_prob)
        return roc_auc_score(labels, preds_prob, multi_class='ovr', average='macro')

    def evaluate(self, labels, preds):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        labels = np.array(labels).ravel()
        preds = np.array(preds).ravel()
        return balanced_accuracy_score(labels, preds)

    def get_original_label(self, model_label: int) -> str:
        """Convert model training index (0-3) back to canonical string label."""
        return MODEL_LABEL_INVERSE.get(model_label, "no_mention")
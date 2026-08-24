import seaborn as sns
import matplotlib.pyplot as plt
from source.data_processing.data_model import *
from source.data_processing.data_reader import PROJECT_ROOT, results_path
import pandas as pd

from source.models.base_model import DatasetSplitter, extract_dsm_level, prepare_columns_to_preprocess
from source.models.medication_change_prediction import MedicationChangePrediction
from source.models.model_cfgs.feature_config import PRIMARY_TRIGGER_ACTIONS
from source.models.model_cfgs.feature_ablation_configs import FULL_CONFIG, _base_config

mapping = {
    'main_added': 'start',
    'main_switch': 'switch',
    'main_stopped': 'stop',
    'main_dose_increase': 'dose increase',
    'main_dose_decrease': 'dose decrease'
}

def add_bar_labels(ax, padding=3, fontsize=12):
    """
    Annotate all bars by placing the value **above the bar**.
    Works for very small and very tall bars.
    """
    for container in ax.containers:
        ax.bar_label(container, padding=padding, fontsize=fontsize)


def load_temporal_data(temp_parquet_path):

    print("Loading event-level dataset for preprocessing....")
    # ------------------------
    # Load main data
    # ------------------------
    data = pd.read_parquet(temp_parquet_path)
    # print("Loaded main data columns:", data.columns)
    print(f"Shape: {data.shape}")

    # filter out pharmaco events
    filtered_events =  [EventType.PHARMACOGENETIC_TEST.value,
                           EventType.ADMISSION_START.value,
                           EventType.ADMISSION_STOP.value,
                           EventType.LAB_TEST.value,
                           EventType.DIAGNOSIS.value,
                           EventType.APPOINTMENT.value,
                           EventType.NOTE.value
                           ]
    #filtered_events = []
    data = data[~data[TemporalEvent.TYPE.value].isin(filtered_events)]
    print(f"Shape after filtering nurse notes: {data.shape}")
    return data

def make_split_comparison_latex(val_df, test_df, outcome, overall_df,
                                 event_type_col=TemporalEvent.TRIGGER_MED_ACTION_TYPE.value,
                                 trajectory_id_col=Trajectory.ID.value,
                                 patient_id_col=Patient.ID.value,
                                 gender_col=TemporalEvent.PATIENT_GENDER.value,
                                 age_col=TemporalEvent.PATIENT_AGE.value,
                                 event_date_col=TemporalEvent.DATE.value,
                                 diagnosis_main_col=TemporalEvent.ACTIVE_DIAGNOSIS_MAIN.value,
                                 diagnosis_secondary_col=TemporalEvent.ACTIVE_DIAGNOSIS_SECONDARY.value,
                                 caption="Dataset, Validation, and Test Set Statistics",
                                 label="tab:val_test_stats"):
    """
    Build a LaTeX table comparing overall / val / test splits side by side:
    trajectory-level stats (duration, #events, diagnosis at start),
    patient-level stats (gender, age), event-type distribution, and outcome rates.
    """
    top_n_dsm = 10
    def get_first_event_diagnoses(df):
        """Sort by (trajectory, date), take first event per trajectory,
        return level-truncated main/secondary diagnosis series."""
        dates = pd.to_datetime(df[event_date_col])
        df_sorted = df.assign(**{event_date_col: dates}).sort_values([trajectory_id_col, event_date_col])
        first_events = df_sorted.groupby(trajectory_id_col).first()

        diag_main = first_events[diagnosis_main_col].apply(lambda c: extract_dsm_level(c, 2))
        diag_secondary = first_events[diagnosis_secondary_col].apply(lambda c: extract_dsm_level(c, 2))
        return diag_main, diag_secondary


    def stats(df):
        n_traj = df[trajectory_id_col].nunique()
        n_patients = df[patient_id_col].nunique()
        n_events = len(df)

        event_counts = df[event_type_col].value_counts()
        event_pct = df[event_type_col].value_counts(normalize=True) * 100

        pos_rate = df[outcome].mean() * 100
        traj_pos = df.groupby(trajectory_id_col)[outcome].max()
        n_traj_pos = int(traj_pos.sum())
        traj_pos_pct = 100 * n_traj_pos / n_traj

        per_event_pos = {}
        for et, group in df.groupby(event_type_col):
            per_event_pos[et] = (group[outcome].mean() * 100, len(group))

        # --- Trajectory-level: number of events per trajectory ---
        n_events_per_traj = df.groupby(trajectory_id_col).size()
        n_events_mean, n_events_std = n_events_per_traj.mean(), n_events_per_traj.std()

        # --- Trajectory-level: duration = max(date) - min(date) per trajectory ---
        dates = pd.to_datetime(df[event_date_col])
        traj_dates = dates.groupby(df[trajectory_id_col])
        traj_duration_days = (traj_dates.max() - traj_dates.min()).dt.days
        duration_mean, duration_std = traj_duration_days.mean(), traj_duration_days.std()

        # --- Trajectory-level: DSM diagnosis (main + secondary) at first event ---
        diag_main, diag_secondary = get_first_event_diagnoses(df)
        diag_main_counts = diag_main.value_counts(dropna=True)
        diag_secondary_counts = diag_secondary.value_counts(dropna=True)

        # --- Patient-level: gender ---
        patients_unique = df.drop_duplicates(subset=[patient_id_col])
        gender_counts = patients_unique[gender_col].value_counts()
        gender_pct = patients_unique[gender_col].value_counts(normalize=True) * 100

        # --- Patient-level: age ---
        age_mean, age_std = None, None
        age_mean = patients_unique[age_col].mean()
        age_std = patients_unique[age_col].std()
        age_max = patients_unique[age_col].max()
        age_min = patients_unique[age_col].min()

        return dict(
            n_traj=n_traj, n_patients=n_patients, n_events=n_events,
            event_counts=event_counts, event_pct=event_pct,
            pos_rate=pos_rate, n_traj_pos=n_traj_pos, traj_pos_pct=traj_pos_pct,
            per_event_pos=per_event_pos,
            gender_counts=gender_counts, gender_pct=gender_pct,
            n_events_mean=n_events_mean, n_events_std=n_events_std,
            duration_mean=duration_mean, duration_std=duration_std,
            diag_main_counts=diag_main_counts,
            diag_secondary_counts=diag_secondary_counts,
            age_mean=age_mean, age_std=age_std, age_max=age_max,age_min=age_min,
        )

    o = stats(overall_df)
    v = stats(val_df)
    t = stats(test_df)

    event_display = {
        "main_added": "start",
        "main_stopped": "stop",
        "main_switch": "switch",
        "main_dose_increase": "dose increase",
        "main_dose_decrease": "dose decrease",
    }
    event_order = [e for e in event_display
                   if e in o["event_counts"].index or e in v["event_counts"].index or e in t["event_counts"].index]

    all_genders = sorted(
        set(o["gender_counts"].index) | set(v["gender_counts"].index) | set(t["gender_counts"].index)
    )

    def fmt_mean_std(mean, std, suffix=""):
        if mean is None:
            return "--"
        return rf"{mean:.1f} $\pm$ {std:.1f}{suffix}"

    def diagnosis_rows(o_counts, v_counts, t_counts, top_n):
        """
        Restrict to the top `top_n` categories by overall count, group the
        rest into 'Other', and build LaTeX rows for overall/val/test.
        """
        o_total = o_counts.sum()
        v_total = v_counts.sum()
        t_total = t_counts.sum()

        top_categories = o_counts.sort_values(ascending=False).head(top_n).index.tolist()

        rows = []
        for d in top_categories:
            oc = o_counts.get(d, 0)
            vc = v_counts.get(d, 0)
            tc = t_counts.get(d, 0)
            op = 100 * oc / o_total if o_total else 0.0
            vp = 100 * vc / v_total if v_total else 0.0
            tp = 100 * tc / t_total if t_total else 0.0
            rows.append(
                rf"    \quad {d} & {oc} ({op:.1f}\%) & {vc} ({vp:.1f}\%) & {tc} ({tp:.1f}\%) \\"
            )

        # "Other" = everything not in top_categories
        o_other = o_total - o_counts.reindex(top_categories, fill_value=0).sum()
        v_other = v_total - v_counts.reindex(top_categories, fill_value=0).sum()
        t_other = t_total - t_counts.reindex(top_categories, fill_value=0).sum()
        if o_other > 0 or v_other > 0 or t_other > 0:
            op = 100 * o_other / o_total if o_total else 0.0
            vp = 100 * v_other / v_total if v_total else 0.0
            tp = 100 * t_other / t_total if t_total else 0.0
            rows.append(
                rf"    \quad Other & {o_other} ({op:.1f}\%) & {v_other} ({vp:.1f}\%) & {t_other} ({tp:.1f}\%) \\"
            )

        return rows
    print(o['age_max'], o['age_min'])
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"    \centering")
    lines.append(rf"    \caption{{{caption}}}")
    lines.append(r"    \begin{tabular}{l r r r}")
    lines.append(r"    \toprule")
    lines.append(r"    Statistic & Overall & Validation & Test \\")
    lines.append(r"    \midrule")

    lines.append(rf"    number of trajectories & {o['n_traj']} & {v['n_traj']} & {t['n_traj']} \\")
    lines.append(rf"    number of unique patients & {o['n_patients']} & {v['n_patients']} & {t['n_patients']} \\")
    lines.append(rf"    total number of medication events & {o['n_events']} & {v['n_events']} & {t['n_events']} \\")

    # --- Trajectory-level ---
    lines.append(r"    \textbf{Trajectory-level} & & & \\")
    lines.append(
        rf"    duration (mean$\pm$std) & {fmt_mean_std(o['duration_mean'], o['duration_std'], ' days')} & "
        rf"{fmt_mean_std(v['duration_mean'], v['duration_std'], ' days')} & "
        rf"{fmt_mean_std(t['duration_mean'], t['duration_std'], ' days')} \\"
    )
    lines.append(
        rf"    number of medication events (mean$\pm$std) & {fmt_mean_std(o['n_events_mean'], o['n_events_std'])} & "
        rf"{fmt_mean_std(v['n_events_mean'], v['n_events_std'])} & "
        rf"{fmt_mean_std(t['n_events_mean'], t['n_events_std'])} \\"
    )

    lines.append(rf"    Main DSM-5 diagnosis at start (top {top_n_dsm}) & & & \\")
    lines.extend(diagnosis_rows(
        o["diag_main_counts"], v["diag_main_counts"], t["diag_main_counts"], top_n_dsm
    ))

    lines.append(rf"    Secondary DSM-5 diagnosis at start (top {top_n_dsm}) & & & \\")
    lines.extend(diagnosis_rows(
        o["diag_secondary_counts"], v["diag_secondary_counts"], t["diag_secondary_counts"], top_n_dsm
    ))

    # --- Patient-level ---
    lines.append(r"    \textbf{Patient-level} & & & \\")
    for g in all_genders:
        oc, op = o["gender_counts"].get(g, 0), o["gender_pct"].get(g, 0.0)
        vc, vp = v["gender_counts"].get(g, 0), v["gender_pct"].get(g, 0.0)
        tc, tp = t["gender_counts"].get(g, 0), t["gender_pct"].get(g, 0.0)
        lines.append(rf"    {g} & {oc} ({op:.1f}\%) & {vc} ({vp:.1f}\%) & {tc} ({tp:.1f}\%) \\")
    lines.append(
        rf"    age (mean$\pm$std) & {fmt_mean_std(o['age_mean'], o['age_std'])} & "
        rf"{fmt_mean_std(v['age_mean'], v['age_std'])} & "
        rf"{fmt_mean_std(t['age_mean'], t['age_std'])} \\"
    )

    # --- Event-level ---
    lines.append(r"    \textbf{Event-level} & & & \\")
    for e in event_order:
        oc, op = o["event_counts"].get(e, 0), o["event_pct"].get(e, 0.0)
        vc, vp = v["event_counts"].get(e, 0), v["event_pct"].get(e, 0.0)
        tc, tp = t["event_counts"].get(e, 0), t["event_pct"].get(e, 0.0)
        name = event_display.get(e, e)
        lines.append(
            rf"    {name} & {oc} ({op:.1f}\%) & {vc} ({vp:.1f}\%) & {tc} ({tp:.1f}\%) \\"
        )

    # --- Outcome ---
    lines.append(rf"    \textbf{{Outcome: {outcome}}} & & & \\")
    lines.append(
        rf"    positive event rate & {o['pos_rate']:.2f}\% & {v['pos_rate']:.2f}\% & {t['pos_rate']:.2f}\% \\"
    )
    lines.append(
        rf"    trajectories with $\geq$1 positive event & "
        rf"{o['n_traj_pos']} / {o['n_traj']} ({o['traj_pos_pct']:.1f}\%) & "
        rf"{v['n_traj_pos']} / {v['n_traj']} ({v['traj_pos_pct']:.1f}\%) & "
        rf"{t['n_traj_pos']} / {t['n_traj']} ({t['traj_pos_pct']:.1f}\%) \\"
    )

    lines.append(r"    \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(rf"    \label{{{label}}}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)
    print(latex_str)
    return latex_str



def split_temporal_data(data, outcome):
    task = MedicationChangePrediction(outcome=outcome)
 
    splitter = DatasetSplitter(
        outcome_col=task.outcome,
        trajectory_col=Trajectory.ID.value,
        gender_col=Patient.GENDER.value,
        patient_col=Patient.ID.value,
    )
    med_mask = data[TemporalEvent.TYPE.value].isin([
        EventType.MEDICATION_START.value,
        EventType.MEDICATION_STOP.value,
        EventType.MEDICATION_SWITCH.value,
    ])
    data.loc[med_mask, TemporalEvent.TYPE.value] = \
        data.loc[med_mask, TemporalEvent.TRIGGER_MED_ACTION_TYPE.value]
 
    data = data[data[TemporalEvent.TRIGGER_MED_ACTION_TYPE.value].isin({
        "main_added",
        "main_switch",
        "main_dose_increase",
        "main_dose_decrease",
        "main_stopped",
    })]
    input_config = [
            Features.PATIENT,
            Features.DIAGNOSIS,
            Features.LAB_RESULTS,
            Features.ACTIVE_MEDICATIONS,
            Features.ADMISSION,
            Features.TRIGGER_MEDICATIONS,
            Features.EVENT_TYPE,
    ]
    data, final_columns, numeric_columns = prepare_columns_to_preprocess(
        data.copy(), input_config, task.outcome,
        is_baseline=False, finetune_bert=False,
        atc_chars=_base_config.get('atc_chars', 5),
        dsm_level=_base_config.get('dsm_level', 'disorder'),
    )
 
    train_df, val_df, test_df = splitter.create_hs_sets(data, trigger_action_types=PRIMARY_TRIGGER_ACTIONS)
 
    print(f"  train={len(train_df)}, val={len(val_df)}, test={len(test_df) if test_df is not None else '—'}")
    print(f"  Outcome — train:\n{train_df[task.outcome].value_counts().to_string()}")
    print(f"  Outcome — val:\n{val_df[task.outcome].value_counts().to_string()}")
 
    return train_df, val_df, test_df

def create_latex_table(data, outcome):
    train_df, val_df, test_df = split_temporal_data(data, outcome)
    latex_str = make_split_comparison_latex(val_df, test_df, outcome, overall_df=data)
    make_split_comparison_latex(val_df, test_df, outcome, overall_df=data)

    return latex_str


def plot_age_distribution_by_gender(df):
    """
    Grouped by patient (one row per patient), plot the age
    distribution split by gender: histogram of counts with KDE overlay.
    """
    patient_col = TemporalEvent.PATIENT_ID.value
    gender_col = TemporalEvent.PATIENT_GENDER.value
    age_col = TemporalEvent.PATIENT_AGE.value  # adjust name if different

    # One row per patient (age/gender assumed constant per patient)
    patient_df = df[[patient_col, gender_col, age_col]].drop_duplicates(subset=patient_col)
    patient_df = patient_df.dropna(subset=[age_col, gender_col])

    colors = ['#D4A888', '#9AD2C3']

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(
        data=patient_df,
        x=age_col,
        hue=gender_col,
        bins=10,
        multiple="dodge",
        palette=colors,
        alpha=0.5,
        ax=ax,
    )

    ax.set_xlabel('Age', fontsize=16)
    ax.set_ylabel('Frequency', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

    # Style legend (seaborn auto-creates one via hue)
    legend = ax.get_legend()
    legend.set_title('Gender')
    plt.setp(legend.get_title(), fontsize=16)
    plt.setp(legend.get_texts(), fontsize=14)

    fig.tight_layout()
    fig.savefig(results_path / 'figures/age_distribution_by_gender.png', dpi=300)
    plt.close(fig)

def data_analyis(df, outcome_col):
    df_plot = df.copy()
    df_plot[outcome_col] = df_plot[outcome_col].map({0: 'No', 1: 'Yes'})
    colors = ['#D4A888', '#9AD2C3']  # green, orange

    df_plot[outcome_col].value_counts().sort_index().plot(kind='bar', color=colors)
    plt.xlabel(outcome_col, fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(results_path / f'figures/{outcome_col}_distribution.png')
    plt.close()

    # Seaborn style
    sns.set(style="whitegrid")

    # Single plot for event-level distribution
    fig, ax = plt.subplots(figsize=(16, 8))

    event_col = TemporalEvent.TRIGGER_MED_ACTION_TYPE.value
    df_plot[event_col] = df_plot[event_col].replace(mapping)

    event_order = ['start', 'switch', 'stop', 'dose increase', 'dose decrease']
    event_counts = df_plot.groupby([event_col, outcome_col]).size().unstack(fill_value=0)
    event_counts = event_counts.reindex([e for e in event_order if e in event_counts.index])
    ax = event_counts.plot(kind='bar', ax=ax, color=colors, width=0.6, legend=False)

    ax.set_xlabel('')
    ax.set_ylabel('Count', fontsize=20)
    wrapped_labels = [label.get_text().replace(" ", "\n") for label in ax.get_xticklabels()]
    ax.set_xticklabels(wrapped_labels, rotation=0, ha='center', fontsize=22)
    ax.tick_params(axis='y', labelsize=20)
    add_bar_labels(ax)
    for text in ax.texts:
        text.set_fontweight('bold')
        text.set_fontsize(18)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title='Change', loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=len(labels), frameon=False, fontsize=20, title_fontsize=20)

    fig.tight_layout(rect=[0, 0, 1, 0.92])  # leave space for legend on top

    # Save high-resolution figure
    fig.savefig(results_path / f'figures/{outcome_col}_combined.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
if __name__ == "__main__":
    # stats()
    data = load_temporal_data(temp_parquet_path=PROJECT_ROOT / 'results/temporal_data_med_only.parquet')
    # figure 2 in the paper (primary_med_drug_change_combined.png).
    data_analyis(data, EventLevelOutcomes.PRIMARY_MED_DRUG_CHANGE_OUTCOME.value)
    # age_distribution_by_gender.png
    #plot_age_distribution_by_gender(data)
   
    #create_latex_table(data, EventLevelOutcomes.PRIMARY_MED_DRUG_CHANGE_OUTCOME.value)

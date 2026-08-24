from itertools import combinations
from source.data_processing.data_model import EventLevelOutcomes, EventType, Features, TemporalEvent

FULL_CONFIG = [
            Features.TIMESTAMPS,
            Features.EVENT_TYPE,
            Features.PATIENT,
            Features.DIAGNOSIS,
            Features.ACTIVE_MEDICATIONS,
            Features.TRIGGER_MEDICATIONS,
            Features.LAB_RESULTS,
            Features.ADMISSION,
            # Features.SYMPTOM_PREDICTIONS,
            # Features.TEXT_INTERACTION,
            # Features.TEXT_METADATA,
            # Features.BERT, 
            # Features.PAST_DIAGNOSIS,
            # Features.PAST_MEDICATIONS,
]


# ── base outcome/mask config (shared across all trigger configs) ──────────
_primary_med_events_only_w_dosage = lambda df: df[
    TemporalEvent.TRIGGER_MED_ACTION_TYPE.value
].isin({
    "main_added",
    "main_switch",
    "main_dose_increase",
    "main_dose_decrease",
    "main_stopped",
})

_base_config = {
    "outcome":       EventLevelOutcomes.PRIMARY_MED_DRUG_CHANGE_OUTCOME.value,
    "combined_mask": _primary_med_events_only_w_dosage,
    "atc_chars":     5,
    "dsm_level":     "disorder",
}

# ── Core tabular feature set (always present) ──────────────────────────────
_core = [
    Features.PATIENT,              # gender, age
    Features.DIAGNOSIS,            # main/secondary dx + time since dx
    Features.LAB_RESULTS,          # recent abnormal lab / recent lab test / measure / type
    Features.ACTIVE_MEDICATIONS,   # active med ATC codes
    Features.ADMISSION,            # binary admission flag
    Features.TRIGGER_MEDICATIONS,  # trigger action type
    Features.EVENT_TYPE, 
  
]

# ── Text feature groups ──────────────────────────────────────────────────
_symptoms       = [Features.SYMPTOM_PREDICTIONS]
_bert           = [Features.BERT]
_interaction    = [Features.TEXT_INTERACTION]
_text_metadata  = [Features.TEXT_METADATA]
_text_all       = _symptoms + _bert + _interaction + _text_metadata

# ── Past-history feature groups ─────────────────────────────────────────────
_past_dx_all   = [Features.PAST_DIAGNOSIS]
_past_meds      = [Features.PAST_MEDICATIONS]
_past_all       = _past_dx_all + _past_meds

_comb           = [Features.SYMPTOM_PREDICTIONS, Features.PAST_MEDICATIONS]
_full_extra     = _text_all + _past_all  # everything on top of core
_symptoms_interaction = [Features.SYMPTOM_PREDICTIONS, Features.TEXT_INTERACTION]

def _add(*groups):
    """core + one or more feature groups, de-duplicated, order-preserving."""
    seen, out = set(), []
    for f in _core + [f for g in groups for f in g]:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


# ── Named atomic text groups, used to build combinations ────────────────────
_text_groups_named = {
    "symptoms":    _symptoms,
    "bert":        _bert,
    "interaction": _interaction,
  #  "metadata":    _text_metadata,
}


def _text_combo_configs(base_config, add_fn, groups_named, name_prefix):
    """
    Generate one trigger_config entry per non-empty combination of the
    given named feature groups (2^n - 1 combos), each as core + that combo.
    """
    names = list(groups_named.keys())
    configs = []
    for r in range(1, len(names) + 1):
        for combo in combinations(names, r):
            combo_groups = [groups_named[n] for n in combo]
            combo_name = "_".join(combo)
            configs.append({
                **base_config,
                "name": f"{name_prefix}",
                "feature_config": add_fn(*combo_groups),
            })
    return configs


# ── Feature groups to test ──────────────────────────────────────────────────
trigger_configs = [

    #── 1. Core tabular baseline ────────────────────────────────────────────
    {**_base_config, "name": "final",
     "feature_config": _add()},

    # # # # # # # ── 2–16. All 15 combinations of text feature groups on top of core ─────
    # *_text_combo_configs(_base_config, _add, _text_groups_named, name_prefix="final"),

    # # # # #── 17. Core + past diagnosis only (main + secondary) ───────────────────
    # {**_base_config, "name": "final",
    #  "feature_config": _add(_past_dx_all)},

    # # # # # ── 18. Core + past medications only ─────────────────────────────────────
    # {**_base_config, "name": "final",
    #  "feature_config": _add(_past_meds)},

    # # # # # ── 19. Core + all past-history features ──────────────────────────────────
    # {**_base_config, "name": "final",
    #  "feature_config": _add(_past_all)},

]
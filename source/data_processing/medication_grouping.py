"""
Author: Gizem Sogancioglu and Jip Schoneveld
Date: 2026-01-29
"""
import collections
from datetime import timedelta
from pathlib import Path
from source.data_processing.data_model import *
import pandas as pd
import re

DEBUG = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# this medication list is created considering that patient is diagnosed with depression as a main diagnosis and treatment.
# the list can be different if the main diagnosis is psychotic depression or bipolar depression.
MAIN_MEDICATIONS_PRE = r"N06A.*(?<!_M)"  # prefix code, don't match when end in _M
MAIN_MEDICATIONS = {"N05AN01", "H03AA01", "H03AA02", "H03AA03", "H03AA04", "H03AA05", "H03AA51",
                    "N05AH02", "N05AH03_D", "N05AH04_D", "N05AX08", "N05AX13", "N05AX12",  # * D=depression treatment
                    "N05AX16", "N05AX15", "N05AE03", "N05AL05", "N05AL01", "N05AE05",
                    "N03AG01", "N03AX09", "N03AF01", "N03AX11",
                    "N07BA02"  # => N06AX12
                    }

MODIFIERS = {"N05BA12", "N05BA08", "N05CD09", "N05BA09", "N05BA05", "N05BA01",
             "N05CD03", "N05CD01", "N05CD11", "N05BA06", "N05CD06", "N05CD08",
             "N05CD02", "N05BA04", "N05CD14", "N05CD07", "N05CF02", "N05CF01",
             "R06AD02", "N05BB01", "N05CH01", "N05AH05",
             "N02BF02", "N02BF01", "N06BA02", "N06BA12", "N06BA04", "N06BA09", "N06BA07", "N06BA13", "N05AA01",
             "N05AA02", "N05AA03", "N05AA04", "N05AA05", "N05AA06", "N05AA07",
             "N05AB01", "N05AB02", "N05AB03", "N05AB04", "N05AB05", "N05AB06", "N05AB07", "N05AB08", "N05AB09",
             "N05AB10",
             "N05AC01", "N05AC02", "N05AC03", "N05AC04", "N05AD01", "N05AD02", "N05AD04", "N05AD05", "N05AD06",
             "N05AD07", "N05AD08", "N05AD09", "N05AD10", "N05AE01", "N05AE02", "N05AE04",
             "N05AF01", "N05AF02", "N05AF03", "N05AF04", "N05AF05", "N05AG01", "N05AG02", "N05AG03",
             "N05AH01", "N05AH06", "N05AL02", "N05AL03", "N05AL04", "N05AL06", "N05AL07", "N05AX07", "N05AX10",
             "N05AX11", "N05AX14", "N05AX17",
             "N06AX11_M",  # *Mirtazepine as modifier
             "N06AX05_M",  # *Trazodon as modifier
             "N05AH03_M",  # *Olanzapine as modifier
             "N06AA09_M", "N06AA10_M",  # *TCA modifiers
             'N05AH04_M'  # *quetiapine used as modifier, 
             "N03AE01",  # clonazepam
             }


# Maps ATC prefix/code to pharmacological drug class
# Order matters — more specific rules checked first
DRUG_CLASS_MAP = [
# ── PRIMARY (main) MEDICATIONS ──────────────────────────────────────
# SSRIs (N06AB)
# fluoxetine N06AB03, citalopram N06AB04, paroxetine N06AB05,
# sertraline N06AB06, fluvoxamine N06AB08, escitalopram N06AB10
("N06AB", "SSRI"),

# TCAs — non-selective monoamine reuptake inhibitors (N06AA)
# desipramine N06AA01, imipramine N06AA02, clomipramine N06AA04,
# amitriptyline N06AA09, nortriptyline N06AA10, doxepin N06AA12,
# maprotiline N06AA21 (tetracyclic but same clinical group)
# NOTE: N06AA09_M and N06AA10_M are in MODIFIERS at low dose
("N06AA", "TCA"),

# MAOIs — irreversible non-selective (N06AF)
# isocarboxazid N06AF01, phenelzine N06AF03, tranylcypromine N06AF04
# Dutch guideline step 6 drugs
("N06AF", "MAOI_irreversible"),

# MAOIs — reversible MAO-A inhibitor (N06AG)
# moclobemide N06AG02 — present in ATC but NOT recommended
# in Dutch NHG guidelines; map anyway in case present in data
("N06AG", "MAOI_reversible"),

# SNRIs — sit under N06AX in WHO (no dedicated subgroup)
("N06AX16", "SNRI"),   # venlafaxine
("N06AX21", "SNRI"),   # duloxetine
("N06AX17", "SNRI"),   # milnacipran
("N06AX23", "SNRI"),   # desvenlafaxine
("N06AX28", "SNRI"),   # levomilnacipran

# NaSSAs
("N06AX11", "NaSSA"),  # mirtazapine — also N06AX11_M as modifier
("N06AX03", "NaSSA"),  # mianserin

# NDRI
("N06AX12", "NDRI"),   # bupropion as antidepressant
("N06AX62", "NDRI"),   # bupropion + dextromethorphan

# SARIs
("N06AX05", "SARI"),   # trazodone — also N06AX05_M as modifier
("N06AX06", "SARI"),   # nefazodone

# Melatonergic
("N06AX22", "melatonergic"),  # agomelatine

# Multimodal
("N06AX26", "multimodal"),    # vortioxetine

# Esketamine (treatment resistance marker — step 4b in Dutch guideline)
("N06AX27", "esketamine"),    # esketamine nasal spray (Spravato)

# Lithium (N05AN01) — augmentation steps 3 and 5
("N05AN01", "lithium"),

# Atypical antipsychotics at depression dose (_D suffix = above threshold)
("N05AH02",   "atypical_AP"),  # clozapine (rare, no dose threshold in your code)
("N05AH03_D", "atypical_AP"),  # olanzapine ≥5mg
("N05AH04_D", "atypical_AP"),  # quetiapine >100mg
("N05AX08",   "atypical_AP"),  # risperidone
("N05AX13",   "atypical_AP"),  # aripiprazole
("N05AX12",   "atypical_AP"),  # asenapine
("N05AX16",   "atypical_AP"),  # brexpiprazole
("N05AX15",   "atypical_AP"),  # cariprazine
("N05AE03",   "atypical_AP"),  # ziprasidone
("N05AE05",   "atypical_AP"),  # lurasidone
("N05AL05",   "atypical_AP"),  # amisulpride
("N05AL01",   "atypical_AP"),  # sulpiride

# Thyroid hormones (T3 augmentation)
# H03AA01 levothyroxine, H03AA02 liothyronine (T3), H03AA03-05 variants
# H03AA51 combinations
("H03AA", "thyroid"),

# Anti-epileptics used in depression augmentation
("N03AG01", "antiepileptic"),  # valproic acid / valproate
("N03AX09", "antiepileptic"),  # lamotrigine
("N03AF01", "antiepileptic"),  # carbamazepine
("N03AX11", "antiepileptic"),  # topiramate

# Bupropion for smoking cessation — same molecule as N06AX12
# included in MAIN_MEDICATIONS with comment "=> N06AX12"
("N07BA02", "NDRI"),   # bupropion (smoking cessation ATC code)

# ── MODIFIER MEDICATIONS ─────────────────────────────────────────────
# These are features only, not outcomes.
# Grouped by pharmacological subclass.

# Benzodiazepine anxiolytics (N05BA)
# diazepam N05BA01, chlordiazepoxide N05BA02, oxazepam N05BA04,
# clorazepate N05BA05, lorazepam N05BA06, bromazepam N05BA08,
# clobazam N05BA09, alprazolam N05BA12
("N05BA", "benzodiazepine_anxiolytic"),

("N03AE01", "benzodiazepine_anxiolytic"),  # clonazepam — under N03 not N05BA
# Benzodiazepine hypnotics (N05CD)
# flunitrazepam N05CD03, lormetazepam N05CD06, temazepam N05CD07,
# midazolam N05CD08, brotizolam N05CD09, loprazolam N05CD11,
# remimazolam N05CD14
("N05CD", "benzodiazepine_hypnotic"),

# Z-drugs / non-benzodiazepine hypnotics (N05CF)
# zopiclone N05CF01, zolpidem N05CF02
("N05CF", "z_drug_hypnotic"),

# Antihistamine anxiolytic (N05BB)
# hydroxyzine N05BB01
("N05BB01", "antihistamine_anxiolytic"),

# Melatonin (N05CH)
# melatonin N05CH01 — sleep modifier
("N05CH01", "melatonin"),

# Phenothiazines — low-dose sedating typical antipsychotics (modifiers)
# N05AA: aliphatic phenothiazines
# chlorpromazine N05AA01, levomepromazine N05AA02, promazine N05AA03,
# cyamemazine N05AA06
("N05AA", "typical_AP_phenothiazine"),

# N05AB: phenothiazines with piperazine structure
# fluphenazine N05AB02, perphenazine N05AB03, trifluoperazine N05AB06,
# thioproperazine N05AB08, perazine N05AB10
("N05AB", "typical_AP_phenothiazine"),

# N05AC: phenothiazines with piperidine structure
# periciazine N05AC01, thioridazine N05AC02, pipotiazine N05AC04
("N05AC", "typical_AP_phenothiazine"),

# N05AD: butyrophenone derivatives
# haloperidol N05AD01, pipamperone N05AD05, bromperidol N05AD06,
# droperidol N05AD08, benperidol N05AD07, melperone N05AD03
("N05AD", "typical_AP_butyrophenone"),

# N05AE: indole derivatives
# ziprasidone N05AE03 is in MAIN_MEDICATIONS
# sertindole N05AE02, zotepine N05AE04 are in MODIFIERS
("N05AE01", "typical_AP_indole"),   # oxypertine
("N05AE02", "atypical_AP"),         # sertindole — atypical despite being in MODIFIERS set
("N05AE04", "atypical_AP"),         # zotepine — atypical

# N05AF: thioxanthene derivatives
# flupenthixol N05AF01, clopenthixol N05AF02, chlorprothixene N05AF03,
# tiotixene N05AF04, zuclopenthixol N05AF05
("N05AF", "typical_AP_thioxanthene"),

# N05AG: diphenylbutylpiperidine derivatives
# fluspirilene N05AG01, pimozide N05AG02, penfluridol N05AG03
("N05AG", "typical_AP_diphenylbutyl"),

# N05AH: diazepines — low-dose entries in MODIFIERS
# clozapine N05AH02 is MAIN, olanzapine N05AH03_M is modifier
# quetiapine N05AH04_M is modifier, quetiapine N05AH04_D is MAIN
# asenapine N05AH05 is in MODIFIERS
("N05AH01", "atypical_AP"),         # loxapine
("N05AH05", "atypical_AP"),         # asenapine low dose (in your MODIFIERS)
("N05AH06", "atypical_AP"),         # clotiapine
("N05AH03_M", "modifier_atypical_AP_low_dose"),
("N05AH04_M", "modifier_atypical_AP_low_dose"),

# N05AL: benzamides — low-dose entries in MODIFIERS
# sulpiride N05AL01 is MAIN, amisulpride N05AL05 is MAIN
# remoxipride N05AL02, levosulpiride N05AL03 are in MODIFIERS
("N05AL02", "typical_AP_benzamide"),
("N05AL03", "typical_AP_benzamide"),
("N05AL04", "typical_AP_benzamide"),  # veralipride
("N05AL06", "typical_AP_benzamide"),  # sultopride
("N05AL07", "typical_AP_benzamide"),  # tiapride

# N05AX: other antipsychotics — low-dose entries in MODIFIERS
# risperidone N05AX08 is MAIN
# aripiprazole N05AX13 is MAIN
("N05AX07", "atypical_AP"),   # prothipendyl
("N05AX10", "atypical_AP"),   # mosapride (unusual here)
("N05AX11", "atypical_AP"),   # iloperidone
("N05AX14", "atypical_AP"),   # paliperidone
("N05AX17", "atypical_AP"),   # lumateperone

# Gabapentinoids (N02BF) — used for anxiety/pain in depression context
# pregabalin N02BF03, gabapentin N02BF01 (note: N02BF02 is not standard)
# Your MODIFIERS has N02BF02 and N02BF01
("N02BF01", "gabapentinoid"),  # gabapentin
("N02BF02", "gabapentinoid"),  # pregabalin (note: pregabalin is N03AX16 in newer WHO
                                # but N02BF03 or country-specific codes may vary)

# Stimulants (N06BA) — ADHD/fatigue modifiers
# dexamphetamine N06BA02, methylphenidate N06BA04,
# modafinil N06BA07, atomoxetine N06BA09,
# lisdexamfetamine N06BA12, armodafinil N06BA13
("N06BA", "stimulant"),

# Phenothiazine antihistamine — promethazine used as sedative modifier
# promethazine R06AD02
("R06AD02", "antihistamine_sedative"),  # promethazine

# Dose-dependent modifier versions of antidepressants
("N06AX11_M", "modifier_NaSSA_low_dose"),    # mirtazapine low dose
("N06AX05_M", "modifier_SARI_low_dose"),     # trazodone low dose
("N06AA09_M", "modifier_TCA_low_dose"),      # amitriptyline low dose
("N06AA10_M", "modifier_TCA_low_dose"),      # nortriptyline low dose
]

def get_drug_class(atc_code: str) -> str:
    """
    Returns pharmacological class for a given ATC code.
    Checks full code first, then progressively shorter prefixes.
    """
    if not atc_code or pd.isna(atc_code):
        return "unknown"
    atc = str(atc_code).strip()
    # check exact match first (handles _D/_M suffixed codes)
    for prefix, drug_class in DRUG_CLASS_MAP:
        if atc == prefix:
            return drug_class
    # then prefix match
    for prefix, drug_class in DRUG_CLASS_MAP:
        if atc.startswith(prefix):
            return drug_class
    return "other"

def set_type_w_ATC_code(medications):
    """
          Append D or M to ATC_codes of medications depending on the dosage amount.
          Parameters:
          medications (df): dataframe.
          Returns:
          medications: same data with ATC_codes modified for dosage dependent medications.
       """
    for i, m in medications.iterrows():
        ATC = m[Medication.ATC_CODE.value]
        if DEBUG: print(ATC)
        # when quetiapine
        if ATC == "N05AH04":
            if DEBUG: print("found quetiapine:", m)
            if m[Medication.DAY_DOSES.value] > 100.0:
                if DEBUG: print("main depression medication -> _D")
                medications.loc[i, Medication.ATC_CODE.value] = ATC + '_D'
            else:
                medications.loc[i, Medication.ATC_CODE.value] = ATC + '_M'

        # when TCA
        elif ATC in ["N06AA09", "N06AA10"]:
            if DEBUG: print("found TCA:", m)
            if m[Medication.DAY_DOSES.value] >= 50.0:
                if DEBUG: print("main depression medication -> _D")
                medications.loc[i, Medication.ATC_CODE.value] = ATC + '_D'
            else:
                medications.loc[i, Medication.ATC_CODE.value] = ATC + '_M'

        # when Olanzapine
        elif ATC == "N05AH03":
            if DEBUG: print("found Olanzapine:", m)
            if m[Medication.DAY_DOSES.value] >= 5:
                if DEBUG: print("main depression medication -> _D")
                medications.loc[i, Medication.ATC_CODE.value] = ATC + '_D'
            else:
                medications.loc[i, Medication.ATC_CODE.value] = ATC + '_M'

        # when Trazodon
        elif ATC == "N06AX05":
            if DEBUG: print("found Trazodon:", m)
            if m[Medication.DAY_DOSES.value] > 100:
                if DEBUG: print("main depression medication -> _D")
                medications.loc[i, Medication.ATC_CODE.value] = ATC + '_D'
            else:
                medications.loc[i, Medication.ATC_CODE.value] = ATC + '_M'

        # when Mirtazepine
        elif ATC == "N06AX11":
            if DEBUG: print("found Mirtazepine:", m)
            if m[Medication.DAY_DOSES.value] >= 15:
                if DEBUG: print("main depression medication -> _D")
                medications.loc[i, Medication.ATC_CODE.value] = ATC + '_D'
            else:
                medications.loc[i, Medication.ATC_CODE.value] = ATC + '_M'

    return medications


def set_IS_ANTIDEPRESSANT(medications):
    """
              IS_ANTIDEPRESSANT column is filled by this function: "main", "modifier" or "other".
              Parameters:
              medications (df): dataframe.
              Returns:
              medications: same data with IS_ANTIDEPRESSANT filled.
           """
    # TODO: if medication dosage value is NaN, we assume that it was used as modifier medication.
    ## dosage dependent medications

    # first keep the original atc in the base_atc_code 
    medications[Medication.BASE_ATC_CODE.value] = medications[
        Medication.ATC_CODE.value
    ].astype(str)

    medications = set_type_w_ATC_code(medications)
    medications.loc[:, Medication.ANTIDEPRESSANT_TYPE.value] = medications[Medication.ATC_CODE.value].apply(
        determine_type)

    # drug_class derived from final ATC_CODE (with _D/_M in place)
    # so N05AH04_D -> atypical_AP, N05AH04_M -> modifier_atypical_AP_low_dose
    medications[Medication.DRUG_CLASS.value] = (
        medications[Medication.ATC_CODE.value].apply(get_drug_class)
    )
    print("Drug class annotation complete.")
    print(f"  drug_class distribution (top 3):\n"
      f"{medications[Medication.DRUG_CLASS.value].value_counts().head(3).to_string()}")
    unmapped = medications[
    medications[Medication.DRUG_CLASS.value] == "other"
    ][Medication.ATC_CODE.value].value_counts()
    if not unmapped.empty:
        print(f"  Unmapped ATC codes falling through to 'other' (top 3):\n"
            f"{unmapped.head(3).to_string()}")
   
    return pd.DataFrame(medications)


def determine_type(atc_code):
    if atc_code in MAIN_MEDICATIONS or re.fullmatch(MAIN_MEDICATIONS_PRE, atc_code):
        return "main"
    elif atc_code in MODIFIERS:
        return "modifier"
    else:
        return "other"  # or "other" if you want to keep unmatched rows


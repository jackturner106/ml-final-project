import gc
import os
import pandas as pd


def load_data(
    sample_size=None,
    icu_vitals=True,
    top_n_labs=20,
    top_n_drugs=20,
    top_n_procedures=20,
    observation_hours=48,
    prediction_hours=24,
    step_hours=24,
    keep_features=None,
    missingness_indicators=False,
):
    """Load data for decompensation prediction using rolling observation windows.

    For each admission, generates multiple samples at every step_hours interval.
    At each time point t, uses clinical data from [t - observation_hours, t] to
    predict death within the next prediction_hours.

    keep_features -- optional list of feature names (e.g. from Lasso). When provided,
        only labs/drugs/procedures whose feature names appear in this list will be loaded,
        reducing memory usage.
    missingness_indicators -- if True, add binary _missing columns for lab/vital features
        before median imputation.

    Returns (X, y, groups) where groups is subject_id per window for patient-level
    train/test splitting.
    """

    # ------------------------------------- Cache Check -------------------------------------
    os.makedirs("derived/cache", exist_ok=True)
    keep_hash = "" if keep_features is None else f"_kf{len(keep_features)}"
    miss_hash = "_miss" if missingness_indicators else ""
    cache_key = (
        f"decomp_rolling_n{sample_size}_v{int(icu_vitals)}"
        f"_l{top_n_labs}_d{top_n_drugs}_p{top_n_procedures}"
        f"_o{observation_hours}_pred{prediction_hours}_s{step_hours}{keep_hash}{miss_hash}"
    )
    cache_x = f"derived/cache/{cache_key}_X.parquet"
    cache_y = f"derived/cache/{cache_key}_y.parquet"
    cache_g = f"derived/cache/{cache_key}_groups.parquet"

    if os.path.exists(cache_x) and os.path.exists(cache_y) and os.path.exists(cache_g):
        print(f"Loading from cache: {cache_key}")
        X_imputed = pd.read_parquet(cache_x)
        y = pd.read_parquet(cache_y).squeeze()
        groups = pd.read_parquet(cache_g).squeeze()
        print(
            f"Returning {X_imputed.shape[0]} windows with {X_imputed.shape[1]} features. "
            f"y distribution:\n{y.value_counts(True)}"
        )
        return X_imputed, y, groups

    # ------------------------------------- Patient and Admission -------------------------------------
    patients = pd.read_csv(
        "mimic-iv-3.1/hosp/patients.csv", usecols=["subject_id", "gender", "anchor_age"]
    )
    patients["gender_female"] = (patients["gender"] == "F").astype(int)
    patients.drop(columns="gender", inplace=True)

    admissions = pd.read_csv(
        "mimic-iv-3.1/hosp/admissions.csv",
        usecols=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "admission_type",
            "admission_location",
            "insurance",
            "race",
            "hospital_expire_flag",
        ],
    )

    def race_map(race):
        if race in [
            "WHITE",
            "WHITE - RUSSIAN",
            "WHITE - OTHER EUROPEAN",
            "WHITE - BRAZILIAN",
            "WHITE - EASTERN EUROPEAN",
            "PORTUGUESE",
        ]:
            return "WHITE"
        if (
            race.startswith("HISPANIC/LATINO")
            or race == "HISPANIC OR LATINO"
            or race == "SOUTH AMERICAN"
        ):
            return "HISPANIC/LATINO"
        if race.startswith("BLACK"):
            return "BLACK"
        if race.startswith("ASIAN"):
            return "ASIAN"
        if "NATIVE" in race:
            return "NATIVE AMERICAN/ALASKA NATIVE"
        else:
            return "UNKNOWN OR MULTIPLE"

    admissions["race"] = admissions["race"].map(race_map)

    admissions["admittime"] = pd.to_datetime(admissions["admittime"])
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])
    admissions["deathtime"] = pd.to_datetime(admissions["deathtime"])

    # Only keep admissions long enough for at least one observation window
    admissions = admissions[
        (admissions["dischtime"] - admissions["admittime"]).dt.total_seconds() / 3600
        > observation_hours
    ].copy()

    if sample_size is not None:
        # 50/50 split of death and survived admissions
        half = sample_size // 2
        died = admissions[admissions["deathtime"].notna()]
        survived = admissions[admissions["deathtime"].isna()]
        n_died = min(half, len(died))
        n_survived = min(half, len(survived))
        died = died.sample(n_died, random_state=42)
        survived = survived.sample(n_survived, random_state=42)
        admissions = pd.concat([died, survived])
        print(
            f"Sampled {len(admissions)} admissions ({len(died)} deaths, {len(survived)} survived)"
        )
    sampled_hadm_ids = set(admissions["hadm_id"])
    sampled_subject_ids = set(admissions["subject_id"])

    # ------------------------------------- Generate Rolling Windows -------------------------------------
    # First window ends at admittime + observation_hours, then every step_hours
    admissions["first_window_end"] = admissions["admittime"] + pd.Timedelta(
        hours=observation_hours
    )

    # Number of windows per admission (up to but not including dischtime)
    admissions["n_windows"] = (
        (
            (
                admissions["dischtime"] - admissions["first_window_end"]
            ).dt.total_seconds()
            / (step_hours * 3600)
        )
        .clip(lower=0)
        .astype(int)
    ) + 1

    # Explode into individual windows
    windows = admissions.loc[admissions.index.repeat(admissions["n_windows"])].copy()
    windows["window_idx"] = windows.groupby("hadm_id").cumcount()
    windows["window_end"] = windows["first_window_end"] + windows[
        "window_idx"
    ] * pd.Timedelta(hours=step_hours)
    windows["window_start"] = windows["admittime"]
    windows["hours_since_admission"] = (
        windows["window_end"] - windows["admittime"]
    ).dt.total_seconds() / 3600

    # Only keep windows where the patient is still alive and in the hospital
    windows = windows[windows["window_end"] < windows["dischtime"]].copy()

    # Target: death within prediction_hours after window_end
    windows["y"] = (
        windows["deathtime"].notna()
        & (windows["deathtime"] >= windows["window_end"])
        & (
            windows["deathtime"]
            <= windows["window_end"] + pd.Timedelta(hours=prediction_hours)
        )
    ).astype(int)

    windows["window_id"] = range(len(windows))

    avg_los = (admissions["dischtime"] - admissions["admittime"]).dt.total_seconds().mean() / 3600
    avg_windows = len(windows) / len(admissions)
    print(f"Avg length of stay: {avg_los:.1f} hours ({avg_los/24:.1f} days)")
    print(f"Generated {len(windows)} windows from {admissions.shape[0]} admissions ({avg_windows:.1f} windows/admission)")
    print(f"Positive windows: {windows['y'].sum()} ({windows['y'].mean():.4%})")

    # Build DataFrame with per-admission features
    df = windows.merge(patients, on="subject_id", how="left")

    # Keep window keys for time-windowed feature merges
    window_keys = df[["hadm_id", "window_id", "window_start", "window_end"]].copy()

    # ------------------------------------- Weight, Height, BP (baseline, no time windowing) ------
    omr = pd.read_csv(
        "mimic-iv-3.1/hosp/omr.csv",
        usecols=["subject_id", "result_name", "result_value"],
    )
    omr = omr[omr["subject_id"].isin(sampled_subject_ids)].copy()

    omr_bp = omr[omr["result_name"] == "Blood Pressure"].copy()
    omr_bp[["bp_systolic", "bp_diastolic"]] = (
        omr_bp["result_value"].str.split("/", expand=True).astype(float)
    )
    bp_by_subject = omr_bp.groupby("subject_id")[["bp_systolic", "bp_diastolic"]].mean()

    omr_weight = omr[omr["result_name"] == "Weight (Lbs)"].copy()
    omr_weight["result_value"] = omr_weight["result_value"].astype(float)
    weight_by_subject = omr_weight.groupby("subject_id")[["result_value"]].mean()

    omr_height = omr[omr["result_name"] == "Height (Inches)"].copy()
    omr_height["result_value"] = pd.to_numeric(
        omr_height["result_value"], errors="coerce"
    )
    height_by_subject = omr_height.groupby("subject_id")[["result_value"]].mean()

    df = df.merge(bp_by_subject, on="subject_id", how="left")
    df = df.merge(weight_by_subject, on="subject_id", how="left")
    df = df.merge(height_by_subject, on="subject_id", how="left")
    del omr, omr_bp, omr_weight, omr_height
    gc.collect()

    # ------------------------------------- ICU Vital signs (per window) ----------------------------
    if icu_vitals:
        VITAL_ITEMS = {
            220045: "heart_rate",
            220210: "respiratory_rate",
            220277: "spo2",
            223761: "temperature_f",
        }

        chunks = []
        for chunk in pd.read_csv(
            "mimic-iv-3.1/icu/chartevents.csv",
            usecols=["hadm_id", "itemid", "valuenum", "charttime"],
            chunksize=100000,
            low_memory=True,
        ):
            chunk = chunk[
                chunk["itemid"].isin(VITAL_ITEMS)
                & chunk["valuenum"].notna()
                & chunk["hadm_id"].isin(sampled_hadm_ids)
            ]
            if not chunk.empty:
                chunks.append(chunk)

        icu_vitals_raw = pd.concat(chunks, ignore_index=True)
        icu_vitals_raw["hadm_id"] = icu_vitals_raw["hadm_id"].astype(int)
        icu_vitals_raw["charttime"] = pd.to_datetime(icu_vitals_raw["charttime"])
        icu_vitals_raw["vital_name"] = icu_vitals_raw["itemid"].map(VITAL_ITEMS)

        # Merge with windows and filter to each window's observation period
        vitals_windowed = icu_vitals_raw.merge(window_keys, on="hadm_id")
        vitals_windowed = vitals_windowed[
            (vitals_windowed["charttime"] >= vitals_windowed["window_start"])
            & (vitals_windowed["charttime"] <= vitals_windowed["window_end"])
        ]

        icu_vitals_df = (
            vitals_windowed.groupby(["window_id", "vital_name"])["valuenum"]
            .mean()
            .unstack("vital_name")
            .add_prefix("vital_")
        )
        df = df.merge(icu_vitals_df, on="window_id", how="left")
        del icu_vitals_raw, vitals_windowed, icu_vitals_df
        gc.collect()

    # ------------------------------------- Lab Results (per window) --------------------------------
    if top_n_labs > 0:
        lab_items = pd.read_csv("derived/lab_counts.csv")
        lab_items = lab_items.nlargest(n=top_n_labs, columns="count")

        # Filter to only Lasso-selected labs
        if keep_features is not None:
            kept_lab_names = {f.removeprefix("lab_") for f in keep_features if f.startswith("lab_")}
            lab_items = lab_items[lab_items["label"].str.replace(" ", "_").str.lower().isin(kept_lab_names)]
            print(f"  Labs filtered to {len(lab_items)} by keep_features")

        lab_events = pd.read_csv(
            "mimic-iv-3.1/hosp/labevents.csv",
            usecols=["hadm_id", "itemid", "valuenum", "charttime"],
            low_memory=True,
        )
        lab_events = lab_events.dropna(subset=["hadm_id", "valuenum"])
        lab_events["hadm_id"] = lab_events["hadm_id"].astype(int)
        lab_events["charttime"] = pd.to_datetime(lab_events["charttime"])
        lab_events = lab_events[
            lab_events["itemid"].isin(lab_items["itemid"])
            & lab_events["hadm_id"].isin(sampled_hadm_ids)
        ]

        id_to_label = lab_items.set_index("itemid")["label"].to_dict()
        lab_events["lab_name"] = (
            lab_events["itemid"].map(id_to_label).str.replace(" ", "_").str.lower()
        )

        # Merge with windows and filter to each window's observation period
        labs_windowed = lab_events.merge(window_keys, on="hadm_id")
        labs_windowed = labs_windowed[
            (labs_windowed["charttime"] >= labs_windowed["window_start"])
            & (labs_windowed["charttime"] <= labs_windowed["window_end"])
        ]

        lab_features = (
            labs_windowed.groupby(["window_id", "lab_name"])["valuenum"]
            .mean()
            .unstack(level="lab_name")
            .add_prefix("lab_")
        )
        df = df.merge(lab_features, on="window_id", how="left")
        del lab_events, labs_windowed, lab_features
        gc.collect()

    # ------------------------------------- Prescribed Drugs (per window) ---------------------------
    if top_n_drugs > 0:
        drug_counts = pd.read_csv("derived/prescription_counts.csv")
        drug_counts = drug_counts.nlargest(n=top_n_drugs, columns="count")

        # Filter to only Lasso-selected drugs
        if keep_features is not None:
            kept_drug_names = {f.removeprefix("drug_") for f in keep_features if f.startswith("drug_")}
            drug_counts = drug_counts[drug_counts["drug"].str.replace(" ", "_").str.lower().isin(kept_drug_names)]
            print(f"  Drugs filtered to {len(drug_counts)} by keep_features")

        prescriptions = pd.read_csv(
            "mimic-iv-3.1/hosp/prescriptions.csv",
            usecols=["hadm_id", "ndc", "dose_val_rx", "starttime", "stoptime"],
            low_memory=True,
        )
        prescriptions = prescriptions.dropna(subset=["hadm_id", "dose_val_rx"])
        prescriptions["hadm_id"] = prescriptions["hadm_id"].astype(int)
        prescriptions["starttime"] = pd.to_datetime(prescriptions["starttime"])
        prescriptions["stoptime"] = pd.to_datetime(prescriptions["stoptime"])
        prescriptions["dose_val_rx"] = pd.to_numeric(
            prescriptions["dose_val_rx"], errors="coerce"
        )
        prescriptions = prescriptions[
            prescriptions["ndc"].isin(drug_counts["ndc"])
            & prescriptions["hadm_id"].isin(sampled_hadm_ids)
        ]

        id_to_label = drug_counts.set_index("ndc")["drug"].to_dict()
        prescriptions["ndc"] = (
            prescriptions["ndc"].map(id_to_label).str.replace(" ", "_").str.lower()
        )

        # Drug active if [starttime, stoptime] overlaps [window_start, window_end]
        drugs_windowed = prescriptions.merge(window_keys, on="hadm_id")
        drugs_windowed = drugs_windowed[
            (drugs_windowed["starttime"] < drugs_windowed["window_end"])
            & (drugs_windowed["window_start"] < drugs_windowed["stoptime"])
        ]

        drug_features = (
            drugs_windowed.groupby(["window_id", "ndc"])["dose_val_rx"]
            .mean()
            .unstack(level="ndc")
            .add_prefix("drug_")
        )
        drug_features = drug_features.fillna(0)
        df = df.merge(drug_features, on="window_id", how="left")
        del prescriptions, drugs_windowed, drug_features
        gc.collect()

    # ------------------------------------- Procedures (per admission, no time windowing) -----------
    if top_n_procedures > 0:
        procedure_counts = pd.read_csv("derived/procedure_counts.csv")
        procedure_counts = procedure_counts.nlargest(
            n=top_n_procedures, columns="count"
        )

        # Filter to only Lasso-selected procedures
        if keep_features is not None:
            kept_proc_codes = {f.removeprefix("procedure_") for f in keep_features if f.startswith("procedure_")}
            procedure_counts = procedure_counts[procedure_counts["icd_code"].isin(kept_proc_codes)]
            print(f"  Procedures filtered to {len(procedure_counts)} by keep_features")

        procedures = pd.read_csv(
            "mimic-iv-3.1/hosp/procedures_icd.csv",
            usecols=["hadm_id", "icd_code"],
            low_memory=True,
        )
        procedures = procedures.dropna(subset=["hadm_id", "icd_code"])
        procedures["hadm_id"] = procedures["hadm_id"].astype(int)

        procedures = procedures[
            procedures["icd_code"].isin(procedure_counts["icd_code"])
            & procedures["hadm_id"].isin(sampled_hadm_ids)
        ]

        procedure_features = (
            procedures.groupby(["hadm_id", "icd_code"])
            .size()
            .unstack(level="icd_code")
            .add_prefix("procedure_")
        )
        procedure_features = procedure_features.fillna(0)
        df = df.merge(procedure_features, on="hadm_id", how="left")

    # ------------------------------------- Data Engineering -------------------------------------
    cat_cols = ["admission_type", "admission_location", "insurance", "race"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    # Save groups before dropping identifiers
    groups = df["subject_id"].reset_index(drop=True)
    y = df["y"].reset_index(drop=True)

    drop_cols = [
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "deathtime",
        "hospital_expire_flag",
        "first_window_end",
        "n_windows",
        "window_idx",
        "window_start",
        "window_end",
        "window_id",
        "y",
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]

    if missingness_indicators:
        # Add binary _missing columns for lab and vital features before imputing
        miss_cols = [c for c in X.columns if c.startswith("lab_") or c.startswith("vital_")]
        miss_df = X[miss_cols].isna().astype(int).rename(columns=lambda c: c + "_missing")
        X = pd.concat([X, miss_df], axis=1)

    X_imputed = X.fillna(X.median(numeric_only=True)).reset_index(drop=True)

    # ------------------------------------- Cache Save -------------------------------------
    X_imputed.to_parquet(cache_x)
    y.to_frame().to_parquet(cache_y)
    groups.to_frame().to_parquet(cache_g)
    print(f"Cached to: {cache_key}")

    print(
        f"Returning {X_imputed.shape[0]} windows with {X_imputed.shape[1]} features. "
        f"y distribution:\n{y.value_counts(True)}"
    )

    return X_imputed, y, groups

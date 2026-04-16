import pandas as pd


def load_data(
    sample_size,
    icu_vitals=True,
    top_n_labs=20,
    top_n_drugs=20,
    top_n_procedures=20,
    observation_hours=48,
):
    """Load data for decompensation prediction. This means data in a time-bound window up to
    24 hours before death or discharge.

    Keyword arguments:
    sample_size -- The number of data rows to be returned (up to ~500,000)
    icu_vitals -- Whether or not to include ICU vital data. ICU file is ~40Gb, so this increases load times significantly
    top_n_labs -- Number of lab tests to include as features. The most common n labs will be selected
    top_n_drugs -- Number of drug prescriptions to include as features. The most commonly prescribed n drugs will be included
    top_n_procedures -- Number of procedures to include as features. The most commonly performed n procedures will be included
    observation_hours -- Size of the observation window

    """

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
        if race in ["WHITE", "WHITE - RUSSIAN", "WHITE - OTHER EUROPEAN", "WHITE - BRAZILIAN", "WHITE - EASTERN EUROPEAN", "PORTUGUESE"]:
            return "WHITE"
        if race.startswith("HISPANIC/LATINO") or race == "HISPANIC OR LATINO" or race == "SOUTH AMERICAN":
            return "HISPANIC/LATINO"
        if race.startswith("BLACK"):
            return "BLACK"
        if race.startswith("ASIAN"):
            return "ASIAN"
        if "NATIVE" in race:
            return "NATIVE AMERICAN/ALASKA NATIVE"
        else:
            return "UNKNOWN OR MULTIPLE"
    admissions['race'] = admissions['race'].map(race_map)

    admissions["admittime"] = pd.to_datetime(admissions["admittime"])
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])
    admissions["deathtime"] = pd.to_datetime(admissions["deathtime"])
    # Looks like discharge time is set to deathtime for patients who died in hospital
    admissions["los_hours"] = (
        admissions["dischtime"] - admissions["admittime"]
    ).dt.total_seconds() / 3600

    # Observation cutoff window ends 24 hours before death or discharge
    admissions["obs_end_cutoff"] = admissions["dischtime"] - pd.Timedelta(hours=24)
    admissions["obs_begin_cutoff"] = admissions["obs_end_cutoff"] - pd.Timedelta(
        hours=observation_hours
    )
    obs_cutoffs = admissions[["hadm_id", "obs_begin_cutoff", "obs_end_cutoff"]]
    subject_obs_cutoff = admissions[
        ["subject_id", "obs_begin_cutoff", "obs_end_cutoff"]
    ].drop_duplicates("subject_id")

    # Only keep patients who were alr in the hospital at the start of the observation window
    admissions = admissions[
        admissions["admittime"] <= admissions["obs_begin_cutoff"]
    ].copy()

    # Ain't running ts on all 500,000 patients
    admissions = admissions.sample(sample_size, random_state=42)

    sampled_hadm_ids = set(admissions["hadm_id"])
    sampled_subject_ids = set(admissions["subject_id"])

    df = admissions.merge(patients, on="subject_id", how="left")

    # ------------------------------------- Weight, Height, BP -------------------------------------
    # Weight, height, Blood pressure from hosp/omr.csv
    # omr also includes BMI, but that would be collinear with weight and height
    omr = pd.read_csv(
        "mimic-iv-3.1/hosp/omr.csv",
        usecols=["subject_id", "result_name", "result_value", "chartdate"],
    )
    omr["chartdate"] = pd.to_datetime(omr["chartdate"])
    omr["charttime"] = omr["chartdate"].dt.to_period("D").dt.end_time
    omr = omr[omr["subject_id"].isin(sampled_subject_ids)].copy()

    # Remove records that aren't in the observation window
    omr = omr.merge(subject_obs_cutoff, on="subject_id", how="left")
    omr = omr[
        (omr["obs_begin_cutoff"] <= omr["charttime"])
        & (omr["charttime"] <= omr["obs_end_cutoff"])
    ]

    omr_bp = omr[(omr["result_name"] == "Blood Pressure")].copy()
    omr_bp[["bp_systolic", "bp_diastolic"]] = (
        omr_bp["result_value"].str.split("/", expand=True).astype(float)
    )
    # this takes the mean of all readings for a particular subject, not the mean of the systolic and diastolic
    bp_by_subject = omr_bp.groupby("subject_id")[["bp_systolic", "bp_diastolic"]].mean()

    omr_weight = omr[(omr["result_name"] == "Weight (Lbs)")].copy()
    omr_weight["result_value"] = omr_weight["result_value"].astype(float)
    weight_by_subject = omr_weight.groupby("subject_id")[["result_value"]].mean()

    omr_height = omr[(omr["result_name"] == "Height (Inches)")].copy()
    omr_height["result_value"] = pd.to_numeric(
        omr_height["result_value"], errors="coerce"
    )
    height_by_subject = omr_height.groupby("subject_id")[["result_value"]].mean()

    df = df.merge(bp_by_subject, on="subject_id", how="left")
    df = df.merge(weight_by_subject, on="subject_id", how="left")
    df = df.merge(height_by_subject, on="subject_id", how="left")

    # ------------------------------------- ICU Vital signs -------------------------------------
    if icu_vitals:
        # Heart rate, respiratory rate, SpO2, temperature from icu/chartevents.csv
        VITAL_ITEMS = {
            220045: "heart_rate",
            220210: "respiratory_rate",
            220277: "spo2",
            223761: "temperature_f",
        }

        # File is 40Gb, read in chunks and keep only records for the patients we sampled
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

        # Remove records that aren't in the observation window
        icu_vitals_raw = icu_vitals_raw.merge(obs_cutoffs, on="hadm_id", how="left")
        icu_vitals_raw = icu_vitals_raw[
            (icu_vitals_raw["obs_begin_cutoff"] < icu_vitals_raw["charttime"])
            & (icu_vitals_raw["charttime"] <= icu_vitals_raw["obs_end_cutoff"])
        ]

        icu_vitals_df = (
            icu_vitals_raw.groupby(["hadm_id", "vital_name"])["valuenum"]
            .mean()
            .unstack("vital_name")
            .add_prefix("vital_")
        )
        df = df.merge(icu_vitals_df, on="hadm_id", how="left")

    # ------------------------------------- Lab Results -------------------------------------
    if top_n_labs > 0:
        # Get the n most frequently ordered labs
        lab_items = pd.read_csv("derived/lab_counts.csv")
        lab_items = lab_items.nlargest(n=top_n_labs, columns="count")

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

        # Remove records that aren't in the observation window
        lab_events = lab_events.merge(obs_cutoffs, on="hadm_id", how="left")
        lab_events = lab_events[
            (lab_events["obs_begin_cutoff"] <= lab_events["charttime"])
            & (lab_events["charttime"] <= lab_events["obs_end_cutoff"])
        ]

        id_to_label = lab_items.set_index("itemid")["label"].to_dict()
        lab_events["lab_name"] = (
            lab_events["itemid"].map(id_to_label).str.replace(" ", "_").str.lower()
        )

        lab_features = (
            lab_events.groupby(["hadm_id", "lab_name"])["valuenum"]
            .mean()
            .unstack(level="lab_name")
            .add_prefix("lab_")
        )
        df = df.merge(lab_features, on="hadm_id", how="left")

    # ------------------------------------- Prescribed Drugs -------------------------------------
    if top_n_drugs > 0:
        drug_counts = pd.read_csv("derived/prescription_counts.csv")
        drug_counts = drug_counts.nlargest(n=top_n_drugs, columns="count")

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

        prescriptions = prescriptions.merge(obs_cutoffs, on="hadm_id", how="left")
        prescriptions = prescriptions[
            (prescriptions["starttime"] < prescriptions["obs_end_cutoff"])
            & (prescriptions["obs_begin_cutoff"] < prescriptions["stoptime"])
        ]

        id_to_label = drug_counts.set_index("ndc")["drug"].to_dict()
        prescriptions["ndc"] = (
            prescriptions["ndc"].map(id_to_label).str.replace(" ", "_").str.lower()
        )

        drug_features = (
            prescriptions.groupby(["hadm_id", "ndc"])["dose_val_rx"]
            .mean()
            .unstack(level="ndc")
            .add_prefix("drug_")
        )

        drug_features = drug_features.fillna(0)

        df = df.merge(drug_features, on="hadm_id", how="left")

    # ------------------------------------- Procedures Performed -------------------------------------
    if top_n_procedures > 0:
        procedure_counts = pd.read_csv("derived/procedure_counts.csv")
        procedure_counts = procedure_counts.nlargest(
            n=top_n_procedures, columns="count"
        )

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
            .size()  # number of times the procedure was performed
            .unstack(level="icd_code")
            .add_prefix("procedure_")
        )
        procedure_features = procedure_features.fillna(0)
        procedure_features.head(100)
        df = df.merge(procedure_features, on="hadm_id", how="left")

    # ------------------------------------- Data Engineering -------------------------------------

    # One-hot encode categorical variables
    # This kind of sucks, end up with 32 different columns just to represent race...
    cat_cols = ["admission_type", "admission_location", "insurance", "race"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    # These columns aren't used as features
    drop_cols = [
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "deathtime",
        "obs_cutoff",
        "hospital_expire_flag",
        "obs_begin_cutoff",
        "obs_end_cutoff",
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    # Impute for now so we can lasso, obviously won't impute once we're featurizing the presence/absence of a lab
    X_imputed = X.fillna(X.median(numeric_only=True))

    # Is the patient dead within 24 hours?
    y = (
        df["deathtime"].notna()
        & (df["deathtime"] <= df["obs_end_cutoff"] + pd.Timedelta(hours=24))
    ).astype(int)

    print(
        f"Returing {X_imputed.shape[0]} patient records with {X_imputed.shape[1]} columns. y distribution: \n{y.value_counts(True)}"
    )

    return X_imputed, y

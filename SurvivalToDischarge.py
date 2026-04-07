import pandas as pd

def load_data(sample_size, icu_vitals=True, top_n_labs=20, top_n_drugs=20, top_n_procedures=20):
    """ Load data for predicting survival to discharge. This includes bio data, vitals and labs
    with no time bound.

    Keyword arguments:
    sample_size -- The number of data rows to be returned (up to ~500,000)
    icu_vitals -- Whether or not to include ICU vital data. ICU file is ~40Gb, so this increases load times significantly
    top_n_labs -- Number of lab tests to include as features. The most common n labs will be selected
    top_n_drugs -- Number of drug prescriptions to include as features. The most commonly prescribed n drugs will be included
    top_n_procedures -- Number of procedures to include as features. The most commonly performed n procedures will be included

    """

    # ------------------------------------- Patient and Admission -------------------------------------
    patients = pd.read_csv('mimic-iv-3.1/hosp/patients.csv',
                        usecols=['subject_id', 'gender', 'anchor_age'])
    patients['gender_female'] = (patients['gender'] == 'F').astype(int)
    patients.drop(columns='gender', inplace=True)

    admissions = pd.read_csv('mimic-iv-3.1/hosp/admissions.csv',
                            usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime',
                                    'deathtime', 'admission_type', 'admission_location',
                                    'insurance', 'race', 'hospital_expire_flag'])
    
    # Sorry but we can't train very well with 32 different columns for race
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

    admissions['admittime'] = pd.to_datetime(admissions['admittime'])
    admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
    admissions['deathtime'] = pd.to_datetime(admissions['deathtime'])
    # Looks like discharge time is set to deathtime for patients who died in hospital
    admissions['los_hours'] = (admissions['dischtime'] - admissions['admittime']).dt.total_seconds() / 3600

    # Ain't running ts on all 500,000 patients
    admissions = admissions.sample(sample_size, random_state=42)
    sampled_hadm_ids = set(admissions['hadm_id'])
    sampled_subject_ids = set(admissions['subject_id'])

    df = admissions.merge(patients, on='subject_id', how='left')

    # ------------------------------------- Weight, Heigh, BP -------------------------------------
    # Weight, height, Blood pressure from hosp/omr.csv
    omr = pd.read_csv('mimic-iv-3.1/hosp/omr.csv',
                    usecols=['subject_id', 'result_name', 'result_value'])
    # omr also includes BMI, but that would be collinear with weight and height
    omr_bp = omr[
        (omr['result_name'] == 'Blood Pressure') &
        omr['subject_id'].isin(sampled_subject_ids)
    ].copy()
    omr_bp[['bp_systolic', 'bp_diastolic']] = (
        omr_bp['result_value'].str.split('/', expand=True).astype(float)
    )
    bp_by_subject_mean = omr_bp.groupby('subject_id')[['bp_systolic', 'bp_diastolic']].mean().add_prefix('mean') # this takes the mean of all readings for a particular subject, not the mean of the systolic and diastolic
    bp_by_subject_max = omr_bp.groupby('subject_id')[['bp_systolic', 'bp_diastolic']].max().add_prefix('max')
    bp_by_subject_min = omr_bp.groupby('subject_id')[['bp_systolic', 'bp_diastolic']].min().add_prefix('min')
    omr_weight = omr[(omr['result_name'] == 'Weight (Lbs)') & omr['subject_id'].isin(sampled_subject_ids)].copy()
    omr_weight['result_value'] = omr_weight['result_value'].astype(float)
    weight_by_subject = omr_weight.groupby('subject_id')[['result_value']].mean()
    omr_height = omr[(omr['result_name'] == 'Height (Inches)') & omr['subject_id'].isin(sampled_subject_ids)].copy()
    omr_height['result_value'] = pd.to_numeric(omr_height['result_value'], errors='coerce')
    height_by_subject = omr_height.groupby('subject_id')[['result_value']].mean()

    df = df.merge(bp_by_subject_mean, on='subject_id', how='left')
    df = df.merge(bp_by_subject_max, on='subject_id', how='left')
    df = df.merge(bp_by_subject_min, on='subject_id', how='left')
    df = df.merge(weight_by_subject, on='subject_id', how='left')
    df = df.merge(height_by_subject, on='subject_id', how='left')

    # ------------------------------------- ICU Vital signs -------------------------------------
    if icu_vitals:
        # Heart rate, respiratory rate, SpO2, temperature from icu/chartevents.csv
        # chartevents includes systolic/diastolic blood pressure readings, for now just using the ones from omr
        # but we may find that omr is missing data for some patients that we can fill in with chartevents
        VITAL_ITEMS = {
            220045: 'heart_rate',
            220210: 'respiratory_rate',
            220277: 'spo2',
            223761: 'temperature_f',
        }

        # File is 40Gb, read in chunks
        chunks = []
        for chunk in pd.read_csv(
            'mimic-iv-3.1/icu/chartevents.csv',
            usecols=['hadm_id', 'itemid', 'valuenum'],
            chunksize=100000,
            low_memory=True,
        ):
            chunk = chunk[
                chunk['itemid'].isin(VITAL_ITEMS) &
                chunk['valuenum'].notna() &
                chunk['hadm_id'].isin(sampled_hadm_ids)
            ]
            if not chunk.empty:
                chunks.append(chunk)

        icu_vitals_raw = pd.concat(chunks, ignore_index=True)
        icu_vitals_raw['hadm_id'] = icu_vitals_raw['hadm_id'].astype(int)
        icu_vitals_raw['vital_name'] = icu_vitals_raw['itemid'].map(VITAL_ITEMS)

        icu_vitals_mean = (icu_vitals_raw
                    .groupby(['hadm_id', 'vital_name'])['valuenum']
                    .mean()
                    .unstack('vital_name')
                    .add_prefix('vital_mean_'))
        icu_vitals_max = (icu_vitals_raw
                    .groupby(['hadm_id', 'vital_name'])['valuenum']
                    .max()
                    .unstack('vital_name')
                    .add_prefix('vital_max_'))
        icu_vitals_min = (icu_vitals_raw
                    .groupby(['hadm_id', 'vital_name'])['valuenum']
                    .min()
                    .unstack('vital_name')
                    .add_prefix('vital_min_'))
        df = df.merge(icu_vitals_mean, on='hadm_id', how='left')
        df = df.merge(icu_vitals_max, on='hadm_id', how='left')
        df = df.merge(icu_vitals_min, on='hadm_id', how='left')
    
    # ------------------------------------- Lab Results -------------------------------------
    if top_n_labs > 0:
        lab_items = pd.read_csv('derived/lab_counts.csv')
        lab_items = lab_items.nlargest(n=top_n_labs, columns="count")
        
        lab_events = pd.read_csv('mimic-iv-3.1/hosp/labevents.csv',
                            usecols=['hadm_id', 'itemid', 'valuenum'],
                            low_memory=True)
        lab_events = lab_events.dropna(subset=['hadm_id', 'valuenum'])
        lab_events['hadm_id'] = lab_events['hadm_id'].astype(int)
        lab_events = lab_events[
            lab_events['itemid'].isin(lab_items['itemid']) &
            lab_events['hadm_id'].isin(sampled_hadm_ids)
        ]

        id_to_label = lab_items.set_index('itemid')['label'].to_dict()
        lab_events['lab_name'] = lab_events['itemid'].map(id_to_label).str.replace(' ', '_').str.lower()

        lab_features = (lab_events
                        .groupby(['hadm_id', 'lab_name'])['valuenum']
                        .mean()
                        .unstack(level='lab_name')
                        .add_prefix('lab_'))
        df = df.merge(lab_features, on='hadm_id', how='left')

    # ------------------------------------- Prescribed Drugs -------------------------------------
    if top_n_drugs > 0:
        drug_counts = pd.read_csv('derived/prescription_counts.csv')
        drug_counts = drug_counts.nlargest(n=top_n_drugs, columns="count")

        prescriptions = pd.read_csv('mimic-iv-3.1/hosp/prescriptions.csv',
                            usecols=['hadm_id', 'ndc', 'dose_val_rx'],
                            low_memory=True)
        prescriptions = prescriptions.dropna(subset=['hadm_id', 'dose_val_rx'])
        prescriptions['hadm_id'] = prescriptions['hadm_id'].astype(int)
        prescriptions['dose_val_rx'] = pd.to_numeric(prescriptions['dose_val_rx'], errors='coerce')
        prescriptions = prescriptions[
            prescriptions['ndc'].isin(drug_counts['ndc']) &
            prescriptions['hadm_id'].isin(sampled_hadm_ids)
        ]

        id_to_label = drug_counts.set_index('ndc')['drug'].to_dict()
        prescriptions['ndc'] = prescriptions['ndc'].map(id_to_label).str.replace(' ', '_').str.lower()

        drug_features = (prescriptions
                        .groupby(['hadm_id', 'ndc'])['dose_val_rx']
                        .mean()
                        .unstack(level='ndc')
                        .add_prefix('drug_'))
       
        drug_features = drug_features.fillna(0)

        df = df.merge(drug_features, on='hadm_id', how='left')

    # ------------------------------------- Procedures Performed -------------------------------------
    if top_n_procedures > 0:
        procedure_counts = pd.read_csv('derived/procedure_counts.csv')
        procedure_counts = procedure_counts.nlargest(n=top_n_procedures, columns="count")

        procedures = pd.read_csv('mimic-iv-3.1/hosp/procedures_icd.csv',
                            usecols=['hadm_id', 'icd_code'],
                            low_memory=True)
        procedures = procedures.dropna(subset=['hadm_id', 'icd_code'])
        procedures['hadm_id'] = procedures['hadm_id'].astype(int)

        procedures = procedures[
            procedures['icd_code'].isin(procedure_counts['icd_code']) &
            procedures['hadm_id'].isin(sampled_hadm_ids)
        ]

        procedure_features = (procedures
                        .groupby(['hadm_id', 'icd_code'])
                        .size() # number of times the procedure was performed
                        .unstack(level='icd_code')
                        .add_prefix('procedure_'))
        procedure_features = procedure_features.fillna(0)
        procedure_features.head(100)
        df = df.merge(procedure_features, on='hadm_id', how='left')


    # ------------------------------------- Data Engineering -------------------------------------

    # One-hot encode categorical variables
    cat_cols = ['admission_type', 'admission_location', 'insurance', 'race']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    # These columns aren't used as features
    drop_cols = ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'hospital_expire_flag']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    # Impute for now so we can lasso, obviously won't impute once we're featurizing the presence/absence of a lab
    X_imputed = X.fillna(X.median(numeric_only=True))
    
    y = df['hospital_expire_flag']
    
    print(f"Returing {X_imputed.shape[0]} patient records with {X_imputed.shape[1]} columns. y distribution: \n{y.value_counts(True)}")
    
    return X_imputed, y

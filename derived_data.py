# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd

# %%
prescriptions = pd.read_csv('mimic-iv-3.1/hosp/prescriptions.csv',
                        usecols=['subject_id', 'drug', 'ndc', 'gsn'])

drug_counts = prescriptions['ndc'].value_counts().reset_index()
drug_counts.columns = ['ndc', 'count']
merged = pd.merge(drug_counts, prescriptions[['ndc', 'drug', 'gsn']], on='ndc', how='right').drop_duplicates('ndc')
print(merged.head(20))

merged.to_csv("derived/prescription_counts.csv", index=False)

# %%
top_n_drugs = 10
drug_counts = pd.read_csv('derived/prescription_counts.csv')
drug_counts = drug_counts.nlargest(n=top_n_drugs, columns="count")

prescriptions = pd.read_csv('mimic-iv-3.1/hosp/prescriptions.csv',
                    usecols=['hadm_id', 'ndc', 'dose_val_rx'],
                    low_memory=True)
prescriptions = prescriptions.dropna(subset=['hadm_id', 'dose_val_rx'])
prescriptions['hadm_id'] = prescriptions['hadm_id'].astype(int)
prescriptions['dose_val_rx'] = pd.to_numeric(prescriptions['dose_val_rx'], errors='coerce')
prescriptions = prescriptions[
    prescriptions['ndc'].isin(drug_counts['ndc']) #&
    #prescriptions['hadm_id'].isin(sampled_hadm_ids)
]

id_to_label = drug_counts.set_index('ndc')['drug'].to_dict()
prescriptions['ndc'] = prescriptions['ndc'].map(id_to_label).str.replace(' ', '_').str.lower()

drug_features = (prescriptions
                .groupby(['hadm_id', 'ndc'])['dose_val_rx']
                .mean()
                .unstack(level='ndc')
                .add_prefix('drug_'))
drug_features = drug_features.fillna(0)
drug_features.head(100)
#df = df.merge(drug_features, on='hadm_id', how='left')

# %%
lab_items = pd.read_csv('derived/lab_counts.csv')
lab_items = lab_items.nlargest(n=100, columns="count")

lab_events = pd.read_csv('mimic-iv-3.1/hosp/labevents.csv',
                    usecols=['hadm_id', 'itemid', 'valuenum'],
                    low_memory=True)
lab_events = lab_events.dropna(subset=['hadm_id', 'valuenum'])
lab_events['hadm_id'] = lab_events['hadm_id'].astype(int)
lab_events = lab_events[
    lab_events['itemid'].isin(lab_items['itemid']) #&
    #lab_events['hadm_id'].isin(sampled_hadm_ids)
]

id_to_label = lab_items.set_index('itemid')['label'].to_dict()
lab_events['lab_name'] = lab_events['itemid'].map(id_to_label).str.replace(' ', '_').str.lower()

lab_features = (lab_events
                .groupby(['hadm_id', 'lab_name'])['valuenum']
                .mean()
                .unstack(level='lab_name')
                .add_prefix('lab_'))
lab_features.head(100)
#df = df.merge(lab_features, on='hadm_id', how='left')

# %%
procedures = pd.read_csv('mimic-iv-3.1/hosp/procedures_icd.csv',
                        usecols=['subject_id', 'seq_num', 'chartdate', 'icd_code'])

code_to_name = pd.read_csv('mimic-iv-3.1/hosp/d_icd_procedures.csv')

procedure_counts = procedures['icd_code'].value_counts().reset_index()
procedure_counts.columns = ['icd_code', 'count']
merged = pd.merge(procedure_counts, procedures[['icd_code']], on='icd_code', how='right').drop_duplicates('icd_code')
code_to_label = code_to_name.set_index('icd_code')['long_title'].to_dict()
merged["long_title"] = merged['icd_code'].map(code_to_label)
print(merged.head(20))

merged.to_csv("derived/procedure_counts.csv", index=False)

# %%
top_n_procedures = 10
procedure_counts = pd.read_csv('derived/procedure_counts.csv')
procedure_counts = procedure_counts.nlargest(n=top_n_procedures, columns="count")

procedures = pd.read_csv('mimic-iv-3.1/hosp/procedures_icd.csv',
                    usecols=['hadm_id', 'icd_code'],
                    low_memory=True)
procedures = procedures.dropna(subset=['hadm_id', 'icd_code'])
procedures['hadm_id'] = procedures['hadm_id'].astype(int)

procedures = procedures[
    procedures['icd_code'].isin(procedure_counts['icd_code']) #&
    #procedures['hadm_id'].isin(sampled_hadm_ids)
]

procedure_features = (procedures
                .groupby(['hadm_id', 'icd_code'])
                .size() # number of times the procedure was performed
                .unstack(level='icd_code'))
procedure_features = procedure_features.fillna(0)
procedure_features.head(100)
#df = df.merge(drug_features, on='hadm_id', how='left')

# %%
admissions = pd.read_csv('mimic-iv-3.1/hosp/admissions.csv',
                            usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime',
                                    'deathtime', 'admission_type', 'admission_location',
                                    'insurance', 'race', 'hospital_expire_flag'])
print(admissions['race'].unique())
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
admissions['race'].map(race_map).unique()

# %%
admissions = pd.read_csv('mimic-iv-3.1/hosp/admissions.csv',
                            usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime',
                                    'deathtime', 'admission_type', 'admission_location',
                                    'insurance', 'race', 'hospital_expire_flag'])
print(admissions['admission_location'].unique())

# %%
admissions = pd.read_csv('mimic-iv-3.1/hosp/admissions.csv',
                            usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime',
                                    'deathtime', 'admission_type', 'admission_location',
                                    'insurance', 'race', 'hospital_expire_flag'])

print("Number of stays", admissions.shape[0])
print("Unique patients", admissions['subject_id'].unique().shape[0])

admissions['admittime'] = pd.to_datetime(admissions['admittime'])
admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
admissions['los_hours'] = (admissions['dischtime'] - admissions['admittime'])
print("Median stay:", admissions['los_hours'].median())
print("Mean stay:", admissions['los_hours'].mean())

num_dead = admissions[admissions['hospital_expire_flag']==1]['hospital_expire_flag'].count()
print("Dead in hospital", num_dead, num_dead / admissions.shape[0])

# %%
lab_items = pd.read_csv('derived/lab_counts.csv')
lab_items = lab_items.sort_values('count', ascending=False)
lab_items = lab_items[lab_items['count'] > 546028/200]
print("Lab items used on more than 1 percent of patients:", lab_items.shape[0])
lab_items.head(10)

# %%
procedure_counts = pd.read_csv('derived/procedure_counts.csv')
procedure_counts = procedure_counts.sort_values('count', ascending=False)
procedure_counts = procedure_counts[procedure_counts['count'] > 546028/200]
print("Procedures performed on more than 1 percent of patients:", procedure_counts.shape[0])
procedure_counts.head(10)

# %%
prescription_counts = pd.read_csv('derived/prescription_counts.csv')
prescription_counts = prescription_counts.sort_values('count', ascending=False)
prescription_counts = prescription_counts[prescription_counts['count'] > 546028/200]
print("Prescriptions given to more than 1 percent of patients:", prescription_counts.shape[0])
prescription_counts.head(10)

# Featurization and Feature Selection for ICU Data

This repository contains code for logistic regression, LASSO regression, decision trees, and random forests on the MIMIC-IV dataset.

## Data Processing

There are 3 files for data processing

`SurvivalToDischarge.py` Builds a feature matrix for the survival to discharge task. This includes Patient information, Vitals data, Lab results, Prescribed medication, and Procedure information. Each row of the dataframe represent one admission to the hospital, and all data taken over the course of that admission is present.

`DecompensationImputed.py` Builds the feature matrix for the decompensation prediction task. Each row contains data from a single 24 hour window of a patient's stay, with the goal of predicting whether the patient decompensates at the end of 24 hours or not.

`FeaturizedMissingness.py` Builds a feature matrix where the missingness of lab information is added as a feature. That is, instead of imputing missing lab values with the mean, each lab column contains a 0 or 1 indicating whether that lab was performed or not.

## Training and Inferencing

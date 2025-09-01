# QSAR Model for Predicting Antioxidant Activity

This repository contains the full pipeline for developing five machine learning–based QSAR (Quantitative Structure–Activity Relationship) models to classify antioxidant compounds. The workflow covers data preprocessing, molecular descriptor calculation & processing, feature selection, model training and evaluation, and external validation on the ZINC Natural Products dataset using the final model.

---
##  Project Overview
- Objective: Build a robust binary classifier for antioxidant activity.
- Approach: Train on balanced datasets via resampling, evaluate with cross-validation & an independent test set, and assess generalizability using external ZINC compounds.
- Techniques Used:
  - Random-Forest–based feature importance & elbow-point feature selection
  - Multi-metric evaluation (F1, MCC, AUC, Accuracy, Specificity)
  - Robustness checks
    1. External screening on ZINC;
    2. Training/Evaluation without resampling (imbalanced negative pool).
- Dataset:
  - In-house libraries (SelleckChem): antioxidant (positive) vs. non-antioxidant (negative) compounds
  - External: ZINC Natural Products
- Final Model: **"XGboost classifier with the Top 51 descriptors"**
---

## Data Availability Policy  
Due to the file size limit of GitHub and license restrictions of third-party databases (e.g., SelleckChem, ZINC),  
⚠️ Note: Raw datasets from SelleckChem cannot be redistributed due to license restrictions.
Please download them directly from the official sources: (https://www.selleckchem.com)
The positive dataset consists of antioxidant compounds (**Positive**) sourced from **Antioxidant-Compound-Library-96-well.xlsx**. 
Since these compounds are also included within the broader **Bioactive-Compound-Library-I-96-well.xlsx**, we define the negative set by subtracting the positive compounds from the total bioactive library.

### 1. Data
   - Raw 
       - Data/01_Raw/antioxi_data_817.smi : known antioxidant compounds ( positive).
       - Data/01_Raw/total_data_9972.smi : Bioactive compounds with unknown antioxidant activity (negative).
       - Data/01_Raw/ZINC_Natural_Products_ADMET_9644.smi : ZINC Natural Products dataset used for external validation.
   - Preprocessed Data
       - Remove_duplication_NaN
         - Data/02_Processed/01_Remove_duplication_NaN/antioxi_data_802.smi : SMILES file containing antioxidant compounds (positive set) after removing duplicates and NaN values, used for descriptor calculation.
         - Data/02_Processed/01_Remove_duplication_NaN/only_neg_data_8898.smi : SMILES file containing non-antioxidant compounds (negative set) after removing duplicates and NaN values, used for descriptor calculation.
         - Data/02_Processed/01_Remove_duplication_NaN/ZINC_Natural_Products_ADMET_5810.smi : ZINC Natural Products ADMET dataset after removing duplicates and NaN values.
       - Filtered_metal_ions
         - Data/02_Preprocessed/02_Filtered_metal_ions/filtered_antioxi_data_776.smi : SMILES file of antioxidant compounds after removing metal ion-containing structures.
         - Data/02_Preprocessed/02_Filtered_metal_ions/filtered_only_neg_data_8443.smi : SMILES file of non-antioxidant compounds after removing metal ion-containing structures.
         - Data/02_Preprocessed/02_Filtered_metal_ions/ZINC_Natural_Products_ADMET_filtered.smi : ZINC dataset after filtering out compounds containing metal ions.
       - Feature_selection
         - top_51_features.txt : The final selected top 51 descriptors.
   - Model_training
       - Data_load
         - Data/03_Model_training/01_Data_load/test_set_438.csv 
         - Data/03_Model_training/01_Data_load/train_cls_1016.csv : Combined dataset of 508 positive compounds and 508 newly sampled negative compounds. 
       - Output
         - Data/03_Model_training/02_Output/best_hyperparameters.csv : Optimal hyperparameters for each machine learning model.
         - Data/03_Model_training/02_Output/performance_scores.csv : Performance metrics (Accuracy, AUC, F1-score, MCC, Specificity) for each model.
         - Data/03_Model_training/02_Output/ZINC_external_predictions_XGB51.csv : Prediction results from the final model (XGBoost Top51) applied to the ZINC dataset.  
   - Without_resampled_strategy
       - Output
         - Data/04_Without_resampled_strategy/01_Output/best_hyperparameters_with_rawdata.csv : Best hyperparameters obtained using the full unbalanced dataset (without resampling).
         - Data/04_Without_resampled_strategy/01_Output/performance_scores_with_rawdata.csv : Performance metrics from training without resampling strategy for robustness testing.
### 2. Model
  - XGBoost_Top51_model.pkl :
    - Final trained model (XGBoost classifier with Top 51 selected descriptors) used for prediction and external validation.
### 3. Notebooks
  - 01_Data_preprocessing&Resampling.ipynb
  - 02_Feature_selection.ipynb
  - 03_Model_training_validation.ipynb
  - 04_Validation_without_resampled.ipynb
---

## How to Run
### 1. Environment Setup
- Install the required Python libraries:
```
pip install -r Requirements.txt
```
  - Python ≥ 3.9
  - Java Runtime Environment (JRE) ≥ 11 (for PaDEL-Descriptor)
  - The required library dependencies for this project are listed below:
```
pandas==2.2.3
numpy==2.2.2
matplotlib==3.10.0
scikit-learn==1.6.1
kneed==0.8.5
xgboost==2.1.4
matplotlib==3.7.1
joblib==1.2.0
openpyxl==3.1.5
padelpy==0.1.16
seaborn==0.13.2
statsmodels==0.14.4
```
**Note:**
- **PadelPy** requires Java Runtime Environment (JRE) to be installed on your system.
- To check Java installation:
```
    java -version
```
- If Java is not installed:
  - **Linux**:`sudo apt-get install default-jre` 
  - **Windows**: Download from Java.com

### 2. Data preprocessing & Resampling
Preprocessing of raw data, removal of duplicates/NaN, filtering metal-containing compounds, calculating molecular descriptors, and generating balanced training sets through resampling.
- Run: Notebooks/01_Data_preprocessing&Resampling.ipynb

### 3. Feature Selection
Feature importance calculation using Random Forest and selection of optimal descriptor subsets via elbow-point method.
- Run: Notebooks/02_Feature_selection.ipynb
- Output: Top51_descriptors.txt

### 4. Model Training & Validation (+ ZINC external)
Model training and evaluation with resampled datasets, including hyperparameter optimization, performance assessment (ACC, AUC, F1, MCC, Specificity),  
and **external validation using the ZINC Natural Products dataset** with the final XGBoost_Top51 model.
- Run: Notebooks/03_Model_training_validation.ipynb
- Outputs:
  -  best_hyperparameters.csv, performance_scores.csv
  -  XGBoost_Top51_model.pkl
  -  ZINC_external_predictions_XGB51.csv
  
### 5. Validation without resampled strategy
Alternative validation strategy where the model is trained and evaluated **without resampling**, using the entire negative pool directly.  
This test ensures the robustness of the proposed resampling strategy by comparing performance under unbalanced conditions.
- Run: Notebooks/04_Validation_without_Resampled.ipynb
- Outputs:
  - best_hyperparameters_with_rawdata.csv
  - performance_scores_with_rawdata.csv

### 6. Compound Screening
Use the trained model to screen new compounds.
```
import joblib
import pandas as pd

# Load saved model
final_model = joblib.load("XGBoost_Top51_model.pkl")

# Load selected descriptor names
with open("top51_descriptors.txt", "r") as f:
    top_51_features = [line.strip() for line in f]

# Load screening data and filtered data
screening_data = pd.read_csv('screening_data.csv')
X = screening_data[top_51_features]

# Predict probabilities and classes
screening_data["Predicted_Probability"] = final_model.predict_proba(X)[:, 1]
screening_data["Predicted_Class"] = final_model.predict(X)

# Sort by predicted probability
screening_df = screening_data[["Predicted_Probability", "Predicted_Class"]]
screening_df = screening_df.sort_values(by="Predicted_Probability", ascending=False)

# Save results
screening_df.to_csv("Screening_predictions_XGB51.csv")
```

## Releases 
- Users can download the file via the provided link.
- Only essential processed data and results are included in this repository.  
- These files can be **re-generated** by following the provided Jupyter notebooks, ensuring reproducibility of the entire workflow.  

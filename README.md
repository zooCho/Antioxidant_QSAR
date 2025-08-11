# QSAR Model for Predicting Antioxidant Activity
This repository contains the full pipeline for developing five machine learning-based QSAR (Quantitative Structure–Activity Relationship) model to classify antioxidant compounds. The workflow includes data preprocessing, molecular descriptor calculation and processing, feature selection, model training, performance evaluation, and external validation using the ZINC Natural Products dataset with the final model.
---

##  Project Overview
- Objective: Build a robust binary classification model to predict antioxidant activity.
- Approach: Train models on a balanced dataset via resampling, evaluate them using cross-validation and test set, and validate generalizability using external ZINC compounds.
- Techniques Used:
  - Feature selection using Random Forest importance
  - Extensive model evaluation (F1, MCC, AUC, Acc, Specificity)
  - Additional validation for robustness testing
    1. screening external compounds using the final model.
    2. training and evaluating models without applying the resampling strategy.
- Dataset:
  - Main data from SellechChem's libraries, including antioxidant compounds categorized as positive and non-antioxidant (negative) compounds.
  - External data from ZINC Natural Products dataset
- Final Model: "XGboost classifier with the Top 51 descriptors"  
---

## Key Componets
### 1. Data
   - Raw data
       - Data/01_Raw/20231011-L6500-Antioxidant-Compound-Library-96-well.xlsx : known antioxidant compounds ( positive).
       - Data/01_Raw/20240327-L1700-Bioactive-Compound-Library-I-96-well.xlsx : Bioactive compounds with unknown antioxidant activity (negative).
       - Data/01_Raw/ZINC_Natural_Products_ADMET_9644.smi : ZINC Natural Products dataset used for external validation.
   - Preprocessed Data
       - Input
         - Data/02_Processed/01_Input/antioxi_data_802.smi : Only antioxidant compounds SMILES

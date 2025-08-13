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

## Components description
### 1. Data
   - Raw data
       - Data/01_Raw/20231011-L6500-Antioxidant-Compound-Library-96-well.xlsx : known antioxidant compounds ( positive).
       - Data/01_Raw/20240327-L1700-Bioactive-Compound-Library-I-96-well.xlsx : Bioactive compounds with unknown antioxidant activity (negative).
       - Data/01_Raw/ZINC_Natural_Products_ADMET_9644.smi : ZINC Natural Products dataset used for external validation.
   - Preprocessed Data
       - Remove_duplication_NaN
         - Data/02_Processed/01_Input/antioxi_data_802.smi : SMILES file containing antioxidant compounds (positive set) after removing duplicates and NaN values, used for descriptor calculation.
         - Data/02_Processed/01_Input/only_neg_data_8898.smi : SMILES file containing non-antioxidant compounds (negative set) after removing duplicates and NaN values, used for descriptor calculation.
       - Filtered_metal_ions
         - Data/02_Preprocessed/02_Filtered_metal_ions/filtered_antioxi_data_776.smi : SMILES file of antioxidant compounds after removing metal ion-containing structures.
         - Data/02_Preprocessed/02_Filtered_metal_ions/filtered_only_neg_data_8443.smi : SMILES file of non-antioxidant compounds after removing metal ion-containing structures.
       - Calculated_descriptors
         - Data/02_Preprocessed/03_Calculated_descriptors/antioxi_des_776.csv : Molecular descriptors calculated from antioxidant compounds after filtering metal ions.
         - Data/02_Preprocessed/03_Calculated_descriptors/negative_des_8443.csv : Molecular descriptors calculated from non-antioxidant compounds after filtering metal ions.
       - Processed_descriptors
         - Data/02_Preprocessed/04_Processed_descriptors/antioxidant_des_dupna_727.csv : Processed antioxidant descriptors after removing duplicates and NaN values.
         - Data/02_Preprocessed/04_Processed_descriptors/negative_des_dupna_6677.csv : Processed non-antioxidant descriptors after removing duplicates and NaN values.
       - Split_train_test
         - Data/02_Preprocessed/05_Split_train_test/train_pos_508.csv : Positive training set.
         - Data/02_Preprocessed/05_Split_train_test/train_total_neg_6458.csv : Total negative training pool.
         - Data/02_Preprocessed/05_Split_train_test/test_set_438.csv : Independent test set for model evaluation.
   - Model_training
       - Data_load
         - Data/03_Model_training/01_Data_load/train_new_neg_508.csv : Randomly sampled 508 negative compounds from the total negative pool.
         - Data/03_Model_training/01_Data_load/train_cls_1016.csv : Combined dataset of 508 positive compounds and 508 newly sampled negative compounds. 
       - Output
         - Data/03_Model_training/02_Output/best_hyperparameters.csv : Optimal hyperparameters for each machine learning model.
         - Data/03_Model_training/02_Output/performance_scores.csv : Performance metrics (Accuracy, AUC, F1-score, MCC, Specificity) for each model.
   - External_validation
       - Remove_duplication_NaN
         - Data/04_External_validation/01_Remove_duplication_NaN/ZINC_Natural_Products_ADMET_5810.smi : 
       - Filtered_metal_ions
         - Data/04_External_validation/02_Filtered_metal_ions/ZINC_Natural_Products_ADMET_filtered.smi :
       - Calculated_descriptors
         - Data/04_External_validation/03_Calculated_descriptors/ZINC_external_des_5810.csv :
       - Processed_descriptors
         - Data/04_External_validation/04_Processed_descriptors/ZINC_external_5790.csv :
       - Validation_final_model
         - Data/04_external_validation/05_Validation_final_model/ZINC_external_predictions_XGB51.csv :   

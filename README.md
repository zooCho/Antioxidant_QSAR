# QSAR Model for Predicting Antioxidant Activity
This repository contains the full pipeline for developing five machine learning-based QSAR (Quantitative Structure–Activity Relationship) model to classify antioxidant compounds. The workflow includes data preprocessing, molecular descriptor calculation and processing, feature selection, model training, performance evaluation, and external validation using the ZINC Natural Products dataset with the final model.

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

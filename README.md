# ML_bugprediction
Software bug detection with different Machine Learning Models

**GENERAL :**
This repository provides the resources and implementation to reproduce the results from the 2024 scientific paper:

"Development of Partial Least Squares Regression with Discriminant Analysis for Software Bug Prediction"
Authors: Róbert Rajkó, István Siket, Péter Hegedűs, Rudolf Ferenc
University of Szeged, Hungary (2024)
DOI: 10.1016/j.heliyon.2024.e35045
Link: https://www.cell.com/heliyon/fulltext/S2405-8440(24)11076-6

The paper presents a method that combines Partial Least Squares (PLS) Regression with Discriminant Analysis to improve software bug prediction. The results are compared to several machine learning and a deep learning model.

**DATA :**
The dataset used in this project is publicly available on Mendeley Data: https://data.mendeley.com/datasets/cb22t5225n/1
The dataset includes:
- Source metrics for software modules
- Bug count labels
- Preprocessed data for PLS regression
- Evaluation scripts and sample outputs for the PLS-DA

**CODE :**
The comparison models referenced in the original paper were not publicly available.
As a result, all baseline models used for evaluation were implemented entirely by me, based on the methodological descriptions in the paper.
These implementations are published in this repository.

The models built are as follows: **Deep Neural Network, Random Forest, K-Neirest Neighbor, Support Vector Machines, Decision Trees, Logistic Regression, Linear Regression, Naive Bayes**

**comparison_functions.py :**
This file serves as the central evaluation engine for the bug prediction study. It takes any classifier and runs a consistent training and evaluation routine, allowing fair comparison across models and resampling strategies. The function supports optional hyperparameter tuning via grid search, applies 10-fold cross-validation, and outputs detailed and summary results to both Excel and CSV files for reproducibility.
Functionally, it calculates several performance metrics (precision, recall, F1-score, and MCC) using a custom scoring function and handles both binary classifiers and regressors (e.g., linear regression with a 0.5 cutoff). Results are stored in structured files to enable easy aggregation and reporting. This modular approach ensures that any model can be plugged in with minimal changes.

**comparison_main.py :**
This script acts as the entry point to the project, handling data loading, preprocessing, model setup, and calling the training pipeline for all classifiers and resampling variants. It first loads the normalized binary dataset, splits it into training and dev sets, and creates three versions: the original, a replication-upsampled set, and a SMOTE-balanced set.
It then defines a variety of classification models—ranging from traditional algorithms (e.g., Logistic Regression, SVM) to a Deep Neural Network built using Keras. Each model is paired with a hyperparameter grid and passed to the training pipeline. This ensures that the same evaluation criteria are used across all configurations and provides a comprehensive picture of how well each model performs under different data balancing strategies.


**CITATION :**
If you use this project or data in your own research, please cite the original paper:

Rajkó, Róbert, István Siket, Péter Hegedűs, and Rudolf Ferenc. 2024. “Development of
partial least squares regression with discriminant analysis for software bug
prediction.” Heliyon 10 (15): 2405-8440. 10.1016/j.heliyon.2024.e35045.

**CONTACT :** 
For questions, suggestions, or issues, feel free to get in touch or open an issue in this repository.

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from imblearn.over_sampling import SMOTE
import os

def upsampling_replication(data):
    bugged = data[data.iloc[:,36] == 1]
    not_bugged = data[data.iloc[:,36] == 0]
    l_bugged = len(bugged)
    l_not_bugged = len(not_bugged)
    extra_rows = l_not_bugged - (l_bugged*4)
    data_upsampled = pd.concat([not_bugged] + [bugged] * 4 + [bugged.iloc[:extra_rows, :]], ignore_index=True)
    data_upsampled_shuffled = data_upsampled.sample(frac=1, random_state=42).reset_index(drop=True)
    return data_upsampled_shuffled


def upsampling_smote(X, y):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X, y)


#class 0 = not bugged = n, class 1 = bugged = p
def calculate_metrics(y_true, y_pred):
    TP = ((y_true == 1) & (y_pred == 1)).sum()
    TN = ((y_true == 0) & (y_pred == 0)).sum()
    FP = ((y_true == 0) & (y_pred == 1)).sum()
    FN = ((y_true == 1) & (y_pred == 0)).sum()

    precision_0 = TN / (TN + FN) if (TN + FN) > 0 else 0
    precision_1 = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall_0 = TN / (TN + FP) if (TN + FP) > 0 else 0
    recall_1 = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
    mcc = (TP * TN - FP * FN) / np.sqrt((TP+FP) * (FN+TN) * (FP+TN) * (TP+FN)) if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) > 0 else 0

    return {
        'Precision (p)': precision_1,
        'Recall (p)': recall_1,
        'F1 (p)': f1_1,
        'Precision (n)': precision_0,
        'Recall (n)': recall_0,
        'F1 (n)': f1_0,
        'MCC': mcc,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    }

def save(data, output_file, sheet_name):
    if os.path.exists(output_file):
        with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
            data.to_excel(writer, index=False, sheet_name=sheet_name)
    else:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            data.to_excel(writer, index=False, sheet_name=sheet_name)


def train_models(clf, name, param_grid, X_model, y_model, X_dev, y_dev, datatype):
    print(f"{name}")

    #metrics storage
    results = []

    #hyperparameters
    N_splits = 10
    random_state = 42

    if param_grid:
        #grid search on dev set, optimizing for f1
        grid = GridSearchCV(clf, param_grid, scoring='f1', cv=3, n_jobs=-1)
        grid.fit(X_dev, y_dev)
        model = grid.best_estimator_

        #best params
        best_params = grid.best_params_
        params_df = pd.DataFrame([{
            'Parameter': key,
            'Value': value,
        } for key, value in best_params.items()])
        save(params_df, f'best_params_{datatype}.xlsx', name)
    else:
        model = clf

    #cross-validation
    kf = KFold(n_splits=N_splits, shuffle=True, random_state=random_state)
    results = []
    i = 1
    for train_idx, val_idx in kf.split(X_model):
        print(f"    {i}. fold")
        X_fold_train, X_fold_val = X_model.iloc[train_idx], X_model.iloc[val_idx]
        y_fold_train, y_fold_val = y_model.iloc[train_idx], y_model.iloc[val_idx]

        model.fit(X_fold_train, y_fold_train)
        pred = model.predict(X_fold_val)
        if name == "Lin. Reg.":
            pred = (pred > 0.5).astype(int) #make binary if linear regression is used
        results.append(calculate_metrics(y_fold_val, pred))

        i += 1

    #average cross-validation results
    results.append(pd.DataFrame(results).mean().to_dict())

    #individual results
    results_df = pd.DataFrame(results)
    save(results_df, f'model_evaluation_results_{datatype}.xlsx', name)

    #summary of results
    summary_mean = results_df.mean()
    summary_mean_row = pd.DataFrame([summary_mean.values], columns=summary_mean.index, index=[f'{name} Mean'])
    summary_mean_row.to_csv(f'summary_results_{datatype}.csv', mode='a', header=not pd.io.common.file_exists(f'summary_results_{datatype}.csv'), index=True)
    summary_std= results_df.std()
    summary_std_row = pd.DataFrame([summary_std.values], columns=summary_std.index, index=[f'{name} Std.'])
    summary_std_row.to_csv(f'summary_results_{datatype}.csv', mode='a', header=not pd.io.common.file_exists(f'summary_results_{datatype}.csv'), index=True)
    
    print(f"Results of {name} saved to model_evaluation_results_{datatype}.xlsx and summary_results.csv")


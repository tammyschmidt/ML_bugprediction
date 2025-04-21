import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from comparison_standard_training import train_models
from comparison_standard_training import upsampling_replication
from comparison_standard_training import upsampling_smote
from sklearn.model_selection import train_test_split


#data    
path = 'unified-bug-dataset-normalized-binary.csv'
data_orig = pd.read_csv(path)

#feature and target extraction
X_orig = data_orig.iloc[:,:34] #columns 1-36
y_orig = data_orig.iloc[:,36] #column 37

#data split into 90% model and 10% "dev set" (p.11)
X_orig_model, X_orig_dev, y_orig_model, y_orig_dev = train_test_split(X_orig, y_orig, test_size=0.1, stratify=y_orig, random_state=42)
input_dim_orig = X_orig_dev.shape[1]

#balancing
#replication-based
data_rep = upsampling_replication(data_orig)
X_rep = data_rep.iloc[:,:34] #columns 1-36
y_rep = data_rep.iloc[:,36] #column 37
X_rep_model, X_rep_dev, y_rep_model, y_rep_dev = train_test_split(X_rep, y_rep, test_size=0.1, stratify=y_rep, random_state=42)
input_dim_rep = X_rep_dev.shape[1]

#smote
X_smote, y_smote = upsampling_smote(X_orig, y_orig)
X_smote_model, X_smote_dev, y_smote_model, y_smote_dev = train_test_split(X_smote, y_smote, test_size=0.1, stratify=y_smote, random_state=42)
input_dim_smote = X_smote_dev.shape[1]

#Random Forest
rf = RandomForestClassifier(random_state=42)
rf_name = "RF"
rf_param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

#K-Nearest Neighbors
knn = KNeighborsClassifier()
knn_name = "KNN"
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9],        
    'weights': ['uniform', 'distance'], 
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

#SVM
svm = SVC(random_state=42)
svm_name = "SVM"
svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

#Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt_name = "Dec. Tree"
dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],   
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 5]
}

#Logistic Regression
logr = LogisticRegression(random_state=42, solver='liblinear')
logr_name = "Log. Reg."
logr_param_grid = {
    'C': [0.1, 1, 10, 100],            
    'penalty': ['l1', 'l2']
}

#Linear Regression
lin_reg = LinearRegression()
lin_reg_name = "Lin. Reg."
lin_reg_param_grid = {} #no applicable parameters

#Naive Bayes (a type of Bayesian classifier)
bayes = GaussianNB()
bayes_name = "Bayes"
bayes_param_grid = {} #no parameters

#DNN
def create_DNN(optimizer='adam', input_dim=None):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_dim=input_dim),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])  #binary classification
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['f1_score'])
    return model

dnn_orig = KerasClassifier(build_fn=create_DNN, input_dim=input_dim_orig) #wraper for use with scikit-learn
dnn_rep = KerasClassifier(build_fn=create_DNN, input_dim=input_dim_rep) #wraper for use with scikit-learn
dnn_smote = KerasClassifier(build_fn=create_DNN, input_dim=input_dim_smote) #wraper for use with scikit-learn
dnn_name = "DNN"
dnn_param_grid = {
    'batch_size': [16, 32, 64],
    'epochs': [10, 20],
    'optimizer': ['adam', 'sgd']
}


"""train_models(rf, rf_name, rf_param_grid, X_orig_model, y_orig_model, X_orig_dev, y_orig_dev, "orig")
train_models(knn, knn_name, knn_param_grid, X_orig_model, y_orig_model, X_orig_dev, y_orig_dev, "orig")
train_models(svm, svm_name, svm_param_grid, X_orig_model, y_orig_model, X_orig_dev, y_orig_dev, "orig")
train_models(dt, dt_name, dt_param_grid, X_orig_model, y_orig_model, X_orig_dev, y_orig_dev, "orig")
train_models(logr, logr_name, logr_param_grid, X_orig_model, y_orig_model, X_orig_dev, y_orig_dev, "orig")
train_models(lin_reg, lin_reg_name, lin_reg_param_grid, X_orig_model, y_orig_model, X_orig_dev, y_orig_dev, "orig")
train_models(bayes, bayes_name, bayes_param_grid, X_orig_model, y_orig_model, X_orig_dev, y_orig_dev, "orig")
train_models(dnn_orig, dnn_name, dnn_param_grid, X_orig_model, y_orig_model, X_orig_dev, y_orig_dev, "orig")
 
train_models(rf, rf_name, rf_param_grid, X_rep_model, y_rep_model, X_rep_dev, y_rep_dev, "rep")
train_models(knn, knn_name, knn_param_grid, X_rep_model, y_rep_model, X_rep_dev, y_rep_dev, "rep")
train_models(svm, svm_name, svm_param_grid, X_rep_model, y_rep_model, X_rep_dev, y_rep_dev, "rep")
train_models(dt, dt_name, dt_param_grid, X_rep_model, y_rep_model, X_rep_dev, y_rep_dev, "rep")
train_models(logr, logr_name, logr_param_grid, X_rep_model, y_rep_model, X_rep_dev, y_rep_dev, "rep")
train_models(lin_reg, lin_reg_name, lin_reg_param_grid, X_rep_model, y_rep_model, X_rep_dev, y_rep_dev, "rep")
train_models(bayes, bayes_name, bayes_param_grid, X_rep_model, y_rep_model, X_rep_dev, y_rep_dev, "rep")
train_models(dnn_rep, dnn_name, dnn_param_grid, X_rep_model, y_rep_model, X_rep_dev, y_rep_dev, "rep")"""

train_models(rf, rf_name, rf_param_grid, X_smote_model, y_smote_model, X_smote_dev, y_smote_dev, "smote")
train_models(knn, knn_name, knn_param_grid, X_smote_model, y_smote_model, X_smote_dev, y_smote_dev, "smote")
train_models(svm, svm_name, svm_param_grid, X_smote_model, y_smote_model, X_smote_dev, y_smote_dev, "smote")
train_models(dt, dt_name, dt_param_grid, X_smote_model, y_smote_model, X_smote_dev, y_smote_dev, "smote")
train_models(logr, logr_name, logr_param_grid, X_smote_model, y_smote_model, X_smote_dev, y_smote_dev, "smote")
train_models(lin_reg, lin_reg_name, lin_reg_param_grid, X_smote_model, y_smote_model, X_smote_dev, y_smote_dev, "smote")
train_models(bayes, bayes_name, bayes_param_grid, X_smote_model, y_smote_model, X_smote_dev, y_smote_dev, "smote")
train_models(dnn_smote, dnn_name, dnn_param_grid, X_smote_model, y_smote_model, X_smote_dev, y_smote_dev, "smote")

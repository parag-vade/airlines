"""
Importing the required libraries
"""

from pandas import read_csv, get_dummies, Series, DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score,make_scorer
from imblearn.pipeline import Pipeline
import warnings
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

air_data=data1=read_csv('/content/drive/MyDrive/Akasa_airlines.csv')

air_data.shape

air_data.head()

air_data.info()

air_data.isna().sum() # check for null values

air_data['Arrival_Delay_in_Minutes'] = air_data['Arrival_Delay_in_Minutes'].fillna(air_data['Arrival_Delay_in_Minutes'].median()) # removing the null values

air_data.isna().sum() # check for null values

#printing the unique values in all the categorical columns
print("Unique Columns values in Gender - ",air_data['Gender'].unique())
print("Unique Columns values in Customer_Type - ",air_data['Customer_Type'].unique())
print("Unique Columns values in Type_of_Travel - ",air_data['Type_of_Travel'].unique())
print("Unique Columns values in Class - ",air_data['Class'].unique())
print("Unique Columns values in satisfaction - ",air_data['satisfaction'].unique())

air_data['Gender'] = air_data['Gender'].map({'Male':1, 'Female':0}) #encoding

air_data['Customer_Type'] = air_data['Customer_Type'].map({'Loyal Customer':1, 'disloyal Customer':0}) #encoding

air_data['Type_of_Travel'] = air_data['Type_of_Travel'].map({'Business travel':1, 'Personal Travel':0})#encoding

air_data['satisfaction']=air_data['satisfaction'].map({'dissatisfied':1,'satisfied':0})
#encoding dissatisfied as 1 because our business problem is to reduce churning of customers

air_data['Class']=air_data['Class'].map({'Eco':0,'Eco Plus':1,'Business': 2})#encoding

air_data.head()

air_data.info()

from plotly import graph_objs, figure_factory

#Heat map is drawn to understand the important features required for the model
correlation  = air_data.corr()
f = figure_factory.create_annotated_heatmap(correlation.values,list(correlation.columns),list(correlation.columns),correlation.round(2).values,showscale=False)
f.show()

#dropping the non required features
X = air_data.drop(['id', 'Gender','Departure/Arrival_time_convenient','Ease_of_Online_booking','Gate_location', 'Food_and_drink','Baggage_handling','Departure_Delay_in_Minutes','Arrival_Delay_in_Minutes','satisfaction'], axis = 1) # Features
Y = air_data['satisfaction'] #Creating a target variable/feature

print(type(X))
print(type(Y))
print(X.shape)
print(Y.shape)

# Create a StandardScaler instance, which will standardize (scale) your features
feature_scaler = StandardScaler()

# Use the fit_transform method of the StandardScaler to standardize your feature matrix X
X_scaled = feature_scaler.fit_transform(X)

# Now, X_scaled contains the scaled (standardized) features, which have a mean of 0 and a standard deviation of 1

"""Random Forest Classifier with recall score"""

#feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#SMOTE oversampling
smote = SMOTE(random_state=101)
X_resampled, Y_resampled = smote.fit_resample(X_scaled, Y)

# Define the model pipeline
model_pipeline = Pipeline([
    ('classification', RandomForestClassifier(criterion='entropy', max_features='auto', random_state=1))
])

# Define the parameter grid for hyperparameter tuning (number of estimators)
param_grid = {
    'classification__n_estimators': [100, 200, 300]
}

# GridSearchCV for hyperparameter tuning with both recall and precision scoring and 5-fold cross-validation
scoring = {
    'recall': make_scorer(recall_score),
    'precision': make_scorer(precision_score, zero_division=1)
}

grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring=scoring,
    refit=False,
    cv=5
)

# Fit the model using the grid search with resampled data
grid_search.fit(X_resampled, Y_resampled)

# Find the index with the best recall and precision scores
best_recall_index = np.argmax(grid_search.cv_results_['mean_test_recall'])
best_precision_index = np.argmax(grid_search.cv_results_['mean_test_precision'])

# Extract the best parameters and scores
best_parameters_recall = grid_search.cv_results_['params'][best_recall_index]
best_parameters_precision = grid_search.cv_results_['params'][best_precision_index]

best_recall = grid_search.cv_results_['mean_test_recall'][best_recall_index]
best_precision = grid_search.cv_results_['mean_test_precision'][best_precision_index]

# Print the best parameters, best recall, and best precision scores
print("Best Parameters (Recall):", best_parameters_recall)
print("Best Recall:", best_recall)
print("Best Parameters (Precision):", best_parameters_precision)
print("Best Precision:", best_precision)

"""AdaBoost Classifier with Recall Score"""

#feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#SMOTE oversampling
smote = SMOTE(random_state=101)
X_resampled, Y_resampled = smote.fit_resample(X_scaled, Y)

# Creating a pipeline with the AdaBoostClassifier
Adaboost_classifier = AdaBoostClassifier(random_state=1)
pipeline = Pipeline([
    ('classification', Adaboost_classifier)
])

# Define the parameter grid for hyperparameters (number of estimators)
param_grid = {
    'classification__n_estimators': [25, 50, 100]
}

# Create GridSearchCV for hyperparameter tuning with both recall and precision scoring and 5-fold cross-validation
scoring = {
    'recall': make_scorer(recall_score),
    'precision': make_scorer(precision_score, zero_division=1)
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring=scoring,
    refit=False,
    cv=5
)

# Fit the model using the grid search with resampled data
grid_search.fit(X_resampled, Y_resampled)

# Get the best parameters, best recall, and best precision scores for both metrics
best_recall_params = None
best_recall_score = -1
best_precision_params = None
best_precision_score = -1

for i in range(len(grid_search.cv_results_['params'])):
    if grid_search.cv_results_['mean_test_recall'][i] > best_recall_score:
        best_recall_score = grid_search.cv_results_['mean_test_recall'][i]
        best_recall_params = grid_search.cv_results_['params'][i]

    if grid_search.cv_results_['mean_test_precision'][i] > best_precision_score:
        best_precision_score = grid_search.cv_results_['mean_test_precision'][i]
        best_precision_params = grid_search.cv_results_['params'][i]

# Print the best parameters, best recall, and best precision scores for both metrics
print("Best Parameters (Recall):", best_recall_params)
print("Best Recall:", best_recall_score)
print("Best Parameters (Precision):", best_precision_params)
print("Best Precision:", best_precision_score)

"""Support Vector classifier with recall score"""

#feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#SMOTE oversampling
smote = SMOTE(random_state=101)
X_resampled, Y_resampled = smote.fit_resample(X_scaled, Y)

# Create a pipeline with an SVM (Support Vector Machine) for classification
model = Pipeline([
    ('classification', SVC(random_state=1))  # Use SVM for classification
])

# Define the parameter grid for hyperparameters, including 'kernel' and 'C' for SVM
grid_param = {
    'classification__kernel': ['linear'],  # Use a linear kernel
    'classification__C': [0.01, 0.1, 1, 10]  # Test different values of the regularization parameter C
}

# Create a GridSearchCV object for both recall and precision
scoring = {
    'recall': make_scorer(recall_score),
    'precision': make_scorer(precision_score, zero_division=1)
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=grid_param,
    scoring=scoring,  # Using both recall and precision as scoring metrics
    cv=5,  # 5-fold cross-validation
    refit=False  # Set refit to False explicitly to handle multi-metric scoring
)

# Fit the model with the resampled data for both recall and precision
grid_search.fit(X_resampled, Y_resampled)

# Retrieve the best parameters and best recall and precision scores
best_parameters_recall = grid_search.cv_results_['params'][grid_search.cv_results_['rank_test_recall'].argmin()]
best_recall_score = grid_search.cv_results_['mean_test_recall'][grid_search.cv_results_['rank_test_recall'].argmin()]

best_parameters_precision = grid_search.cv_results_['params'][grid_search.cv_results_['rank_test_precision'].argmin()]
best_precision_score = grid_search.cv_results_['mean_test_precision'][grid_search.cv_results_['rank_test_precision'].argmin()]

# Print the best parameters and scores for both recall and precision
print("Best Parameters (SVM - Recall):", best_parameters_recall)
print("Best Recall Score:", best_recall_score)
print("Best Parameters (SVM - Precision):", best_parameters_precision)
print("Best Precision Score:", best_precision_score)

"""Logistic regression"""

#Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform SMOTE oversampling
smote = SMOTE(random_state=101)
X_resampled, Y_resampled = smote.fit_resample(X_scaled, Y)

# Create a pipeline with an SGDClassifier for classification
model = Pipeline([
    ('classification', SGDClassifier(loss='log', penalty='elasticnet', random_state=1))
])

# Define the parameter grid for hyperparameter tuning
grid_param = {
    'classification__eta0': [0.1, 1, 10],  # Learning rate values
    'classification__max_iter': [100, 500, 1000],  # Maximum number of iterations
    'classification__alpha': [0.01, 0.1, 1, 10],  # Regularization strength
    'classification__l1_ratio': [0, 0.5, 1]  # Elastic Net mixing parameter
}

# Create a GridSearchCV object for hyperparameter tuning with both 'recall' and 'precision' scoring
scoring = {
    'recall': make_scorer(recall_score),
    'precision': make_scorer(precision_score, zero_division=1)  # Set zero_division to control behavior
}

grid_search = GridSearchCV(
    estimator=model,  # Use the defined pipeline
    param_grid=grid_param,  # Use the defined parameter grid
    scoring=scoring,  # Optimize for both recall and precision
    cv=5,  # Use 5-fold cross-validation
    refit=False  # Set refit to False to prevent refitting
)

# Fit the model using the grid search
grid_search.fit(X_resampled, Y_resampled)

# Get the best parameters for recall and precision
best_parameters_recall = grid_search.cv_results_['params'][np.argmin(grid_search.cv_results_['rank_test_recall'])]
best_parameters_precision = grid_search.cv_results_['params'][np.argmin(grid_search.cv_results_['rank_test_precision'])]

print("Best Parameters for Recall:", best_parameters_recall)
print("Best Parameters for Precision:", best_parameters_precision)

# Get the best recall and precision scores from the grid search results
best_recall = np.max(grid_search.cv_results_['mean_test_recall'])
best_precision = np.max(grid_search.cv_results_['mean_test_precision'])

print("Best Recall Score:", best_recall)
print("Best Precision Score:", best_precision)

"""### **XGBoost Classifier**"""

#SMOTE Oversampling
smote = SMOTE(random_state=101)
X_resampled, Y_resampled = smote.fit_resample(X_scaled, Y)

# Define the XGBoost classifier pipeline
xgb_classifier = XGBClassifier(random_state=1)
pipeline = Pipeline([
    ('classification', xgb_classifier)
])

# Define the parameter grid for hyperparameters, including 'n_estimators' and 'learning_rate'
param_grid = {
    'classification__n_estimators': [175, 180, 185],
    'classification__learning_rate': [0.1, 1, 10],
    'classification__alpha': [0, 0.5, 1],  # L1 regularization term (alpha)
    'classification__lambda': [0, 0.5, 1]  # L2 regularization term (lambda)
}

# Create a GridSearchCV object for both recall and precision
scoring = {
    'recall': make_scorer(recall_score),
    'precision': make_scorer(precision_score, zero_division=1)
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring=scoring,  # Using both recall and precision as scoring metrics
    cv=5,  # 5-fold cross-validation
    refit=False  # Set refit to False explicitly to prevent the ValueError
)

# Fit the XGBoost model with the resampled data
grid_search.fit(X_resampled, Y_resampled)

# Retrieve the best parameters and best scores for both recall and precision
best_parameters_recall = grid_search.cv_results_['params'][grid_search.cv_results_['rank_test_recall'].argmin()]
best_recall_score = grid_search.cv_results_['mean_test_recall'][grid_search.cv_results_['rank_test_recall'].argmin()]

best_parameters_precision = grid_search.cv_results_['params'][grid_search.cv_results_['rank_test_precision'].argmin()]
best_precision_score = grid_search.cv_results_['mean_test_precision'][grid_search.cv_results_['rank_test_precision'].argmin()]

# Print the best parameters and best scores for both recall and precision
print("Best Parameters (XGBoost - Recall):", best_parameters_recall)
print("Best Recall Score:", best_recall_score)

print("Best Parameters (XGBoost - Precision):", best_parameters_precision)
print("Best Precision Score:", best_precision_score)
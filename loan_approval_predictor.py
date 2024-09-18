# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import plotly.express as px
import plotly.graph_objects as go
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')  # To suppress warnings for cleaner output

# Load and Preprocess the Dataset

# Load dataset
df = pd.read_csv('loan_prediction.csv')

# Drop 'Loan_ID' column
df = df.drop('Loan_ID', axis=1)

# Handle missing values
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']:
    df[col].fillna(df[col].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

# Data Visualization

# Function to create and show plots
def create_plots(df):
    # Loan Status Pie Chart
    loan_status_count = df['Loan_Status'].value_counts()
    fig_loan_status = px.pie(loan_status_count, 
                             names=loan_status_count.index, 
                             title='Loan Approval Status')
    fig_loan_status.show()
    
    # Gender Distribution Bar Chart
    gender_count = df['Gender'].value_counts()
    fig_gender = px.bar(x=gender_count.index, 
                        y=gender_count.values, 
                        labels={'x': 'Gender', 'y': 'Count'},
                        title='Gender Distribution')
    fig_gender.show()
    
    # Marital Status Distribution Bar Chart
    married_count = df['Married'].value_counts()
    fig_married = px.bar(x=married_count.index, 
                         y=married_count.values, 
                         labels={'x': 'Married', 'y': 'Count'},
                         title='Marital Status Distribution')
    fig_married.show()
    
    # Education Distribution Bar Chart
    education_count = df['Education'].value_counts()
    fig_education = px.bar(x=education_count.index, 
                           y=education_count.values, 
                           labels={'x': 'Education', 'y': 'Count'},
                           title='Education Distribution')
    fig_education.show()
    
    # Self-Employment Distribution Bar Chart
    self_employed_count = df['Self_Employed'].value_counts()
    fig_self_employed = px.bar(x=self_employed_count.index, 
                               y=self_employed_count.values, 
                               labels={'x': 'Self-Employed', 'y': 'Count'},
                               title='Self-Employment Distribution')
    fig_self_employed.show()
    
    # Applicant Income Distribution Histogram
    fig_applicant_income = px.histogram(df, x='ApplicantIncome', 
                                        title='Applicant Income Distribution')
    fig_applicant_income.show()
    
    # Loan Status vs Applicant Income Box Plot
    fig_income = px.box(df, x='Loan_Status', 
                        y='ApplicantIncome',
                        color="Loan_Status", 
                        title='Loan_Status vs ApplicantIncome')
    fig_income.show()
    
    # Loan Status vs Coapplicant Income Box Plot
    fig_coapplicant_income = px.box(df, 
                                    x='Loan_Status', 
                                    y='CoapplicantIncome',
                                    color="Loan_Status", 
                                    title='Loan_Status vs CoapplicantIncome')
    fig_coapplicant_income.show()
    
    # Loan Status vs Loan Amount Box Plot
    fig_loan_amount = px.box(df, x='Loan_Status', 
                             y='LoanAmount', 
                             color="Loan_Status",
                             title='Loan_Status vs LoanAmount')
    fig_loan_amount.show()
    
    # Loan Status vs Credit History Histogram
    fig_credit_history = px.histogram(df, x='Credit_History', color='Loan_Status', 
                                      barmode='group',
                                      title='Loan_Status vs Credit_History')
    fig_credit_history.show()
    
    # Loan Status vs Property Area Histogram
    fig_property_area = px.histogram(df, x='Property_Area', color='Loan_Status', 
                                     barmode='group',
                                     title='Loan_Status vs Property_Area')
    fig_property_area.show()

# Initial Plots
create_plots(df)

# Outlier Removal

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    initial_shape = df.shape
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    final_shape = df_filtered.shape
    print(f"Removed {initial_shape[0] - final_shape[0]} outliers from {column}")
    return df_filtered

# Remove outliers for ApplicantIncome
df = remove_outliers(df, 'ApplicantIncome')

# Remove outliers for CoapplicantIncome
df = remove_outliers(df, 'CoapplicantIncome')

# Remove outliers for LoanAmount
df = remove_outliers(df, 'LoanAmount')

# Updated Plots after Outlier Removal
create_plots(df)


# Identify categorical columns
cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

# One-Hot Encode categorical variables
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Encode target variable
le = LabelEncoder()
df['Loan_Status'] = le.fit_transform(df['Loan_Status'])

# Split Dataset into Features and Target

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    stratify=y, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Preprocessing Pipeline

# Define numerical and categorical columns
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                  'Loan_Amount_Term', 'Credit_History']
categorical_cols = [col for col in X_train.columns if col not in numerical_cols]

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', StandardScaler())
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', 'passthrough', categorical_cols)
    ])

# Model Training and Evaluation

# Define Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Define models to evaluate
models = {
    'SVM': Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC(random_state=42, probability=True))
    ]),
    'RandomForest': Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
}

# Define hyperparameter grids
param_grids = {
    'SVM': {
        'classifier__C': [0.1, 1, 10],
        'classifier__gamma': [1, 0.1, 0.01],
        'classifier__kernel': ['rbf', 'poly', 'sigmoid']
    },
    'RandomForest': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
}

# To store best models and their performance
best_models = {}
model_performance = {}

for model_name in models:
    print(f"\nTraining and tuning {model_name}...")
    
    # Create a pipeline with SMOTE and the model
    pipeline = ImbPipeline(steps=[
        ('smote', smote),
        ('preprocessor', preprocessor),
        ('classifier', models[model_name].named_steps['classifier'])
    ])
    
    # Define RandomizedSearchCV
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grids[model_name],
        n_iter=50,
        cv=skf,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the model
    search.fit(X_train, y_train)
    
    # Save the best model
    best_models[model_name] = search.best_estimator_
    
    # Predictions
    y_pred = search.predict(X_test)
    
    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, search.predict_proba(X_test)[:,1])
    
    # Store performance
    model_performance[model_name] = {
        'Accuracy': accuracy,
        'ROC_AUC': roc_auc,
        'Classification Report': report,
        'Confusion Matrix': conf_matrix
    }
    
    # Print evaluation
    print(f"Best Parameters for {model_name}: {search.best_params_}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Plot ROC Curve
    y_probs = search.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve ({model_name})'))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random Guess', line=dict(dash='dash')))
    fig.update_layout(title=f'ROC Curve - {model_name}',
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate')
    fig.show()

# Compare Model Performances

# Create a summary DataFrame
performance_df = pd.DataFrame({
    model: {
        'Accuracy': metrics['Accuracy'],
        'ROC_AUC': metrics['ROC_AUC']
    } for model, metrics in model_performance.items()
}).T

print("\nModel Performance Summary:")
print(performance_df)

# Visualize Model Performance
fig_perf = px.bar(performance_df.reset_index(), 
                  x='index', 
                  y=['Accuracy', 'ROC_AUC'],
                  barmode='group',
                  labels={'index': 'Model'},
                  title='Model Comparison')
fig_perf.show()

# Select the Best Model and Final Evaluation

# Assuming RandomForest performed better
best_model_name = performance_df['ROC_AUC'].idxmax()
best_model = best_models[best_model_name]

print(f"\nBest Model: {best_model_name}")

# Final Predictions and Evaluation
y_final_pred = best_model.predict(X_test)
print("Final Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_final_pred))
print("Classification Report:\n", classification_report(y_test, y_final_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_final_pred))

# Adding predictions to X_test for inspection
X_test_df = X_test.copy()
X_test_df['Loan_Status_Predicted'] = y_final_pred
print("\nSample Predictions:")
print(X_test_df.head())

# Save the best model for future use
import joblib
joblib.dump(best_model, 'best_loan_prediction_model.pkl')
print("\nBest model saved as 'best_loan_prediction_model.pkl'")

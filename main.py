import joblib
import pandas as pd
from data_utils import load_data, preprocess_data, plot_count, plot_histogram, plot_roc_auc
from model_utils import split_data, train_knn, train_rf, train_xgb, create_ensemble, train_ensemble, evaluate_model

# Load and preprocess data
data = load_data("bank_dataset.csv")
X, y, df_fraud, df_non_fraud, mappings = preprocess_data(data)

# print(X)

# Save the mappings
joblib.dump(mappings, 'categorical_mappings.pkl')

print(data.head(5))

# Visualize data
plot_count(data)
plot_histogram(df_fraud, df_non_fraud)

# Split data
X_train, X_test, y_train, y_test = split_data(X, y)

# Train individual models
knn = train_knn(X_train, y_train)
rf_clf = train_rf(X_train, y_train)
XGBoost_CLF = train_xgb(X_train, y_train)

# Evaluate individual models
print("Evaluating K-Nearest Neighbours")
evaluate_model(knn, X_test, y_test)
plot_roc_auc(y_test, knn.predict_proba(X_test)[:, 1])

print("Evaluating Random Forest Classifier")
evaluate_model(rf_clf, X_test, y_test)
plot_roc_auc(y_test, rf_clf.predict_proba(X_test)[:, 1])

print("Evaluating XGBoost Classifier")
evaluate_model(XGBoost_CLF, X_test, y_test)
plot_roc_auc(y_test, XGBoost_CLF.predict_proba(X_test)[:, 1])

# Create and train ensemble model
ensemble = create_ensemble(knn, rf_clf, XGBoost_CLF)
trained_ensemble = train_ensemble(ensemble, X_train, y_train)

# Save the trained ensemble model
joblib.dump(trained_ensemble, 'trained_ensemble_model.pkl')

# Evaluate ensemble model
print("Evaluating Ensemble Model")
evaluate_model(trained_ensemble, X_test, y_test)
plot_roc_auc(y_test, trained_ensemble.predict_proba(X_test)[:, 1])

# Base score for comparison
print("Base score we must beat is: ", df_non_fraud.fraud.count() / (df_non_fraud.fraud.count() + df_fraud.fraud.count()) * 100)

# # Load the saved ensemble model
# loaded_model = joblib.load('trained_ensemble_model.pkl')

# # Load the saved mappings
# mappings = joblib.load('categorical_mappings.pkl')

# def preprocess_new_input(new_input, mappings):

#     col_categorical = new_input.select_dtypes(include=['object']).columns
#     for col in col_categorical:
#         mapping = mappings[col]
#         # Assign a unique identifier for unseen categories, here we use max value + 1
#         new_input[col] = new_input[col].map({v: k for k, v in mapping.items()}).fillna(len(mapping)).astype(int)

#     return new_input

# # Example new input feature
# new_input_data = pd.DataFrame([{
#     'customer': 'C450949430',
#     'age': '2',
#     'gender': 'M',
#     'merchant': 'M1823072687',
#     'category': 'es_transportation',
#     'amount': 287.42
# }])

# # Preprocess the new input data
# new_input_processed = preprocess_new_input(new_input_data, mappings)
# print(new_input_processed)

# # Predict fraud or not
# prediction = loaded_model.predict(new_input_processed)
# print("Prediction for new input: ", "Fraud" if prediction[0] == 1 else "Not Fraud")

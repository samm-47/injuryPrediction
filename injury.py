import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score

# Load dataset
df = pd.read_csv('injury_data.csv')


print(df.head())
print(df.isnull().sum())  # Check for missing values


X = df.drop('Likelihood_of_Injury', axis=1)  # Features
y = df['Likelihood_of_Injury']  # Target variable

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=250,        
    max_depth=12,            
    min_samples_split=8,    
    min_samples_leaf=3,      
    max_features='sqrt',     
    random_state=42      
)

rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest model
print("Random Forest Classifier:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))

cv_scores = cross_val_score(rf_model, X_scaled, y, cv=5)  # 5-fold cross-validation
print(f"Cross-validation accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# Feature importance visualization
feature_importances = rf_model.feature_importances_
features = X.columns

# Sort features by importance
indices = np.argsort(feature_importances)[::-1]

# Extracting feature importances
importances = rf_model.feature_importances_
features = X.columns  # Assuming X is your features dataframe

# dataframe of features and their corresponding importance
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
})

# Sorting the features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print(feature_importance_df)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importance (Random Forest)")
plt.barh(range(len(features)), feature_importances[indices], align="center")
plt.yticks(range(len(features)), features[indices])
plt.xlabel("Relative Importance")
plt.gca().invert_yaxis()  #  most important feature on top
plt.show()

# Logistic Regression 
log_model = LogisticRegression(random_state=42)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# Evaluate model
print("\nLogistic Regression Model:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log):.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_log))


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import joblib
# Load the dataset
df = pd.read_csv('dataset/modified dataset.csv')

# Split into features and target
X = df.drop('target', axis=1)
y = df['target']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training model based on Random Forest
# Building and Fitting RF model
random_forest = RandomForestClassifier()
random_forest = random_forest.fit(X_train, y_train)

# Accuracy score for Random Forest
rf_acc = accuracy_score(y_test, random_forest.predict(X_test))
print(f'RandomForestClassifier Accuracy: {rf_acc:.4}')
print()

# Predict class for X_test
y_pred_rf = random_forest.predict(X_test)

# Classification Report of Random Forest model
print(classification_report(y_pred_rf, y_test))


# Grid search for RF
params_rf = {'max_depth': [2, 3, 4, 5],
             'max_features': ['auto', 'sqrt', 'log2'],
             'n_estimators':[0, 10, 50],
             'random_state': [0, 10, 42]}

# Build and fit Random Forest model with the best hyperparameters
random_forest_gscv = GridSearchCV(RandomForestClassifier(), params_rf, cv=5)
random_forest_gscv = random_forest_gscv.fit(X_train, y_train)

# Accuracy score for rf_gscv
rf_gscv_acc = accuracy_score(y_test, random_forest_gscv.predict(X_test))
print(f'Random Forest with GridSearchCV Accuracy: {rf_gscv_acc:.4}')
print()

# Make prediction on test dataset
y_pred_rf_gscv = random_forest_gscv.predict(X_test)

# Classification Report of grid_rf_model
print(classification_report(y_pred_rf_gscv, y_test))
joblib.dump(random_forest, 'dataset/random_forest.joblib')

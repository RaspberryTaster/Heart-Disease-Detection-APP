import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump

df = pd.read_csv('dataset/modified dataset.csv')
df.columns = ['Age', 'Sex', 'Chest_pain_type', 'Resting_bp', 'Cholesterol', 'Fasting_bs', 'Max_heart_rate', 'ST_slope', 'Condition']

X = df.drop(['Condition'], axis=1)
y = df.Condition
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4}')


dump(model, 'models/heart_disease_model.joblib')

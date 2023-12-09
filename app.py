from flask import Flask, render_template, request

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
import plotly.graph_objs as go
import plotly.offline as py

# For Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# For Model Building
from joblib import dump
from joblib import load
from xgboost import XGBClassifier
# For Model Evaluation

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the trained model
model = load('models/heart_disease_model.joblib')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    chest_pain_type = int(request.form['chest_pain_type'])
    resting_bp = int(request.form['resting_bp'])
    cholesterol = int(request.form['cholesterol'])
    fasting_bs = int(request.form['fasting_bs'])
    max_heart_rate = int(request.form['max_heart_rate'])
    st_slope = int(request.form['st_slope'])
    prediction = model.predict([[age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, max_heart_rate, st_slope]])
    
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run()

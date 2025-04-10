from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
from sklearn.linear_model import LinearRegression
import pickle
import pandas as pd
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # CORS enabled

# Load model and dataset
model_path = 'C:/Users/ANKIT/OneDrive/Desktop/Car/car_price_predictor/LinearRegressionModel.pkl'
data_path = 'C:/Users/ANKIT/OneDrive/Desktop/Car/car_price_predictor/Cleaned_Car_data.csv'

# Check if model and data exist
if not os.path.exists(model_path) or not os.path.exists(data_path):
    raise FileNotFoundError("Model or dataset file not found. Check your paths.")

model = pickle.load(open(model_path, 'rb'))
car = pd.read_csv(data_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')

    return render_template('index.html',
                           companies=companies,
                           car_models=car_models,
                           years=years,
                           fuel_types=fuel_types)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        company = request.form.get('company')
        car_model = request.form.get('car_models')
        year = int(request.form.get('year'))
        fuel_type = request.form.get('fuel_type')
        driven = int(request.form.get('kilo_driven'))

        # Prepare input for model
        input_df = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                data=[[car_model, company, year, driven, fuel_type]])

        prediction = model.predict(input_df)
        price = round(prediction[0], 2)

        return str(price)

    except Exception as e:
        print("Prediction error:", str(e))
        return "Error in prediction: " + str(e)

if __name__ == '__main__':
    app.run(debug=True)

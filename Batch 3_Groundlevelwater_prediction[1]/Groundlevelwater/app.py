from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model and data
model = joblib.load('groundwater_model.pkl')  # Pretrained model
data = pd.read_csv('District_Statewise_Well.csv')  # Original dataset

# Home page to render the form for prediction
@app.route('/')
def index():
    states = data['Name of State'].unique()
    return render_template('index.html', states=states)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    state = request.form['state']
    district = request.form['district']
    
    # Extract relevant district data
    district_data = data[(data['Name of State'] == state) & (data['Name of District'] == district)]
    
    if not district_data.empty:
        # Prepare input features for prediction
        input_features = district_data.iloc[0][['Recharge from rainfall During Monsoon Season',
                                                'Recharge from other sources During Monsoon Season',
                                                'Recharge from rainfall During Non Monsoon Season',
                                                'Recharge from other sources During Non Monsoon Season',
                                                'Total Annual Ground Water Recharge',
                                                'Total Natural Discharges',
                                                'Current Annual Ground Water Extraction For Irrigation',
                                                'Current Annual Ground Water Extraction For Domestic & Industrial Use']].values.reshape(1, -1)
        
        # Predict groundwater availability
        prediction = model.predict(input_features)[0]
        
        return render_template('result.html', state=state, district=district, prediction=prediction)
    else:
        return render_template('index.html', error="District not found!", states=data['Name of State'].unique())

if __name__ == '__main__':
    app.run(debug=True)
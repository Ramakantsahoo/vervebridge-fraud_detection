from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and mappings
model = joblib.load('trained_ensemble_model.pkl')
mappings = joblib.load('categorical_mappings.pkl')

def preprocess_new_input(new_input, mappings):
    # Ensure only categorical columns are encoded
    col_categorical = new_input.select_dtypes(include=['object']).columns
    for col in col_categorical:
        if col in mappings:
            mapping = mappings[col]
            category_to_code = {v: k for k, v in mapping.items()}
            max_code = len(mapping)
            new_input[col] = new_input[col].map(lambda x: category_to_code.get(x, max_code)).astype(int)
    return new_input

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        customer = request.form['customer']
        age = request.form['age']
        gender = request.form['gender']
        merchant = request.form['merchant']
        category = request.form['category']
        amount = request.form['amount']
        
        # Create a DataFrame from the input
        new_input_data = pd.DataFrame([{
            'customer': customer,
            'age': age,
            'gender': gender,
            'merchant': merchant,
            'category': category,
            'amount': float(amount)  # Ensure amount is converted to float
        }])
        
        # Preprocess the input
        new_input_processed = preprocess_new_input(new_input_data, mappings)
        
        # Predict
        prediction = model.predict(new_input_processed)
        result = "Fraud" if prediction[0] == 1 else "Not Fraud"
        
        return render_template('index.html', result=result)
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

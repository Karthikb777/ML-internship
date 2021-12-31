import numpy as np
from flask import Flask, jsonify, render_template, request
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
model = pickle.load(open('../pickles/linearRegression.pkl', 'rb'))

'''
things that need to be included in the form
Age                            0 ip -
Income (USD)                   0 ip -
Income Stability               0 dropdown -
Type of Employment             0 dropdown - 
Location                       0 ip - 
Loan Amount Request (USD)      0 ip -
Current Loan Expenses (USD)    0 ip -
Expense Type 1                 0 dropdown - 
Expense Type 2                 0 dropdown - 
Dependents                     0 ip - 
Credit Score                   0 ip - 
No. of Defaults                0 dropdown - 
Has Active Credit Card         0 dropdown - 
Property Location              0 dropdown - 
Co-Applicant                   0 ip -
Property Price                 0 ip
'''

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/details')
def fill_details():
    return render_template('details.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = dict([x for x in request.form.items()])

    # loading the encoded data i.e the categorical data that is encoded
    enc = open('../pickles/encoder.pkl', 'rb')
    enc = pickle.load(enc)
    print(enc)
    print(features)

    # we encode the categorical data here
    features['Income Stability'] = enc[features['Income Stability']]
    features['Type of employment'] = enc[features['Type of employment']]
    features['Location'] = enc[features['Location']]
    features['Expense Type 1'] = 'Y' if features['Expense Type 1'] == 'Yes' else 'N'
    features['Expense Type 1'] = enc[features['Expense Type 1']]
    features['Expense Type 2'] = 'Y' if features['Expense Type 2'] == 'Yes' else 'N'
    features['Expense Type 2'] = enc[features['Expense Type 2']]
    features['Property Location'] = enc[features['Property Location']]
    features['Has Active Credit Card'] = enc[features['Has Active Credit Card']]
    print(features)

    predict_data = [
        features['Age'],
        features['Income (USD)'],
        features['Income Stability'],
        features['Type of employment'],
        features['Location'],
        features['Loan Amount Request (USD)'],
        features['Current Loan Expenses (USD)'],
        features['Expense Type 1'],
        features['Expense Type 2'],
        features['Dependents'],
        features['Credit Score'],
        features['No. of Defaults'],
        features['Has Active Credit Card'],
        features['Property Location'],
        features['Co-Applicant'],
        features['Property Price'],
    ]

    # final preparation of the data for prediction
    for i in range(len(predict_data)):
        if type(predict_data[i]) == str:
            predict_data[i] = float(int(predict_data[i]))
    print('predict data', predict_data)

    predictions = model.predict(np.array(predict_data).reshape((1, -1)))
    print(predictions)

    return render_template('results.html', prediction_text = predictions[0])


if __name__ == '__main__':
    app.run(debug=True)

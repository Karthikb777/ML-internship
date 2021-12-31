import numpy
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

    enc = open('../pickles/encoder.pkl', 'rb')
    enc = pickle.load(enc)
    print(enc)
    print(features)

    # encode the data here

    return render_template('details.html')

if __name__ == '__main__':
    app.run(debug=True)

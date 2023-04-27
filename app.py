import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model_rf.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    one = ['yes', 'present', 'good', 'normal', 'Yes', 'Present', 'Good', 'Normal', 'YES', 'PRESENT', 'GOOD', 'NORMAL']
    zero = ['no', 'notpresent', 'not present', 'poor', 'abnormal', 'No', 'Notpresent', 'NotPresent', 'Not Present', 'Poor', 'Abnormal', 'AbNormal', 'NO', 'NOTPRESENT', 'NOT PRESENT', 'POOR', 'ABNORMAL']
    int_features = []
    for i in request.form.values():
        if i in one:
            int_features.append(1.0)
        elif i in zero:
            int_features.append(0.0)
        else:
            int_features.append(float(i))
            
    final_features = [np.array(int_features)]

    final_features = scaler.transform(final_features)
    prediction = model.predict(final_features)
    
    output = prediction
    
    if output == [0]:
        output = "Kidney Disease Not Detected"
    elif output == [1]:
        output = "Kidney Disease Detected"
    
    return render_template('result.html', prediction_text='Diagnosis Result: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
    

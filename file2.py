import pickle
from flask import Flask, request
import numpy as np


with open(r'C:\Users\achowdhury\Desktop\Demo\rf.pkl', 'rb') as model_file:
    model= pickle.load(model_file)



#flask app
app= Flask(__name__)


@app.route('/predict')
def predict_iris():
    s_length = request.args.get('S_length')
    s_width = request.args.get('S_width')
    p_length = request.args.get('P_length')
    p_width = request.args.get('P_width')
    prediction = model.predict(np.array([[S_length, S_width, P_length, P_width]]))
    
    return str(prediction)

if __name__=='main':
    app.run()
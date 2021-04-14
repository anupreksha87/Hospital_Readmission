from flask import Flask, render_template, request
import pandas as pd
import csv
import os
import pickle
from training_model import trainModel
from werkzeug.utils import secure_filename
import sklearn
import flask_monitoringdashboard as dashboard
import warnings

def warns(*args, **kwargs):
    pass

warnings.warn = warns

app = Flask(__name__)
dashboard.bind(app)

model= pickle.load(open('pickle_files/rm2.pkl', 'rb'))

@app.route('/')
def Home():
    return render_template('index.html')

# @app.route('/dashboard')


@app.route('/bulk_predict',methods=['GET','POST'])
def bulk_predict():

    if request.method == "POST":
        try:
            f = request.files['csvfile']
            data = pd.read_csv(f,header=None)
            data.columns=["__Time_in_Hospital__","__Num_Lab_Procedure__","__Age__"," __Num_Medications__",
                      " __Num_procedures__","__Num_Diagnoses__"]

            output = model.predict(data)
            data['__output__'] = output
            data["__output__"].replace({0:"No Readmission",1:"Readmission"},inplace=True)
            return render_template('simple.html', tables=[data.to_html(classes='data')], titles=data.columns.values)

        except Exception as e:
            print('The Exception message is: ', e)
            return 'Total no of input features does not matched'


@app.route('/retraining',methods=['GET','POST'])
def retraining():
    try:
         if request.method == "POST":
            f = request.files['retrain_file']
            data = pd.read_csv(f)
            a=trainModel()
            a.trainingModel(data)

         return render_template('index.html', text=".. Retraining oF Model Successfull ..")

    except Exception as e:
         print('Invalid Input Files.')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            time_in_hospital = int(request.form['time_in_hospital'])
            num_lab_procedures = int(request.form['num_lab_procedures'])
            age = int(request.form['age'])
            num_medications = int(request.form['num_medications'])
            num_procedures = int(request.form[ 'num_procedures'])
            number_diagnoses = int(request.form[ 'number_diagnoses'])

            prediction = model.predict([[time_in_hospital, num_lab_procedures, age, num_medications,num_procedures,
               number_diagnoses]])
            output =round(prediction[0],1)
            if output==0:
                return render_template('index.html', prediction_text="The patient will not be readmitted after 30 days")
            elif output==1:
                return render_template('index.html', prediction_text="The patient will be readmitted after 30 days")
        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run()
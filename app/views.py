from flask import render_template, request, flash, redirect
from app import app
from .forms import MorgageInputForm
import dill
import numpy as np
import pandas as pd
import os.path

current_path = os.path.split(os.path.abspath(__file__))[0]
with open(os.path.join(current_path, 'sgd-model.dill'), 'rb') as f:
    sgd_model = dill.load(f)


def default_none(input_data):
    if input_data != None:
        return input_data
    else:
        return None


@app.route('/')
def root():
    return redirect('/index')


@app.route('/index', methods=['GET', 'POST'])
def index():
    global sgd_model
    form = MorgageInputForm()
    if (request.method == 'POST') and (form.validate()):
        test_case = pd.DataFrame.from_dict({
            'ORIG_AMT': [float(form.loan_amount.data)],
            'CSCORE_B': [float(form.buyer_credit.data)],
            'CSCORE_C': [default_none(form.cobuyer_credit.data)],
            'OCLTV': [float(form.loan_to_value.data)],
            'DTI': [float(form.debt_to_income.data)],
            'STATE': [form.loan_state.data],
            'PURPOSE': [form.loan_purpose.data],
            'PROP_TYP': [form.property_type.data],
            'OCC_STAT': [form.occupancy_type.data]
        }
        )
        prediction = sgd_model.predict(test_case)[0]
        print(prediction)
        if prediction == 0:
            result = 'OK!'
            status = 'alert alert-success'
        elif prediction == 1:
            result = 'Caution!'
            status = 'alert alert-danger'
        else:
            result = 'Check Input'
            status = 'alert alert-warning'
            flash(form.errors)
    else:
        result = 'N/A'
        status = 'alert alert-info'
        prediction = -1

    return render_template("index.html",
                           title='Mortgage Risk Assessment',
                           form=form,
                           result=result,
                           prediction=prediction,
                           status=status)


@app.route('/expo')
def expo():
    return render_template("expo.html")


@app.route('/learning')
def learning():
    return render_template('learning.html')

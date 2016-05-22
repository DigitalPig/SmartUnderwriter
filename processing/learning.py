#!/usr/bin/env python3

# This python file is to calculate SVC classification using Fannie Mae data

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn
import re
import datetime
import dill
import sklearn as sk
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cross_validation import train_test_split, KFold, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from definitions import summaryfile_type

# Part 1 Read the data file into memory
fannie1 = pd.read_csv('./processed/total/total_2012.csv', dtype=summaryfile_type,
                      nrows=9000000)
fannie1.drop(fannie1.columns[:1], axis=1, inplace=True)
fannie1.rename(columns=lambda x: re.sub('[.]', '_', x), inplace=True)
fannie1_filtered = fannie1.dropna(subset=('OLTV', 'OCLTV', 'DTI', 'CSCORE_B'))
fannie1_known = fannie1_filtered[fannie1_filtered['Zero_Bal_Code'] > 0]
state_mean = fannie1_known.groupby('STATE')[('ORIG_AMT', 'OCLTV', 'DTI')].mean()
state_std = fannie1_known.groupby('STATE')[('ORIG_AMT', 'OCLTV', 'DTI')].std()

# Helper function


class ExtractOrigYear(sk.base.BaseEstimator, sk.base.TransformerMixin):

    def __init__(self):
        self.int = np.vectorize(int)
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return self.int(x['ORIG_DTE'].apply(lambda x: x.split('/')[-1])).reshape(-1, 1)


class extractfeatures(sk.base.BaseEstimator, sk.base.TransformerMixin):

    def __init__(self, column='OCLTV'):
        '''
        We use the colname as the selection rule to judge the
        '''
        self.column = column

    def fit(self, x, y):
        return self

    def transform(self, x):
        return x.loc[:, self.column].values.reshape(-1, 1)


class ExtractLoanStatus(sk.base.BaseEstimator, sk.base.TransformerMixin):

    def __init__(self):
        '''
        Initialize the class with bisection of the loan status: Default or Healthy
        '''
        pass

    def fit(self, x):
        return self

    def transform(self, x):
        '''
        Transform the loan status to a tertiary status: Healthy (0), Default (1)
        '''
        status = x['Zero_Bal_Code'].apply(lambda x: 0 if x <= 1 else 1)
        return status


class ExtractCreditScore(sk.base.BaseEstimator, sk.base.TransformerMixin):

    def __init__(self, is_take_minimum=True):
        self.take_minimum = is_take_minimum
        pass

    def fit(self, x, y):
        return self

    def transform(self, x):
        result = np.where((x['CSCORE_B'] - x['CSCORE_C'] > 0), x['CSCORE_C'], x['CSCORE_B'])
        return result.reshape(-1, 1)


class ExtractCategory(sk.base.BaseEstimator, sk.base.TransformerMixin):

    def __init__(self, colname):
        self.colname = colname
        self.transformer = LabelEncoder()
        pass

    def fit(self, x, y):
        self.transformer.fit(x[self.colname])
        return self

    def transform(self, x):
        return self.transformer.transform(x[self.colname]).reshape(-1, 1)


class ExtractNormalized(sk.base.BaseEstimator, sk.base.TransformerMixin):

    def __init__(self, groupby, target, total_mean=state_mean, total_std=state_std):
        self.groupby = groupby
        self.target = target
        self.total_mean = total_mean
        self.total_std = total_std
        pass

    def fit(self, x, y):
        return self

    def transform(self, x):
        temp1 = x.groupby(self.groupby)[[self.groupby, self.target]
                                        ].apply(lambda x: (x[self.target])).values
        temp2 = x.groupby(self.groupby)[[self.groupby, self.target]].apply(
            lambda x: self.total_mean.ix[x[self.groupby].values, self.target]).values
        temp3 = x.groupby(self.groupby)[[self.groupby, self.target]].apply(
            lambda x: self.total_std.ix[x[self.groupby].values, self.target]).values
        return ((temp1 - temp2) / temp3).reshape(-1, 1)


# Logistic Regression
features = FeatureUnion([
    ('Loan_Amount', ExtractNormalized('STATE', 'ORIG_AMT')),
    #('Interest_Rate', ExtractNormalized('STATE','ORIG_RT')),
    ('credit score', ExtractCreditScore()),
    ('Loan_to_Value', ExtractNormalized('STATE', 'OCLTV')),
    ('Debt_to_income', ExtractNormalized('STATE', 'DTI')),
    ('Loan_purpose', ExtractCategory('PURPOSE')),
    ('Property_Type', ExtractCategory('PROP_TYP')),
    ('Occupancy_Status', ExtractCategory('OCC_STAT'))

])

sss = StratifiedShuffleSplit(ExtractLoanStatus().fit_transform(fannie1_known), 1, test_size=0.15)
for train_index, test_index in sss:
    fannie_train = fannie1_known.iloc[train_index, ]
    fannie_test = fannie1_known.iloc[test_index, ]
    status_train = ExtractLoanStatus().fit_transform(fannie1_known).iloc[train_index, ]
    status_test = ExtractLoanStatus().fit_transform(fannie1_known).iloc[test_index, ]


# fannie_train, fannie_test, status_train, status_test = train_test_split(fannie1_known,
#                                                                         ExtractLoanStatus().fit_transform(fannie1_known),
#                                                                         test_size=0.15)

print('Here is the Logistic regression results...')
model2 = Pipeline([
    ('features', features),
    ('Logistic', LogisticRegression(C=0.00077426, class_weight='balanced'))
])

model2.fit(fannie_train, status_train)
status_pred2 = model2.predict(fannie_test)
# print('Best C is: ', model2.named_steps['Logistic'].C_)
print('Coefficients: ', model2.named_steps['Logistic'].coef_)

print(classification_report(status_test, status_pred2))
print(pd.DataFrame(confusion_matrix(status_test, status_pred2), index=['Actual Healthy',
                                                                       'Actual Default'],
                   columns=['Pred. Healthy', 'Pred. Default']))
print('Area under the curve is', roc_auc_score(status_test, status_pred2))
prec, rec, thres1 = precision_recall_curve(status_test, status_pred2)
fpr, tpr, thres2 = roc_curve(status_test, model2.decision_function(fannie_test))
with open('log_prec_rec.dill', 'wb') as f:
    dill.dump((prec, rec, thres1), f)

with open('log_fpr_tpr.dill', 'wb') as f:
    dill.dump((fpr, tpr, thres2), f)

with open('log_model.dill', 'wb') as f:
    dill.dump(model2, f)

print('finishing dumping Logistic regression results to file!')

# # Support Vector Machine
# features = FeatureUnion([
#     ('Loan_Amount', ExtractNormalized('STATE', 'ORIG_AMT')),
#     #('Interest_Rate', ExtractNormalized('STATE','ORIG_RT')),
#     ('credit score', ExtractCreditScore()),
#     ('Loan_to_Value', ExtractNormalized('STATE', 'OCLTV')),
#     ('Debt_to_income', ExtractNormalized('STATE', 'DTI')),
#     ('Loan_purpose', ExtractCategory('PURPOSE')),
#     ('Property_Type', ExtractCategory('PROP_TYP')),
#     ('Occupancy_Status', ExtractCategory('OCC_STAT'))

# ])

# model3 = Pipeline([
#     ('features', features),
#     ('LinearSVC', LinearSVC(C=1, class_weight='balanced'))
# ])
# model3.fit(fannie_train, status_train)
# status_pred = model3.predict(fannie_test)
# print('SVM\'s ROC is', roc_auc_score(status_test,
#                                      model3.decision_function(fannie_test)))
# print('Now it is SVM time')
# with open('SVC.dill', 'wb') as f:
#     dill.dump(model3, f)
# print('Now let\'s grid search SVM parameters')

# # Grid search of SVC
# cv = ShuffleSplit(len(fannie_train), 1, test_size=0.2)
# model3_2 = Pipeline([
#     ('features', features),
#     ('LinearSVC', LinearSVC(C=1))
# ])
# search_model = GridSearchCV(model3_2,
#                             {'LinearSVC__C': np.logspace(-2, 2, 5)}, cv=3,
#                             scoring='roc_auc')
# search_model.fit(fannie_train, status_train)

# print('best parameters are: ', search_model.best_params_)
# print('Best score we can get: ', search_model.best_score_)

# Stochastic Gradient Descent Method

features = FeatureUnion([
    ('Loan_Amount', ExtractNormalized('STATE', 'ORIG_AMT')),
    #('Interest_Rate', ExtractNormalized('STATE','ORIG_RT')),
    ('credit score', ExtractCreditScore()),
    ('Loan_to_Value', ExtractNormalized('STATE', 'OCLTV')),
    ('Debt_to_income', ExtractNormalized('STATE', 'DTI')),
    ('Loan_purpose', ExtractCategory('PURPOSE')),
    ('Property_Type', ExtractCategory('PROP_TYP')),
    ('Occupancy_Status', ExtractCategory('OCC_STAT'))

])


model4 = Pipeline([
    ('features', features),
    ('scale', StandardScaler()),
    ('SGD', SGDClassifier(loss='hinge', class_weight='balanced'))
])
model4.fit(fannie_train, status_train)
status_pred4 = model4.predict(fannie_test)
print('The AUC of this SGD model is:',
      roc_auc_score(status_test, model4.decision_function(fannie_test)))
pred4 = model4.predict(fannie_test)
print('Classfication Report:')
print(classification_report(status_test, pred4))
print(pd.DataFrame(confusion_matrix(status_test, pred4), index=['Actual Healthy',
                                                                'Actual Default'],
                   columns=['Pred. Healthy', 'Pred. Default']))
print('Writing Stochastic Gradient Descent model to file...')
with open('sgd-model.dill', 'wb') as f:
    dill.dump(model4, f)

# Grid Search on SGD Classifier?
# Be careful to turn this option on as it may complains about your memory
# params = [{'SGD__loss':['hinge'], 'SGD__alpha': np.logspace(-4,1,6)},
#          {'SGD__loss': ['log'], 'SGD__alpha':np.logspace(-4,1,6)}]

# search_model4 = GridSearchCV(model4, param_grid=params, cv = 3, scoring='roc_auc')
# search_model4.fit(fannie_train, status_train)

# print('best parameters are:', search_model4.best_params_)
# print('best score obtained:', search_model4.best_score_)

# GradientBoostingClassifier!
features = FeatureUnion([
    ('Loan_Amount', ExtractNormalized('STATE', 'ORIG_AMT')),
    #('Interest_Rate', ExtractNormalized('STATE','ORIG_RT')),
    ('credit score', ExtractCreditScore()),
    ('Loan_to_Value', ExtractNormalized('STATE', 'OCLTV')),
    ('Debt_to_income', ExtractNormalized('STATE', 'DTI')),
    ('Loan_purpose', ExtractCategory('PURPOSE')),
    ('Property_Type', ExtractCategory('PROP_TYP')),
    ('Occupancy_Status', ExtractCategory('OCC_STAT'))

])

model5 = Pipeline([
    ('features', features),
    ('GDB', GradientBoostingClassifier(n_estimators=200, learning_rate=0.5))
])
model5.fit(fannie_train, status_train)
status_pred5 = model5.predict(fannie_test)
print('AUC score for gradient boosting method is: ',
      roc_auc_score(status_test, model5.decision_function(fannie_test)))

# Grid Search for the gradient boosting
print('Starting Grid search the gradient boosting method...')
params = {'GDB__n_estimators': [200, 300, 500, 1000],
          'GDB__learning_rate': np.logspace(-2, 0, 5)}

search_model5 = GridSearchCV(model5, param_grid=params, cv=3, scoring='roc_auc',
                             n_jobs=5)
search_model5.fit(fannie_train, status_train)
print('The best model\'s paramerters are: ', search_model5.best_params_)
print('The best model\'s score is: ', search_model5.best_score_)

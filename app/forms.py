from flask.ext.wtf import Form
from wtforms import FloatField, RadioField, SelectField
from wtforms.validators import DataRequired, NumberRange, optional

state_choices = [('CA', 'CA'),
                 ('WA', 'WA'),
                 ('VA', 'VA'),
                 ('IL', 'IL'),
                 ('SD', 'SD'),
                 ('TX', 'TX'),
                 ('OK', 'OK'),
                 ('TN', 'TN'),
                 ('AZ', 'AZ'),
                 ('IA', 'IA'),
                 ('MA', 'MA'),
                 ('PA', 'PA'),
                 ('CO', 'CO'),
                 ('GA', 'GA'),
                 ('KY', 'KY'),
                 ('ND', 'ND'),
                 ('MI', 'MI'),
                 ('NJ', 'NJ'),
                 ('UT', 'UT'),
                 ('MS', 'MS'),
                 ('FL', 'FL'),
                 ('NC', 'NC'),
                 ('SC', 'SC'),
                 ('ID', 'ID'),
                 ('LA', 'LA'),
                 ('MD', 'MD'),
                 ('NE', 'NE'),
                 ('OR', 'OR'),
                 ('IN', 'IN'),
                 ('OH', 'OH'),
                 ('NY', 'NY'),
                 ('MN', 'MN'),
                 ('ME', 'ME'),
                 ('AL', 'AL'),
                 ('WI', 'WI'),
                 ('MO', 'MO'),
                 ('CT', 'CT'),
                 ('KS', 'KS'),
                 ('NV', 'NV'),
                 ('NH', 'NH'),
                 ('AR', 'AR'),
                 ('WY', 'WY'),
                 ('RI', 'RI'),
                 ('MT', 'MT'),
                 ('WV', 'WV'),
                 ('DC', 'DC'),
                 ('HI', 'HI'),
                 ('NM', 'NM'),
                 ('VT', 'VT'),
                 ('PR', 'PR'),
                 ('AK', 'AK'),
                 ('VI', 'VI'),
                 ('DE', 'DE'),
                 ('GU', 'GU')
                 ]

loan_purpose_choices = [('P', 'Purchase'),
                        ('R', 'No Cash-out Refinance'),
                        ('C', 'Cash-out Refinance'),
                        ('U', 'Refinance Not Specified')
                        ]

property_type_choices = [('SF', 'Single Family'),
                         ('PU', 'Planned Urban Development'),
                         ('CO', 'Condo'),
                         ('MH', 'Manufactured Housing'),
                         ('CP', 'Co-Op')
                         ]
occupancies_type_choices = [('P', 'Primary Home'),
                            ('S', 'Secondary Home'),
                            ('I', 'Investment Home')
                            ]


class MorgageInputForm(Form):
    loan_amount = FloatField('Originated Amount', validators=[DataRequired()])
    buyer_credit = FloatField('Buyer\'s credit score', validators=[DataRequired(),
                                                                   NumberRange(500.0, 850.0)])
    cobuyer_credit = FloatField('Cobuyer\'s credit score', validators=[optional(),
                                                                       NumberRange(500.0,
                                                                                   850.0,
                                                                                   "Check credit score")])
    loan_to_value = FloatField('Originated Loan to Value ratio',
                               validators=[DataRequired(), NumberRange(0, 100)])
    debt_to_income = FloatField('Debt to Income Ratio', validators=[DataRequired(),
                                                                    NumberRange(0, 100)])
    loan_state = SelectField('State', choices=state_choices)
    loan_purpose = SelectField('Loan Purpose',
                               choices=loan_purpose_choices)
    property_type = SelectField('Property Type',
                                choices=property_type_choices)
    occupancy_type = SelectField('Occupancy Type',
                                 choices=occupancies_type_choices)

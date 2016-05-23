from flask import Flask
import sys
import logging
#from flask_wtf.csrf import CsrfProtect

app = Flask(__name__)
app.config.from_object('config')
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

#app.SECRET_KEY = 'test_fsf'
# CsrfProtect(app)
from app import views

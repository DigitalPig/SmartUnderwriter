from flask import Flask
#from flask_wtf.csrf import CsrfProtect

app = Flask(__name__)
app.config.from_object('config')
#app.SECRET_KEY = 'test_fsf'
# CsrfProtect(app)
from app import views

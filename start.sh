#!/bin/bash

source activate smart_underwriter
gunicorn --bind 0.0.0.0:$PORT run
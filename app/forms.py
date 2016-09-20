from flask import flash
from flask_wtf import Form
from wtforms import StringField, FileField
from wtforms.validators import DataRequired, ValidationError
import validators


class SearchForm(Form):
    file = FileField('file', validators=[DataRequired()])

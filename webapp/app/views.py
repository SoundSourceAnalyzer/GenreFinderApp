from flask import Flask, request, redirect, url_for, flash, render_template
import os
from app import app
from app.forms import SearchForm, RetryForm
import json
from werkzeug.utils import secure_filename

from neuralnet.model import NeuralNetModel
from neuralnet.parser import FeatureExtractor

ALLOWED_EXTENSIONS = set(['mp3'])


# def classify_file(path):
#     ft = FeatureExtractor(path)
#     features = ft.get_features()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def index():
    form = SearchForm()
    if form.validate_on_submit():
        render_template('index.html', form=form)
    else:
        flash_errors(form)
    return render_template('index.html', form=form)


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' in request.files:
        file = request.files['file']
        print(file)
        filename = secure_filename(file.filename)
        filepath = os.path.join('/app/webapp/app/uploaded', filename)
        file.save(filepath)
        features = FeatureExtractor(filepath).fv
        model = NeuralNetModel()
        genre = model.predict_gztan(features)
        del model
        return render_template('result.html', features=json.dumps(features), genre=genre)
    else:
        features = json.loads(request.form['features'])
        print(features)
        print(type(features))
        model = NeuralNetModel()
        genre = model.predict_gztan(features)
        del model
        return render_template('result.html', features=features, genre=genre)


    # check if the post request has the file part



def flash_errors(form):
    for field, errors in form.errors.items():
        for error in errors:
            flash(u"Error - %s" % error)

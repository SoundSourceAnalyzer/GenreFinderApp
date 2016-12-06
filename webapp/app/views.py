from flask import Flask, request, redirect, url_for, flash, render_template
import tensorflow as tf
import os
from app import app
from app.forms import SearchForm
from werkzeug.utils import secure_filename

# from audio_parser.parser import FeatureExtractor


ALLOWED_EXTENSIONS = set(['mp3'])


# def classify_file(path):
#     ft = FeatureExtractor(path)
#     features = ft.get_features()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def index():
    form = SearchForm()
    if form.validate_on_submit():
        render_template('index.html', form=form)
    else:
        flash_errors(form)
    return render_template('index.html', form=form)


@app.route('/', methods=['POST'])
@app.route('/index', methods=['POST'])
def upload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return render_template('loading.html')
        file = request.files['file']
        print(file)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return redirect(url_for('uploaded_file',filename=filename))
            return render_template('loading.html')
        return render_template('loading.html')

def flash_errors(form):
    for field, errors in form.errors.items():
        for error in errors:
            flash(u"Error - %s" % error)

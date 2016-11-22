from flask import render_template, flash
import tensorflow as tf
from app import app
from app.forms import SearchForm
from audio_parser.parser import FeatureExtractor


def classify_file(path):
    ft = FeatureExtractor(path)
    features = ft.get_features()


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = SearchForm()
    if form.validate_on_submit():
        saver = tf.train.Saver()
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            save_path = "../neuralnet/model.ckpt"
            saver.restore(sess, save_path)
            print("Model restored.")
        return render_template('loading.html')
    else:
        flash_errors(form)
    return render_template('index.html', form=form)


def flash_errors(form):
    for field, errors in form.errors.items():
        for error in errors:
            flash(u"Error - %s" % error)

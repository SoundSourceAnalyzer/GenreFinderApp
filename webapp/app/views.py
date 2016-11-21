from flask import render_template, flash
from app import app
from app.forms import SearchForm


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = SearchForm()
    if form.validate_on_submit():
        return render_template('loading.html')
    else:
        flash_errors(form)
    return render_template('index.html', form=form)


def flash_errors(form):
    for field, errors in form.errors.items():
        for error in errors:
            flash(u"Error - %s" % error)

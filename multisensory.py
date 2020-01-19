import os
import tensorflow as tf

from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from os.path import join, dirname, realpath
from forms import UploadForm
import sep.driver

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 48 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = join(dirname(realpath(__file__)), 'uploads/')
app.secret_key = 'super secret key'


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    flash(request.files) # for debugging
    if form.validate_on_submit():
        flash("Form Validated!") # debugging
        file = request.files['input_file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('File successfully uploaded.') # debugging
        return redirect(url_for('uploaded_file', filename=filename)) 

    return render_template('index.html', form=form)


def doSomething():
    pass

if __name__ == '__main__':
    app.run()

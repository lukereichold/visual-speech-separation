import os
import sep.driver as separator
import uuid

from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from os.path import join, dirname, realpath
from forms import UploadForm


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 48 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = join(dirname(realpath(__file__)), 'uploads/')
app.config['RESULTS_FOLDER'] = join(dirname(realpath(__file__)), 'results/')
app.secret_key = 'super secret key'


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename, as_attachment=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    if form.validate_on_submit():
        file = request.files['input_file']
        unique_filename = str(uuid.uuid4())
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename + '.mp4')
        file.save(file_path)

        output_files = separate(file_path)
        return render_template('index.html', form=form, results=output_files)

    return render_template('index.html', form=form)


def separate(input_file):
    args = [input_file, '--duration_mult', '4', '--out', join(dirname(realpath(__file__)), 'results/')]
    return separator.main(args)


@app.errorhandler(RuntimeError)
def handle_error(e):
    flash("Sorry, an error has occurred because the format of the uploaded video is unsupported.")
    return render_template('index.html', form=UploadForm())


if __name__ == '__main__':
    app.run()

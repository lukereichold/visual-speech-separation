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
app.secret_key = 'super secret key'


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    flash(request.files) # for debugging
    if form.validate_on_submit():
        file = request.files['input_file']
        unique_filename = str(uuid.uuid4())
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename + '.mp4')
        file.save(file_path)

        flash('File successfully uploaded.') # debugging
        # separate(file_path)

        return redirect(url_for('uploaded_file', filename=filename))

    return render_template('index.html', form=form)


def separate(input_file):
    args = {'duration_mult': 4, 'out': '../results', 'vid_file': input_file}
    separator.main(args)

    # TODO: expect main() to return tuple containing the final BG and FG video file paths that were saved.

if __name__ == '__main__':
    app.run()

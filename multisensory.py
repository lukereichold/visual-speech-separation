import os
# import forms
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from os.path import join, dirname, realpath
from forms import UploadForm

ALLOWED_EXTENSIONS = {'mp4'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 48 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = join(dirname(realpath(__file__)), 'uploads/')
app.secret_key = 'super secret key'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    # if request.method == 'POST' and form.validate_on_submit():

    if form.validate_on_submit(): # only called for POST
        flash(request.files)
        # input_file = request.files['input_file']
        # Do stuff

    return render_template('index.html', form=form)

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         flash(request.files)
#         if 'file' not in request.files:
#             flash('No file was selected')
#             return redirect(request.url)
#         file = request.files['file']
#         # flash(file)
#         if file.filename == '':
#             flash('No file was selected')
#             return redirect(request.url)
#         if not file or not allowed_file(file.filename):
#             flash('Selected file is unsupported. Try again with a valid .mp4 file.')
#             return redirect(request.url)

#         # TODO! (Happy path)...
#         # filename = secure_filename(file.filename)
#         # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#         # flash('File successfully uploaded.')
#         # return redirect(url_for('uploaded_file', filename=filename))

#     return render_template('index.html')


def doSomething():
    pass

if __name__ == '__main__':
    app.run()

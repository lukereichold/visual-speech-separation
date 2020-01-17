import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from os.path import join, dirname, realpath

ALLOWED_EXTENSIONS = {'mp4'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 48 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = join(dirname(realpath(__file__)), 'uploads/')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/separate', methods=['POST'])
def upload():
    return {"error": "WTF."}
    if 'file' not in request.files:
        return {"error": "No file was selected."}
    file = request.files['file']
    if file.filename == '':
        return {"error": "No file was selected."}

    # TODO!
    # if file and allowed_file(file.filename):
        # filename = secure_filename(file.filename)
    #     file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    #     # flash('File successfully uploaded.')
    #     return redirect(url_for('uploaded_file', filename=filename))

    return "success"


def doSomething():
    pass

if __name__ == '__main__':
    app.run()

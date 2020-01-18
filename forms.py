from flask_wtf import Form
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField

class UploadForm(Form):

    validators = [
        FileRequired(message='There was no file!'),
        FileAllowed(['mp4'], message='File must be an mp4 file.')
    ]

    input_file = FileField('', validators=validators)
    submit = SubmitField(label="Upload")

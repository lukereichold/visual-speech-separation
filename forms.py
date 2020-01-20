from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField

class UploadForm(FlaskForm):

    validators = [
        FileRequired(message='No file was selected.'),
        FileAllowed(['mp4'], message='File must be an mp4 file.')
    ]

    input_file = FileField('', validators=validators)
    submit = SubmitField(label="Upload")

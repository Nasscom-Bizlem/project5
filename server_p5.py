from flask import Flask
import os
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import mammoth
import json

from project_5 import p5_process_file

UPLOAD_FOLDER = './uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

def allowed_file(filename, extensions):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

@app.route('/')
def hello():
    return 'Hello World Project 5'


@app.route('/project5', methods=['POST'])
def project5():
    if 'file' not in request.files:
        return jsonify({ 'error': 'No file provided' }), 400

    file = request.files['file']

    only_extract_html_line = request.form.get('only_extract_html_line', 'false') in set([ 'true', 'True', 1 ])
    print(only_extract_html_line)

    if file and allowed_file(file.filename, ['html', 'json', 'xls', 'xlsx']):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        result = p5_process_file(path, only_extract_html_line=only_extract_html_line)

        return jsonify(result)



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True, port=5025)


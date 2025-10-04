from flask import Flask, request, render_template
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'pdfFile' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['pdfFile']
    
    if file.filename == '':
        return "No selected file", 400

    if file and file.filename.endswith('.pdf'):
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        return "Upload successful!"
    
    return "Invalid file type", 400

if __name__ == '__main__':
    app.run(debug=True)

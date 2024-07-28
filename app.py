from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from preprocess import preprocess_image, count_edges
from model import load_trained_model, preprocess_image_for_prediction, predict_sheet_count

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Load the model once when the app starts
model = load_trained_model()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process the image and get the sheet count
        edge_count = count_edges(file_path)
        sheet_count = predict_sheet_count(file_path, model)

        # Convert numpy.int64 to Python int for JSON serialization
        edge_count = int(edge_count)
        sheet_count = int(sheet_count)

        return jsonify({'sheet_count': sheet_count, 'edge_count': edge_count})


if __name__ == '__main__':
    app.run(debug=True)

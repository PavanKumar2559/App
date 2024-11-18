from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_PATH'] = 'machine/cotton_disease_model.h5.keras'  # Update the model path as needed

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def preprocess_image(image_path):
    image = load_img(image_path, target_size=(150, 150))  # Ensure this size matches the model input size
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize the image
    return image

def predict_disease(image_path):
    model = load_model(app.config['MODEL_PATH'])
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    class_idx = np.argmax(prediction, axis=1)[0]  # Use argmax for multi-class classification
    class_names = ['diseased cotton leaf','diseased cotton plant','fresh cotton leaf','fresh cotton plant']  # Ensure this matches model classes
    return class_names[class_idx]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            prediction = predict_disease(filepath)
            # Generate the URL for the uploaded file
            image_url = url_for('uploaded_file', filename=filename)
            if prediction == 'Downy Mildew':
                return redirect(url_for('downy', image_url=image_url))
            else:
                return f'<h1>{prediction}</h1><img src="{image_url}" alt="Uploaded Image">'
    return render_template('upload.html', image_url=None)

@app.route('/downy')
def downy():
    image_url = request.args.get('image_url')
    description = "Detailed information about Downy Mildew."
    return render_template('disease1.html', image_url=image_url, name="Downy Mildew", description=description)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)

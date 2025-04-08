import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf

# Init
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = tf.keras.models.load_model('blackhole_classifier_model.h5')
class_names = ['Stellarblackhole', 'supermassiveblackhole','Othertypes']  # Change this to match your folder names

# Preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((150, 150))  # Match model input
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', label="No file uploaded")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', label="No selected file")

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = preprocess_image(filepath)
            prediction = model.predict(img)
            predicted_class = class_names[np.argmax(prediction)]

            return render_template('index.html', filename=filename, label=predicted_class)

    return render_template('index.html', filename=None, label=None)

# Run
if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model = tf.keras.models.load_model('model/riceleaf_model.h5')

# Define class names
CLASSES = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']

# Image preprocessing
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))  # Resize to match model input
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image')

        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess and predict
            input_img = preprocess_image(filepath)
            prediction = model.predict(input_img)[0]
            pred_class = CLASSES[np.argmax(prediction)]
            confidence = float(np.max(prediction))

            return render_template('index.html',
                                   prediction=pred_class,
                                   confidence=round(confidence * 100, 2),
                                   img_path=filepath)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

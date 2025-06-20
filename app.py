from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model and nutrition data
model = tf.keras.models.load_model('Image_classify.keras')
nutrition_data = pd.read_excel('Nutrition.xlsx')

data_cat = ['Ayam Goreng', 'Burger', 'French Fries', 'Gado-Gado',
            'Ikan Goreng', 'Mie Goreng', 'Nasi Goreng', 'Nasi Padang',
            'Pizza', 'Rawon', 'Rendang', 'Sate', 'Soto']

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(request.url)
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Image preprocessing
        img_width, img_height = 180, 180
        image = load_img(filepath, target_size=(img_height, img_width))
        img_arr = img_to_array(image)
        img_bat = tf.expand_dims(img_arr, 0) / 255.0
        
        # Prediction
        predict = model.predict(img_bat)
        score = tf.nn.softmax(predict[0])
        predicted_class = data_cat[np.argmax(score)]
        confidence = np.max(score) * 100
        
        # Nutrition info
        if predicted_class in nutrition_data['Name'].values:
            nutrition_info = nutrition_data[nutrition_data['Name'] == predicted_class].iloc[0]
        else:
            nutrition_info = {}
        
        return render_template('index.html',
                               image_url=filepath,
                               predicted_class=predicted_class,
                               confidence=confidence,
                               nutrition_info=nutrition_info.to_dict())

if __name__ == '__main__':
    app.run(debug=True)

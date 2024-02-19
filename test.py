import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import base64
import io

app = Flask(__name__)
CORS(app)
model = tf.keras.models.load_model('num_reader_project.model')

@app.route('/')
def index():
    return render_template('index.html')  # Ensure this HTML file exists in a 'templates' folder


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image']
    
    image_data = image_data.split(",")[1]  # Remove the base64 header
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    # image.save("debug_input_image.png")
    # image.show("debug_input_image.png")
    image = image.convert('L')
    image = image.resize((28, 28))  # Resize to match MNIST image dimensions
    image = ImageOps.invert(image)
    image = np.array(image)
    image = image / 255.0  # Normalize pixel values
    image = image.reshape(1, 28, 28, 1)
    



    prediction = model.predict(image)
    predicted_number = np.argmax(prediction)
    print(predicted_number)
    return jsonify({'prediction': int(predicted_number)})

if __name__ == '__main__':
    app.run(debug=True)

# plt.imshow(image[0], plt.cm.binary)
# plt.show()
from flask import Flask, request, url_for, render_template, jsonify, redirect
import os
from os.path import join, dirname, realpath
import sys
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'static/uploads')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
import json

# Models below
import keras.models
import skimage
import matplotlib.pyplot as plt
import numpy as np
from CNNVoting import predict_best
from PIL import Image

classifier = keras.models.load_model(os.path.join(dirname(realpath(__file__)), "models/classifier.h5"))
classifier._make_predict_function()
from invert import Inverter
inverter = Inverter()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_from_path(path):
    return skimage.img_as_float(plt.imread(path))

@app.route('/process', methods=['POST'])
def process_photo():
    if request.method != 'POST':
        print("Not post")
        return jsonify({"success": False})
    if 'image' not in request.files:
        print("No image")
        return jsonify({"success": False})
    file = request.files['image']
    if file.filename == '':
        print('No selected file')
        return jsonify({"success": False})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Workaround saving and reloading
        save_route = os.path.join(app.config['UPLOAD_FOLDER'], filename) 
        file.save(save_route)
        im = image_from_path(save_route)
        identified_filter = predict_best(im, classifier)
        inverted_image = inverter.invert(im, identified_filter)
        print(inverted_image)
        im = Image.fromarray(inverted_image.astype(np.uint8))
        im.save(save_route)
        return jsonify({"img_url": filename, "filter": identified_filter})

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


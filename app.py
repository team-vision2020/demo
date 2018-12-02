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
        file_parts = filename.rsplit('.', 1)
        invert_name = '{}_invert.{}'.format(file_parts[0], file_parts[1])
        invert_route = os.path.join(app.config['UPLOAD_FOLDER'], invert_name)
        file.save(save_route)
        im = image_from_path(save_route)[..., :3] # Presumably int input, trim alpha
        int_im = (255 * im).astype(np.uint8)
        identified_filter = predict_best(int_im, classifier)
        inverted_image = inverter.invert(im, identified_filter)
        im = Image.fromarray((inverted_image * 255).astype(np.uint8))
        im.save('{}'.format(invert_route))
        return jsonify({"img_url": filename, "invert_url": invert_name, "filter": identified_filter})

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


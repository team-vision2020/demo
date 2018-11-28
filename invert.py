"""
Filter inversion models.

Author: Cem Gokmen
"""


import numpy as np
import keras
import os
import functools

class Filter(object):
    IDENTITY = 'identity'
    GOTHAM = 'gotham'
    CLARENDON = 'clarendon'
    GINGHAM = 'gingham'
    JUNO = 'juno'
    LARK = 'lark'
    REYES = 'reyes'

    FILTER_TYPES = [IDENTITY, GOTHAM, CLARENDON, GINGHAM, JUNO, LARK, REYES]

    def __init__(self, filter_type=IDENTITY):
        self.filter_type = filter_type

    def to_categorical(self):
        return keras.utils.to_categorical(self.FILTER_TYPES.index(self.filter_type), len(self.FILTER_TYPES))

    @classmethod
    def from_categorical(cls, categorical, threshold=0.7):
        max_conf_idx = np.argmax(categorical)
        if categorical[max_conf_idx] >= 0.7:
            return Filter(cls.FILTER_TYPES[max_conf_idx])
        else:
            return None


DEFAULT_MODEL_FOLDER = 'models' # os.path.join(utility.ROOT_DIR, 'models')

class Inverter(object):
    def __init__(self, model_folder=DEFAULT_MODEL_FOLDER):
        self.model_folder = model_folder
        self.models = {}
        self._load_models()

    def _model_path(self, filter_type):
        return os.path.join(self.model_folder, "{}_model.h5".format(filter_type))

    def _dataset_path(self, filter_type):
        return os.path.join(self.model_folder, "{}_dataset.npz".format(filter_type))

    def _load_models(self):
        print("Loading inversion models... Might take a while")

        # Don't generate model for identity op.
        all_filter_types = Filter.FILTER_TYPES.copy()
        del all_filter_types[all_filter_types.index(Filter.IDENTITY)]

        def load_model(filter_type):
            model_path = self._model_path(filter_type)
            if os.path.exists(model_path):
                print("Found {}".format(model_path))
                return keras.models.load_model(model_path), filter_type

            return None, filter_type

        # Try to load models.
        trained_models = [load_model(filter_type) for filter_type in all_filter_types]
        for model, filter_type in trained_models:
            self.models[filter_type] = model

        print("Loaded all models.")

    def invert(self, image, filter):
        if filter == None or filter == Filter.IDENTITY:
            return image


        model = self.models[filter]

        # Extend the image (so that we can pass 3x3 batches at the edges too)
        extended_image = np.zeros((image.shape[0] + 2, image.shape[1] + 2, 3))
        extended_image[1:-1, 1:-1] = image
        
        # Extend the rows
        extended_image[0] = extended_image[1]
        extended_image[-1] = extended_image[-2]
        
        # Extend the columns
        extended_image[:, 0] = extended_image[:, 1]
        extended_image[:, -1] = extended_image[:, -2]
        
        # Finally the corners
        extended_image[0, 0] = extended_image[0, 1]
        extended_image[0, -1] = extended_image[0, -2]
        extended_image[-1, 0] = extended_image[-1, 1]
        extended_image[-1, -1] = extended_image[-1, -2]

        # We split the filtered image into 3x3 windows and store all of those.
        data_points = np.zeros((extended_image.shape[0] - 2, extended_image.shape[1] - 2, 3, 3, 3))

        for cent_y in range(0, data_points.shape[0]):
            for cent_x in range(0, data_points.shape[1]):
                # Get the window
                orig_y = cent_y + 1
                orig_x = cent_x + 1

                sample = extended_image[orig_y - 1:orig_y + 2, orig_x - 1:orig_x + 2, :]
                data_points[cent_y, cent_x] = sample

        # We run the predictions
        outData = np.clip(model.predict(data_points.reshape(-1, 3, 3, 3)), 0, 1)

        # And we convert the predictions to the correct shape
        unfilteredIm = outData.reshape(data_points.shape[0], data_points.shape[1], 3)

        return unfilteredIm
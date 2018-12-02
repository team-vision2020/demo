from scipy.stats import mode
import numpy as np
from invert import Filter

# Converts image into as many 32x32 blocks as it can
# Assumes single input image, shape mxnx3 
# Just drops nonround thingsmi
def split_img(img):
    img_small = []
    x_steps = img.shape[0] // 32
    y_steps = img.shape[0] // 32 
    for i in range(x_steps):
        for j in range(y_steps):
            x_start = i * 32
            y_start = j * 32
            img_small.append(img[x_start:x_start + 32, y_start:y_start + 32])
    return np.asarray(img_small)[..., :3] # Trim alpha channel if it exists

# Provides most popular prediction given an array of probabilities
def prediction_voting(predict):
    mapping = [Filter.IDENTITY, Filter.CLARENDON, Filter.GINGHAM, Filter.JUNO, Filter.LARK, Filter.GOTHAM, Filter.REYES ]
    votes = np.argmax(predict, axis=1)
    vote = mode(votes)[0]
    return mapping[vote[0]] # Assume no ties

# Given an img, split and use model to get best predicts
def predict_best(img, classifier):
    splits = split_img(img)
    print(splits.shape)
    pred_vectors = classifier.predict(splits)
    # pred_vectors = np.asarray([classifier.predict(split) for split in splits])

    print("Pred_vectors shape: {}".format(pred_vectors.shape))
    return prediction_voting(pred_vectors)

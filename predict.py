from utility_functions import get_args, process_image_for_prediction, process_labels

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

args = get_args()

model = tf.keras.models.load_model(args.model_path, custom_objects={'KerasLayer':hub.KerasLayer})

def predict(image, model, k):
    pred_image = process_image_for_prediction(image)
    
    pred = model.predict(pred_image)[0]
    
    # find the indices of the top k probabilities
    idx = np.argpartition(pred, -k)[-k:]
    indices = idx[np.argsort((-pred)[idx])]
    
    top_k_labels = [str(k + 1) for k in indices]
    return pred[indices], top_k_labels

pred, labels = predict(args.image_path, model, args.top_k)

labels = process_labels(labels, args.category_names)

print(list(zip(labels, pred)))
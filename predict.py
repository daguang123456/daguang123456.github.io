import pickle
import cv2
import numpy as np
import joblib

model = joblib.load("cat_dog_cnn.joblib")

img_width, img_height = 150, 150

def predict(image):
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_width, img_height))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    label = int(round(pred[0][0]))
    return label
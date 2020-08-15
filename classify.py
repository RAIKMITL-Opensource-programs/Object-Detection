#python classify.py --model custome_model.model --labelbin lb.pickle --image examples/pen.jpg
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

class detection:
    def __init__(self):
        pass
    def detect(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-m", "--model", required=True,
            help="path to trained model model")
        ap.add_argument("-l", "--labelbin", required=True,
            help="path to label binarizer")
        ap.add_argument("-i", "--image", required=False,
            help="path to input image")
        args = vars(ap.parse_args())

        image = cv2.imread(args["image"])
        output = image.copy()

        image = cv2.resize(image, (96, 96))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        print("[INFO] loading network...")
        model = load_model(args["model"])
        lb = pickle.loads(open(args["labelbin"], "rb").read())

        print("[INFO] classifying image...")
        proba = model.predict(image)[0]
        idx = np.argmax(proba)
        label = lb.classes_[idx]

        filename = args["image"][args["image"].rfind(os.path.sep) + 1:]
        correct = " " if filename.rfind(label) != -1 else " "
        label = "{}: {:.2f}% {}".format(label, proba[idx] * 100, correct)
        output = imutils.resize(output, width=400)
        cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)
                
        print("[INFO] {}".format(label))
        cv2.imshow("Output", output)
        cv2.waitKey(0)
if __name__ == "__main__":
    test = detection()
    test.detect()
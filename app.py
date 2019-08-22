import numpy as np
import os
from scipy import  misc
from keras.models import model_from_json
import pickle

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)


classifier_f = open("int_to_word_out.pickle", "rb")
int_to_word_out = pickle.load(classifier_f)
classifier_f.close()


# load json and create model
json_file = open('model_face.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_face.h5")
print("Model is now loaded in the disk")

def model_predict(img_path, loaded_model):
    image=np.array(misc.imread(img_path))
    image = misc.imresize(image, (64, 64))
    image=np.array([image])
    image = image.astype('float32')
    image = image / 255.0

    prediction=loaded_model.predict(image)

    print(prediction)
    print(np.max(prediction))
    print(int_to_word_out[np.argmax(prediction)])
    result_list = [prediction,np.max(prediction),int_to_word_out[np.argmax(prediction)]]

    return result_list

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result_list = model_predict(file_path, loaded_model)
        all_accuracy = result_list[0]
        accuracy = result_list[1]
        accuracy = float("{0:.2f}".format(accuracy))
        accuracyHuman = accuracy * 100
        obj = result_list[2]

        if accuracy < 0.70:
            result = "Probably " + str(obj) +", but not confident, with " + str(accuracyHuman) +"% confidence"

        else:
            result = str(obj) + " with " + str(accuracyHuman) +"% confidence"


        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        return result # Object Detected: x, Accuracy: y,

    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()

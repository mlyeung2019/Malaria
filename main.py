#https://github.com/keras-team/keras/issues/13353 (replace keras by tensorflow.keras)
from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
#from tensorflow.keras.applications import vgg16
from keras.preprocessing import image
#from tensorflow.keras.preprocessing.image import img_to_array
#import pickle
#import keras.backend.tensorflow_backend as tb
#tb._SYMBOLIC_SCOPE.value = True


app=Flask(__name__)

loaded_model = load_model('malariamodel.h5')
loaded_model._make_predict_function()
#loaded_model = pickle.load(open("malariamodel.pkl","rb"))

@app.route('/')
def home():
    return render_template("home.html")

def ClassPredictor(file):
    #preprocess file
    # use
    print(file.shape) 
    result = loaded_model.predict(file)
    #print("after loaded_model")
    return result[0][0]

def process_image(img):
   
    img = np.expand_dims(img,axis=0)
    print(img.shape)
    #img = vgg16.preprocess_input(img)
    #print(img.dtype)
    img = img.astype('float32')
    #print(img.dtype)
    img /= 255.0
    #print("Hello01")
    result = ClassPredictor(img)
    #print("Hello02")
    return result

@app.route('/result',methods = ['POST'])
def result():
    prediction=''
    if request.method == 'POST':
        file = request.files['file']
        img = image.load_img(file,target_size=(64,64))
        #print(type(img))
        result = process_image(img)       
        print("result from model", result)
        if float(result)<0.5:
            prediction = 'Infected'
        else:
            prediction = 'Not infected'
        print(prediction)
        return render_template("result.html",prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
import pickle
from flask import Flask, request, json, jsonify
import numpy as np
import tensorflow as tf
 
app = Flask(__name__)
 
#---the filename of the saved model---
filename = 'my_model_011323.h5'
 
#---load the saved model---
loaded_model = tf.keras.models.load_model(filename)
 
@app.route('/fruits/v1/predict', methods=['POST'])
def predict():
    #---get the features to predict---
    features = request.json
 
    #---store the value from the key-value pair--- 
    # list na ba to? 
    features_list = features["image"] # extract the "value" of the "key" = image from the client

    #---Assuming that features_list is a list--- 
    input_img = np.array(features_list) # convert the list to numpy array
    input_img = np.expand_dims(input_img, axis=0) #add another bracket -> [input_img]
    input_img = np.vstack([input_img])
 
    #---get the prediction class---
    prediction = loaded_model.predict(input_img)

    #---json does not handle numpy array---
    output_result = prediction[0].tolist() # convert numpy array to list
 
    #---formulate the response to return to client---
    response = {} # create an empty dictionary
    response['prediction'] = output_result #Create a key-value pair for the dict: "key" = prediction (a string) and "value" = prediction[0]
    
    # returns a key-value pair (a dictionary)
    # response = {"predicton": output_result}
    # output_result is a list of prediction probabilities
    # Serialize it again. Turns dict or list into string
    # Alternative: json.dumps(response)
    return  jsonify(response)
 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
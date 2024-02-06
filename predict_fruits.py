import json
import requests
import cv2
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
 
def predict_fruits(image):
    url = 'http://127.0.0.1:5000/fruits/v1/predict'
    data = {"image":image} #a key-value pair where value is a list
    data_json = json.dumps(data) # Serialize: Turns dict or list into string
    headers = {'Content-type':'application/json'} # a way of saying that the data that will be sent as a json format
    response = requests.post(url, data=data_json, headers=headers) # POST - send data to the server and extracts it
    result = json.loads(response.text) #Deserialize response and get the text
    return result
 
if __name__ == "__main__":
    #---take picture of the fruit---
    camera = cv2.VideoCapture(1)

    while True:
        # Capture a frame from the camera
        ret, frame = camera.read()
        # Show the frame
        cv2.imshow("Camera", frame)
        # Wait for the user to press 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    camera.release()
    cv2.destroyAllWindows()

    # Save the captured frame as an image
    cv2.imwrite("image.jpg", frame)
    
    # ret, image = camera.read()
    # if ret:
    #     cv2.imwrite("image.jpg", image)
    #     print("Image captured and saved successfully!")
    # else:
    #     print("Failed to capture image")
    # camera.release()

    #---load, convert to np array, normalize and convert to list---
    img_tf = load_img("image.jpg", target_size=(150,150)) # PIL Object
    img_tf = img_to_array(img_tf) # Convert PIL object to numpy array
    img_tf = img_tf/255 # Normallize
    image = img_tf.tolist() # Convert numpy array to list

    #---feed the image to the function---
    predictions = predict_fruits(image)

    #---Note: predictions["prediction"] is a list---
    preds = np.array(predictions["prediction"])
    
    if preds[0]==np.max(preds):
        print("Fresh Apple")
    elif preds[1]==np.max(preds):
        print("Fresh Banana")
    elif preds[2]==np.max(preds):
        print("Fresh Orange")
    elif preds[3]==np.max(preds):
        print("Rotten Apple")
    elif preds[4]==np.max(preds):
        print("Rotten Banana")
    elif preds[5]==np.max(preds):
        print("Rotten Orange")
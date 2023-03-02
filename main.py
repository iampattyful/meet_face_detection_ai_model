import json  
import boto3
from sanic import Sanic
from sanic.response import json
import imutils
import numpy as np
import cv2
import dotenv
import os
from urllib.request import urlopen
# import requests

app = Sanic("Meet")


dotenv.load_dotenv()

AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")

@app.post("/face_detection")
def face_detection(request):
    print("face detection result:")
    filename = (request.json)["filename"]

    BUCKET_NAME ="meet-tecky"
    AWS_REGION_NAME="us-west-1"
    # AWS_SECRET_KEY = "your secret key"
    # AWS_ACCESS_KEY = "your access key"
    client_s3=boto3.client(service_name="s3", aws_access_key_id=AWS_ACCESS_KEY,aws_secret_access_key=AWS_SECRET_KEY,region_name=AWS_REGION_NAME)
    
    
    presignedURL= client_s3.generate_presigned_url('get_object', Params={'Bucket': BUCKET_NAME, 'Key': filename}, ExpiresIn=3600)    
    print("s3PresignedURL: \n", presignedURL)

    image = imutils.url_to_image(presignedURL)
    # image = cv2.imread(image_file)
    image = imutils.resize(image, width=400)
    (h, w) = image.shape[:2]
    print("image size: ", w,h)
    prototxt = 'deploy.prototxt'
    model = 'res10_300x300_ssd_iter_140000.caffemodel'
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    image = imutils.resize(image, width=400)
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    isFace = False

    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence threshold
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # draw the bounding box of the face along with the associated probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            print("confidence: ",confidence)
            isFace = True
    
    print(isFace)	
    return json({ "isFace" : isFace})


    
def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)
    # return the image
    return image    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, single_process=True)

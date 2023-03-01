from sanic import Sanic
from sanic.response import json
import imutils
import numpy as np
import cv2
from urllib.request import urlopen

app = Sanic("Meet")

# change to post method
# write some logic to get the image url from aws s3
# insert url into the image_file variable
# if the image is a face, return true
# if the image is not a face, return false

@app.get("/")
def face_detection(request):
    print("face detection result:")
    # content = request.json
    # image_file = content["image_file"] # filename = enroll form user icon if pass upload to (users.user_icon)
    image = imutils.url_to_image('https://img.freepik.com/free-photo/close-up-portrait-young-bearded-man-face_171337-2887.jpg?w=1800&t=st=1677637603~exp=1677638203~hmac=833313d628c0a412da7fa44c822ff0fa59d3ea8792a9e67acc0cba6dfe5cc205')
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
    return json(isFace)
    
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

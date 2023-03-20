# meet_face_detection_ai_model

## Meet – Mobile dating web application (https://clsfei.link/)

*Summary*:	Meet is an mobile dating web application for users to meet other users using artificial intelligence technology to conduct face detection verification in registration. After registering an account successfully, users can view other users’ profiles based on the filtered result and send instant messaging within the private chat room only if they have liked each other.  

*Tech Stack*:	OpenCV, Numpy, Sanic, Python, Knex.js, TypeScript, JavaScript, HTML, CSS, Bootstrap, Node.js, Express.js, Socket.IO, PostgreSQL, NGINX, Certbot, AWS EC2, AWS Route 53, AWS S3


I used following blog as a reference:  
[Face detection with OpenCV and deep learning](https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/)

> My contribution in this project:
> - deployed this python model to AWS EC2 at (https://ai.clsfei.link/)
> - utilizing Amazon S3 service to post user image from Express server via RESTful API to this face detection model to verify the presence of human face in the posted image
> - the pre-trained model of the reference is modified by me to fit our project's need

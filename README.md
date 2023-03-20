# meet_face_detection_ai_model

## Meet – Mobile dating web application (https://clsfei.link/)

*Summary*:	Meet is an mobile dating web application for users to meet other users using artificial intelligence technology to conduct face detection verification in registration. After registering an account successfully, users can view other users’ profiles based on the filtered result and send instant messaging within the private chat room only if they have liked each other.  

*Tech Stack*:	OpenCV, Numpy, Sanic, Python, Knex.js, TypeScript, JavaScript, HTML, CSS, Bootstrap, Node.js, Express.js, Socket.IO, PostgreSQL, NGINX, Certbot, AWS EC2, AWS Route 53, AWS S3

> ### *My contribution in this project:*
>
> **Artifical Intelligence Model**
> - deployed this python model to AWS EC2 at (https://ai.clsfei.link/)
> - utilizing Amazon S3 service to post user image from Express server via RESTful API to this face detection model to verify the presence of human face in the posted image
> - modified the pre-trained model with reference to [Face detection with OpenCV and deep learning](https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/) to fit our project's need
>
> Please refer to [meet_server](https://github.com/iampattyful/meet_server) for front end and back end development source codes.
>
> **Front end development**
> * *Login/Enroll new user function:* p3/public/index.html, p3/public/index.css, p3/public/index.js, p3/public/enroll.js
> * *Filter request function:* p3/public/main.html, p3/public/filter_jorm.css, p3/public/filter_jorm.js
> * *Error-handling page:* p3/public/error.html, p3/public/error.css
>
> **Back end development**
> * *User controller and service:* p3/src/controller/userController.ts, p3/src/service/userService.ts, p3/src/guard.ts
> * *Filter controller and service:* p3/src/controller/filterController.ts, p3/src/service/filterService.ts
> * *Amazon S3 helper:* p3/src/aws.ts
> * *File-handling helper:* p3/src/helper.ts
> * *Error-handling helper:* p3/src/error.ts
>
> **Database development**
> * *Knex Seed:* p3/src/seeds/p3.ts

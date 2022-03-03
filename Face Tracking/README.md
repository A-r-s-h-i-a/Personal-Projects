# Face Tracking
A utilization of Haar Cascades/Viola-Jones, an old (2001) but powerful face detection algorithm. This algorithm applies line/edge and feature detection to identify faces. A trained version of the model is available through OpenCV. There are multiple versions of the model for face detection, eye detection, and even upper-body detection amongst other things. Here, OpenCV is used to access an available webcam. Then, frames from the videostream have the face detection algorithm applied. If a face is detected, a box is drawn around it and the webcam output is altered to visualize this detected face-with-box. There is a command-line text output as well.

# Results
![Boxed Face](https://github.com/A-r-s-h-i-a/Personal-Projects/blob/main/Face%20Tracking/detected_face_boxed.png)

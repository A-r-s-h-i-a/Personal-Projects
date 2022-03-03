"""
Arshia Firouzi
07/30/2021
An implementation of Haar Cascades/Viola-Jones , an old (2001) but powerful face detection algorithm.
This algorithm uses line/edge and feature detection to identify faces. A trained version of the model is available through OpenCV.
There are multiple versions of the model for face detection, eye detection, and even upper-body detection amongst other things.
"""

import cv2



if __name__ == '__main__':
	
	#Setup Haar Cascade(s)
	face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
	eyeglasses_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

	#Create webcam feed
	cap = cv2.VideoCapture(0) #0 is the index/ID of the video device to capture from
	if not cap.isOpened(): #Check if webcam is opened correctly
		raise Exception("Webcam not accessible!")

	while True:
		#Capture/Process the current frame from the webcam
		ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert frame to grey for the Haar approach to work
		
		#Search for and ID faces
		faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5) #returns the detected face's rectangle (bottom-left x,y and width+height to top-right)
		for (x, y, w, h) in faces:
			print("FACE DETECTED: ", x, y, w, h)

			#Extract and save face
			roi_gray = gray[y:y+h, x:x+w] #gray region of interest
			roi_color = frame[y:y+h, x:x+w] #color region of interest
			img_item_gray = "detected_face_gray.png"
			img_item_color = "detected_face_color.png"
			cv2.imwrite(img_item_gray, roi_gray)
			cv2.imwrite(img_item_color, roi_color)

			#Draw rectangle around face in-frame
			rectangle_color = (255, 0, 0) #BGR
			stroke = 2 #thickness
			x2 = x+w #top-right
			y2 = y+h #top-right
			cv2.rectangle(frame, (x,y), (x2,y2), rectangle_color, stroke)

			#Search for and ID Haar sub-features
			subitems = eyeglasses_cascade.detectMultiScale(roi_gray) #search within the roi, where the detected face is
			for (sx, sy, sw, sh) in subitems:
				cv2.rectangle(roi_color, (sx,sy), (sx+sw,sy+sh), (0,255,0), 2)

		#Display the resulting frame
		cv2.imshow('Webcam', frame)

		#Provide the opportunity to end the stream
		usr_inpt = cv2.waitKey(1)
		if usr_inpt == 27: #27 is the ASCII value for the Escape key
			break

	#Release webcam and destroy all windows
	cap.release()
	cv2.destroyAllWindows()

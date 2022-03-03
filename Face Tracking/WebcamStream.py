"""
Arshia Firouzi
07/29/2021
The goal of this script is to stream video from the webcam.
"""

import cv2

if __name__ == '__main__':
	
	#Create webcam feed
	cap = cv2.VideoCapture(0) #0 is the index/ID of the video device to capture from
	if not cap.isOpened(): #Check if webcam is opened correctly
		raise Exception("Webcam not accessible!")

	while True:
		ret, frame = cap.read() #read current frame from webcam
		cv2.imshow('Webcam', frame)

		usr_inpt = cv2.waitKey(1)
		if usr_inpt == 27: #27 is the ASCII value for the Escape key
			break

	#Release webcam and destroy all windows
	cap.release()
	cv2.destroyAllWindows()

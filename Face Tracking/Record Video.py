"""
Arshia Firouzi
07/30/2021
Simple video recording script
"""

import cv2

if __name__ == '__main__':

    #Create webcam feed
    cap = cv2.VideoCapture(0) #0 is the index/ID of the video device to capture from
    if cap.isOpened() == False: #Check if webcam is opened correctly
        raise Exception("Webcam not accessible!")

    #Get the webcam's resolution
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)

    #Create video writer object
    video = cv2.VideoWriter('./Videos/60sClip.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, size)

    while True:
        #Capture/Process the current frame from the webcam
        ret, frame = cap.read()

        #Write the frame into the video file
        video.write(frame)

        #Display the frame
        cv2.imshow('Webcam', frame)

        #Provide Opportunity to exit application
        usr_inpt = cv2.waitKey(1)
        if usr_inpt == 27: #27 is the ASCII value for the Escape key
            break

    #Release webcam and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

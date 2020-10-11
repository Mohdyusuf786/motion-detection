import cv2
import numpy as np

cap = cv2.VideoCapture('cctv.mp4')


ret, frame1 = cap.read()
ret, frame2 = cap.read()
print(frame1.shape)
while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2) #to find the absolute difference between frame1 and frame2
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) #we convert BGR to gray scale image because we are finding the contour in the later stage
    blur = cv2.GaussianBlur(gray, (5,5), 0) #to smooth the gray scale video using gaussian blur
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY) #to find the binary threshold of smoothened image
    dilated = cv2.dilate(thresh, None, iterations=3) #dilating the thresholded image to fill all the holes for finding perfect contour
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #to find the contour of dilated image ,cv2.RETR_TREE is the contour retrieval mode
    for contour in contours:
        (x,y,w,h)=cv2.boundingRect(contour) #to draw the rectangle around contours
        if cv2.contourArea(contour)<900: #minimum area of contour to get the rectangle
            continue
        cv2.rectangle(frame1, (x,y), (x+w,y+h), (0,255,0),3)
        cv2.putText(frame1, "Status:{}".format(' Movement'), (10,50), cv2.FONT_ITALIC, 2,(255,255,255), 5)

    if ret == True:
        frame1 = cv2.resize(frame1,(500,500))

    cv2.imshow("feed", frame1)
    frame1 = frame2 #assigning the frame2 to frame1
    ret, frame2 = cap.read() #again reading the frame2 to get the absolute difference between 2 frames
    if cv2.waitKey(40)==ord('q'):
        break

cv2.destroyAllWindows()
cap.release()





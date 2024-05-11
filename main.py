import cv2
import numpy as np
import time 
from utils import maunau
import threading


class Camera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()
        
cap = cv2.VideoCapture(1)

pTime = 0
frame_count = 0
fps = 0

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (480, 360))
    grayed =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayed, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 250)

    _, threshold = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(threshold, kernel, iterations=1)  
    eroded = cv2.erode(dilated, kernel, iterations=1)  

    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Tạo bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Goi ra màu nâu
        roi = frame[y:y+h, x:x+w]
        global_mask = maunau(roi)
        # Kiểm tra xem có màu trong global_mask không
        if cv2.countNonZero(global_mask) > 0:
            # Nếu có màu, vẽ hộp giới hạn màu đỏ
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "ken chet", (x, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "ken trang", (x, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    frame_count += 1
    if frame_count % 10 == 0:  
        cTime = time.time()
        fps = 10 / (cTime - pTime)
        pTime = cTime
        
    cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    
    cv2.imshow('Original Image', frame)
    cv2.imshow('Thresholded Image', threshold)
    cv2.imshow('Dilated and Eroded', eroded)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   
cap.release()
cv2.destroyAllWindows()

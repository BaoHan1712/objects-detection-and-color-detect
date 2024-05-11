import cv2
import numpy as np
from utils import maunau


img = cv2.imread('train\-XZ5065L7D-GNXTM8PM-18_tmb_jpg.rf.90f171f96520a96cad7b248838a28688.jpg',cv2.COLOR_GRAY2BGR )  
img = cv2.resize(img,(480,360))

# Sử dụng Gaussian Blur để làm mờ nhiễu trước khi áp dụng Canny
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Sử dụng Canny edge detection để nhận diện cạnh
edges = cv2.Canny(blurred, 100, 250)

# Áp dụng thresholding để tạo ảnh nhị phân từ cạnh
_, threshold = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)

# Giãn nở và xói mòn để tăng cường cạnh
kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(threshold, kernel, iterations=1)  # Giãn nở các cạnh
eroded = cv2.erode(dilated, kernel, iterations=1)  # Xói mòn để làm rõ cạnh

contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Vẽ hình chữ nhật xung quanh các đường viền
for contour in contours:
    # Tạo bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    roi = img[y:y+h, x:x+w]
    global_mask = maunau(roi)
    
    # Kiểm tra xem có màu trong global_mask không
    if cv2.countNonZero(global_mask) > 0:
            # Nếu có màu, vẽ hộp giới hạn màu đỏ
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, "ken chet", (x, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "ken trang", (x, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


    
cv2.imshow('Original Image', img)
cv2.imshow('Canny Edges', edges)


# cv2.imshow('Laplacian Edges', laplacian)

cv2.waitKey(0)
cv2.destroyAllWindows()

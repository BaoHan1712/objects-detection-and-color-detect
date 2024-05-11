import cv2
import numpy as np

def maunau(roi):
    # Tạo ROI từ hình ảnh

    # Chuyển đổi ROI từ không gian màu BGR sang HSV
    roi_HSV = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Phạm vi màu da cho không gian màu HSV
    HSV_mask = cv2.inRange(roi_HSV, (0, 15, 0), (17,170,255))
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    # Chuyển đổi ROI từ gbr sang không gian màu YCbCr
    roi_YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)

    # Phạm vi màu da cho không gian màu YCrCb
    YCrCb_mask = cv2.inRange(roi_YCrCb, (0, 135, 85), (255,180,135))
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    # Tạo mặt nạ toàn cầu bằng cách kết hợp hai mặt nạ
    global_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
    global_mask = cv2.medianBlur(global_mask, 3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))

    return global_mask

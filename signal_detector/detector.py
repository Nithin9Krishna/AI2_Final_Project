import cv2
import numpy as np

class SignalDetector:
    def __init__(self):
        pass

    def detect_signals(self, rgb_image):
        if rgb_image is None:
            return "No Image"

        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

        # Red light
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

        # Yellow light
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Green light
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        red_pixels = cv2.countNonZero(red_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        green_pixels = cv2.countNonZero(green_mask)

        max_pixels = max(red_pixels, yellow_pixels, green_pixels)

        if max_pixels < 100:  # Too small to consider
            return "No Signal"
        elif max_pixels == red_pixels:
            return "Red"
        elif max_pixels == yellow_pixels:
            return "Yellow"
        else:
            return "Green"

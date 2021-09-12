import cv2
import mediapipe as mp
import time

class display_annotation:
    """display_annotation class is for displaying fps and other annotations from
    displaying on screen"""
    prev_frame_time = 0
    new_frame_time = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    def __init__(self):
        pass

    def fps(self, image):
        """display fps on display"""
        self.image = image
        display_annotation.new_frame_time = time.time()
        fps = 1/(display_annotation.new_frame_time-display_annotation.prev_frame_time)
        display_annotation.prev_frame_time = display_annotation.new_frame_time
        fps = int(fps)
        fps = str(fps)
        return cv2.putText(self.image, fps, (7, 70), display_annotation.font, 1, (100, 255, 0), 3, cv2.LINE_AA)

    def hand_landmarks(self, image, multi_hand_landmarks):
        """display hand_landmarks on display"""
        self.image = image
        self.multi_hand_landmarks = multi_hand_landmarks

        for hand_landmarks in multi_hand_landmarks:
          return display_annotation.mp_drawing.draw_landmarks(
              image,
              hand_landmarks,
              display_annotation.mp_hands.HAND_CONNECTIONS,
              display_annotation.mp_drawing_styles.get_default_hand_landmarks_style(),
              display_annotation.mp_drawing_styles.get_default_hand_connections_style())

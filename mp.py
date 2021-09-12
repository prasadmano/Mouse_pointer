import cv2
import mediapipe as mp
import time
import viewing

# if frames per second is needed put fpsdisplay to True
FPSDISPLAY = True
HANDLANDMARK = True

DISPLAY = viewing.display_annotation()

mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)
    #print(results)

    image_hight, image_width, _ = image.shape
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # for fps on screen
    if FPSDISPLAY:
        image = DISPLAY.fps(image)

    # for hand_landmarks on screen
    if ((results.multi_hand_landmarks != None) and (HANDLANDMARK == True)) :
        DISPLAY.hand_landmarks(image, results.multi_hand_landmarks)

    #cv2.circle(image, (20,100), radius=2, color=(100, 255, 0), thickness=2)
    cv2.imshow('MediaPipe Hands', image)

    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()

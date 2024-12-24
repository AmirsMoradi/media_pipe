import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]

            screen_x = int(index_finger_tip.x * screen_width)
            screen_y = int(index_finger_tip.y * screen_height)

            pyautogui.moveTo(screen_x, screen_y)

            distance = abs(index_finger_tip.y - index_finger_dip.y)

            thumb_distance = ((thumb_tip.x - index_finger_tip.x)**2 +
                              (thumb_tip.y - index_finger_tip.y)**2)**0.5

            if distance < 0.02:
                pyautogui.click()

            if thumb_distance < 0.05:
                pyautogui.rightClick()

            x, y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])
            cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)

    cv2.imshow("Hand Tracking Mouse Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import numpy as np
import time
import os

GESTURES = {
    0: "pinch",
    1: "fist",
    2: "open_hand",
    3: "peace",
    4: "thumbs_up"
}

# Create folder
if not os.path.exists("data"):
    os.makedirs("data")

for g in GESTURES.values():
    os.makedirs(f"data/{g}", exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(hand_landmarks):
    """Flatten 21 (x,y,z) landmarks into a list of 63 values."""
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

cap = cv2.VideoCapture(700)

with mp_hands.Hands(max_num_hands=1) as hands:
    print("Press keys to record gestures:")
    print(GESTURES)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            landmarks = extract_landmarks(results.multi_hand_landmarks[0])
            mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Collecting Data", frame)

        key = cv2.waitKey(1)

        if key in [ord(str(k)) for k in GESTURES.keys()]:
            g = GESTURES[int(chr(key))]
            timestamp = time.time()
            np.save(f"data/{g}/{timestamp}.npy", landmarks)
            print(f"Saved sample for gesture: {g}")

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

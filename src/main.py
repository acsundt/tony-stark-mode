import cv2
import mediapipe as mp
import numpy as np
import torch
from model_train import GestureModel  # Your MLP class definition
from gesture_actions import GESTURE_ACTIONS
import time

GESTURES = ["fist", "open_hand", "pinch", "peace", "thumbs_up"]

# Configuration
COOLDOWN = 0.5  # seconds between actions
BUFFER_SIZE = 20  # frames for smoothing
DEVICE = "cpu"

# State
last_action_time = 0
gesture_buffer = []
last_triggered_gesture = None

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load PyTorch model
model = GestureModel().to(DEVICE)
model.load_state_dict(torch.load("jarvis.pth", map_location=DEVICE))
model.eval()




# Extract 21 hand landmarks
def extract_landmarks(hand_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()



cap = cv2.VideoCapture(700)  # adjust camera index

with mp_hands.Hands(max_num_hands=1) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract features and predict
            features = extract_landmarks(hand_landmarks)
            features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred = model(features).argmax(dim=1).item()

            # Add to buffer for smoothing
            gesture_buffer.append(pred)
            if len(gesture_buffer) > BUFFER_SIZE:
                gesture_buffer.pop(0)

            # Determine most common gesture in buffer
            most_common = max(set(gesture_buffer), key=gesture_buffer.count)

            # Check cooldown and trigger action if stable
            current_time = time.time()
            if most_common != last_triggered_gesture and gesture_buffer.count(most_common) > BUFFER_SIZE // 2:
                if current_time - last_action_time > COOLDOWN:
                    last_triggered_gesture = most_common
                    last_action_time = current_time

                    gesture_name = GESTURES[most_common]
                    if gesture_name in GESTURE_ACTIONS:
                        GESTURE_ACTIONS[gesture_name]()  # trigger action
                        print(f"Action triggered: {gesture_name}")

            # Display current prediction
            cv2.putText(frame, GESTURES[most_common], (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Jarvis Copilot", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

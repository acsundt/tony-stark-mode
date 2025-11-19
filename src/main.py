import cv2
import mediapipe as mp
import numpy as np
import torch
from model_train import GestureModel  # Import your MLP class definition

GESTURES = ["fist", "open_hand", "pinch", "peace"]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load model
DEVICE = "cpu"
model = GestureModel().to(DEVICE)
model.load_state_dict(torch.load("jarvis.pth", map_location=DEVICE))
model.eval()


BUFFER_SIZE = 20  # Number of frames to check
gesture_buffer = []
last_triggered_gesture = None


# Hand landmarks extraction
def extract_landmarks(hand_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

cap = cv2.VideoCapture(700)  # your camera index

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

            features = extract_landmarks(hand_landmarks)
            features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # PyTorch prediction
            with torch.no_grad():
                pred = model(features).argmax(dim=1).item()

            # Add to buffer
            gesture_buffer.append(pred)
            if len(gesture_buffer) > BUFFER_SIZE:
                gesture_buffer.pop(0)  # keep buffer size constant

            # Find most common gesture in buffer
            most_common = max(set(gesture_buffer), key=gesture_buffer.count)

            # Trigger only if it has changed and is stable
            if most_common != last_triggered_gesture and gesture_buffer.count(most_common) > BUFFER_SIZE // 2:
                last_triggered_gesture = most_common
                # Here you can call your gesture action, e.g., volume up/down
                print(f"Gesture triggered: {GESTURES[most_common]}")

            # Display prediction
            cv2.putText(frame, GESTURES[pred], (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Jarvis Copilot", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

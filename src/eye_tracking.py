# eye_tracking.py - Updated for eye/eyebrow gestures
import cv2
import pyautogui
import mediapipe as mp
import numpy as np


class EyeTracker:
    def __init__(self, smoothing=0.05, sensitivity=200):
        """
        :param smoothing: Duration for pyautogui.moveTo
        :param sensitivity: Pixel movement for gestures
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.screen_width, self.screen_height = pyautogui.size()
        self.smoothing = smoothing
        self.sensitivity = sensitivity

        self.x, self.y = self.screen_width / 2, self.screen_height / 2
        self.eyebrow_center = None
        self.calibration_frames = []
        self.calibrated = False

    def _eye_aspect_ratio(self, landmarks, indices):
        """Compute a simple eye aspect ratio to detect blinking."""
        # Vertical distances
        v1 = np.linalg.norm(np.array([landmarks[indices[1]].x, landmarks[indices[1]].y]) -
                            np.array([landmarks[indices[5]].x, landmarks[indices[5]].y]))
        v2 = np.linalg.norm(np.array([landmarks[indices[2]].x, landmarks[indices[2]].y]) -
                            np.array([landmarks[indices[4]].x, landmarks[indices[4]].y]))
        # Horizontal distance
        h = np.linalg.norm(np.array([landmarks[indices[0]].x, landmarks[indices[0]].y]) -
                           np.array([landmarks[indices[3]].x, landmarks[indices[3]].y]))
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def _eyebrow_raise_ratio(self, landmarks, eye_indices, brow_indices):
        """Compare eyebrow height to eye to detect raising/scrunching."""
        eye_center_y = np.mean([landmarks[i].y for i in eye_indices])
        brow_center_y = np.mean([landmarks[i].y for i in brow_indices])
        return eye_center_y - brow_center_y  # Positive if brows are up

    def calibrate(self, num_frames=60):
        """Call this before main loop with frames"""
        print(f"Calibrating for {num_frames} frames...")

    def update(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            try:
                # Landmark indices for eyes and eyebrows (mediapipe face mesh)
                LEFT_EYE = [33, 160, 158, 133, 153, 144]
                RIGHT_EYE = [362, 385, 387, 263, 373, 380]
                LEFT_BROW = [70, 63, 105, 66]
                RIGHT_BROW = [300, 293, 334, 296]

                # Compute eye aspect ratio
                left_ear = self._eye_aspect_ratio(landmarks, LEFT_EYE)
                right_ear = self._eye_aspect_ratio(landmarks, RIGHT_EYE)

                # Compute eyebrow ratio
                brow_ratio = (self._eyebrow_raise_ratio(landmarks, LEFT_EYE, LEFT_BROW) +
                              self._eyebrow_raise_ratio(landmarks, RIGHT_EYE, RIGHT_BROW)) / 2

                # Calibration phase for eyebrow center
                if not self.calibrated and len(self.calibration_frames) < 60:
                    self.calibration_frames.append(brow_ratio)
                    if len(self.calibration_frames) == 60:
                        self.eyebrow_center = np.mean(self.calibration_frames)
                        self.calibrated = True
                        print(f"✓ Calibrated! Eyebrow center ratio: {self.eyebrow_center:.3f}")

                # Gesture-based movement
                if self.calibrated:
                    dx = 0
                    dy = 0

                    # Eye blink gestures (thresholds may need tuning)
                    if right_ear < 0.18:  # Right eye closed
                        dx += self.sensitivity
                    if left_ear < 0.18:  # Left eye closed
                        dx -= self.sensitivity

                    # Eyebrow gestures
                    brow_offset = brow_ratio - self.eyebrow_center
                    if brow_offset > 0.02:  # Raised eyebrows
                        dy -= self.sensitivity
                    elif brow_offset < -0.015:  # Scrunched eyebrows
                        dy += self.sensitivity

                    # Update cursor position
                    self.x = max(0, min(self.screen_width, self.x + dx))
                    self.y = max(0, min(self.screen_height, self.y + dy))
                    pyautogui.moveTo(self.x, self.y, duration=self.smoothing)

            except Exception as e:
                print(f"Eye tracking error: {e}")

        return self.x, self.y

    def reset_calibration(self):
        self.calibration_frames = []
        self.calibrated = False
        self.eyebrow_center = None
        print("Calibration reset")

    def release(self):
        pass

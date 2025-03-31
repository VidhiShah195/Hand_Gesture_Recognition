import cv2
import mediapipe as mp
import numpy as np
import csv
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

GESTURES = {
    "open_palm": 0,  # No filter
    "fist": 1,  # Grayscale
    "peace_sign": 2,  # Sepia
    "thumbs_up": 3,  # Blur
    "pointing_finger": 4,  # Edge detection
    "ok_sign": 5  # Cartoon filter
}

DATA_PATH = "gesture_data"
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

def normalize_landmarks(landmarks):
    data = np.array([[lm.x, lm.y] for lm in landmarks])
    center = data[0]
    data -= center
    
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data /= max_val
    
    return data.flatten()

def collect_data(gesture_name):
    label = GESTURES[gesture_name]
    output_file = os.path.join(DATA_PATH, f"{gesture_name}.csv")

    print(f"Collecting data for: {gesture_name} (Press 's' to save, 'q' to quit)")

    cap = cv2.VideoCapture(0)
    count = 0
    data_samples = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                normalized_data = normalize_landmarks(hand_landmarks.landmark)

                if cv2.waitKey(1) & 0xFF == ord('s'):
                    data_samples.append([label] + list(normalized_data))
                    count += 1
                    print(f"Saved {count} samples for {gesture_name}")

                if count >= 100:
                    print(f"Finished collecting data for {gesture_name}")
                    cap.release()
                    cv2.destroyAllWindows()
                    break

        cv2.imshow("Gesture Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data_samples)
    
    print(f"Data saved in {output_file}")

for gesture in GESTURES.keys():
    collect_data(gesture)

import cv2
import mediapipe as mp
import torch
import numpy as np
import torch.nn as nn

class GestureModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GestureModel, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc3 = nn.Linear(hidden_size, output_size) 
        self.softmax = nn.Softmax(dim=1) 

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.softmax(x)

def grayscale(image):
    return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

def sepia(image):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_img = cv2.transform(image, sepia_filter)
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
    return sepia_img

def blur(image, ksize=15):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def cartoon(image):
    color = cv2.bilateralFilter(image, 9, 300, 300)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

input_size = 42 
hidden_size = 128
output_size = 6 
model = GestureModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
model.load_state_dict(torch.load('gesture_model.pth'))
model.eval()

gesture_to_filter = {
    0: lambda x: x,  # open_palm - no filter (return original)
    1: grayscale,  # fist
    2: sepia,  # peace_sign
    3: blur,  # thumbs_up
    4: edge_detection,  # pointing_finger
    5: cartoon  # ok_sign
}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def normalize_landmarks(landmarks):
    data = np.array([[lm.x, lm.y] for lm in landmarks])
    center = data[0]
    data -= center
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data /= max_val
    return data.flatten()

def predict_gesture(landmarks):
    normalized_data = normalize_landmarks(landmarks)
    input_tensor = torch.tensor(normalized_data, dtype=torch.float32).unsqueeze(0)
    output = model(input_tensor)
    _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    filtered_frame = frame.copy() 

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            gesture_class = predict_gesture(hand_landmarks.landmark)
            
            filtered_frame = gesture_to_filter.get(gesture_class, lambda x: x)(frame)
            
            frame_with_landmarks = frame.copy()

            dot_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
            line_spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1) 

            mp_drawing.draw_landmarks(frame_with_landmarks, hand_landmarks, mp_hands.HAND_CONNECTIONS, dot_spec, line_spec)

            gesture_names = {
                0: "No Filter",
                1: "Filter: Grayscale",
                2: "Filter: Sepia",
                3: "Filter: Blur",
                4: "Filter: Edges",
                5: "Filter: Cartoon"
            }
            
            gesture_text = gesture_names.get(gesture_class, " ")
            cv2.putText(filtered_frame, gesture_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            
            combined_frame = np.hstack((frame_with_landmarks, filtered_frame))
            cv2.imshow('Gesture Filtering App', combined_frame)

    else:
        combined_frame = np.hstack((frame, filtered_frame))
        cv2.imshow('Gesture Filtering App', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
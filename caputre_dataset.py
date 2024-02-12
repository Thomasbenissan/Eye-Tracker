import cv2
import mediapipe as mp
import pyautogui
import time
import os

# Directory to save the dataset
dataset_dir = "eye_tracking_dataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Initialize camera and face mesh detector
cam = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

frame_count = 0
while True:
    ret, frame = cam.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    screen_w, screen_h = pyautogui.size()
    cursor_x, cursor_y = pyautogui.position()

    if output.multi_face_landmarks:
        for face_landmarks in output.multi_face_landmarks:
            # Specify landmarks around the eyes
            eye_landmarks = [
                33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7,  # Left eye
                362, 466, 388, 387, 386, 385, 384, 398, 263, 373, 374, 380, 381, 382, 390, 249  # Right eye
            ] 
            x_coords = []
            y_coords = []

            for landmark in eye_landmarks:
                x = int(face_landmarks.landmark[landmark].x * frame.shape[1])
                y = int(face_landmarks.landmark[landmark].y * frame.shape[0])
                x_coords.append(x)
                y_coords.append(y)

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Expand the bounding box slightly
            expand_margin = 10
            x_min = max(x_min - expand_margin, 0)
            x_max = min(x_max + expand_margin, frame.shape[1])
            y_min = max(y_min - expand_margin, 0)
            y_max = min(y_max + expand_margin, frame.shape[0])

            # Crop the image to the eyes
            cropped_image = frame[y_min:y_max, x_min:x_max]

            # Save the cropped image and cursor position
            filename = f"{frame_count}.png"
            cv2.imwrite(os.path.join(dataset_dir, filename), cropped_image)
            with open(os.path.join(dataset_dir, f"{frame_count}.txt"), 'w') as f:
                f.write(f"{cursor_x},{cursor_y}")

            frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

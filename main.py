import cv2
import mediapipe as mp

# Initialize camera and face mesh detector
cam = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    
    frame_h, frame_w, _ = frame.shape
    cropped_image = frame.copy()

    if output.multi_face_landmarks:
        for face_landmarks in output.multi_face_landmarks:
            eye_landmarks = [
                33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7,  # Left eye
                362, 466, 388, 387, 386, 385, 384, 398, 263, 373, 374, 380, 381, 382, 390, 249  # Right eye
            ]   
            x_coords = []
            y_coords = []

            for landmark in eye_landmarks:
                x = int(face_landmarks.landmark[landmark].x * frame_w)
                y = int(face_landmarks.landmark[landmark].y * frame_h)
                x_coords.append(x)
                y_coords.append(y)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Draw eye landmarks

            # Calculate the bounding box for cropping
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Expand the bounding box slightly to ensure it encapsulates the full eye regions
            expand_margin = 10  # Adjust as needed
            x_min = max(x_min - expand_margin, 0)
            x_max = min(x_max + expand_margin, frame_w)
            y_min = max(y_min - expand_margin, 0)
            y_max = min(y_max + expand_margin, frame_h)

            # Crop the image to the calculated bounding box
            cropped_image = frame[y_min:y_max, x_min:x_max]

    # Flip the frame for a mirror view
    flipped_frame = cv2.flip(cropped_image, 1)
    cv2.imshow('Eyes Only', flipped_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

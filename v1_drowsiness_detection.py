import cv2
import dlib
import numpy as np
import time

# Initialize variables for closed eye duration tracking
closed_eyes_start_time = None
CLOSED_EYES_DURATION_THRESHOLD = 1

# Load dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor_path = "C:\\Users\\Wesley\\Desktop\\Drowsiness Detection\\shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def detect_faces(gray):
    # Detect faces using dlib's face detector
    faces = detector(gray)

    return faces

def detect_eyes(face_region):
    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

    # Create a rectangle to specify the face region
    face_rect = dlib.rectangle(0, 0, face_region.shape[1], face_region.shape[0])

    # Detect eyes using dlib's facial landmark predictor
    landmarks = predictor(gray, face_rect)
    left_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
    right_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

    # Calculate eye bounding rectangles
    left_eye_rect = cv2.boundingRect(left_eye_points)
    right_eye_rect = cv2.boundingRect(right_eye_points)

    return [left_eye_rect, right_eye_rect]

def eye_aspect_ratio(eye):
    if len(eye) >= 6:
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
    else:
        return 0.0  # Return a default value if eye has insufficient points

def display_message(frame, text):
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Drowsiness Detection", frame)

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)

    left_eye_ear = 0.0
    right_eye_ear = 0.0

    for face in faces:
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        face_region = frame[y:y + h, x:x + w]
        eyes = detect_eyes(face_region)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        shape = predictor(gray, face)
        shape = shape_to_np(shape)

        for i, (x, y) in enumerate(shape):
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            #cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for (ex, ey, ew, eh) in eyes:
            eye_region = face_region[ey:ey + eh, ex:ex + ew]
            eye_ear = eye_aspect_ratio(eye_region)

            if ex < ew // 2:  # Left eye
                left_eye_ear = eye_ear
            else:  # Right eye
                right_eye_ear = eye_ear

        if left_eye_ear < EAR_THRESHOLD and right_eye_ear < EAR_THRESHOLD:
            if closed_eyes_start_time is None:
                closed_eyes_start_time = time.time()
            else:
                elapsed_time = time.time() - closed_eyes_start_time
                if elapsed_time >= CLOSED_EYES_DURATION_THRESHOLD:
                    display_message(frame, "Wake up!")
        else:
            closed_eyes_start_time = None

    return frame

def run_drowsiness_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open the webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read the frame.")
            break

        processed_frame = process_frame(frame)
        cv2.imshow("Video", processed_frame)

        if cv2.waitKey(1) == ord('q'):
            break

        time.sleep(0.01)  # Add a small delay between frames

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    EAR_THRESHOLD = 0.25  # Adjust this value according to your needs
    run_drowsiness_detection()

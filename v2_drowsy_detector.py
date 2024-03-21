import cv2
import dlib
import pyttsx3
from scipy.spatial import distance
import time

# INITIALIZING THE pyttsx3 SO THAT
# ALERT AUDIO MESSAGE CAN BE DELIVERED
engine = pyttsx3.init()

# SETTING UP OF CAMERA TO 1 YOU CAN
# EVEN CHOOSE 0 IN PLACE OF 1
cap = cv2.VideoCapture(0)

# FACE DETECTION OR MAPPING THE FACE TO
# GET THE Eye AND EYES DETECTED
face_detector = dlib.get_frontal_face_detector()

# PUT THE LOCATION OF .DAT FILE (FILE FOR
# PREDECTING THE LANDMARKS ON FACE )
dlib_facelandmark = dlib.shape_predictor(
	"C:\\Users\\Wesley\\Desktop\\CODE\\Drowsiness Detection\\shape_predictor_68_face_landmarks.dat")

# set a variable to calculate elapsed time
closed_eyes_start_time = 0
EAR_THRESHOLD = 0.3  # Adjust this value according to your needs
DROWSINESS_THRESHOLD = 1 #how long in seconds for it to register as drowsy


# FUNCTION CALCULATING THE ASPECT RATIO FOR
# THE Eye BY USING EUCLIDEAN DISTANCE FUNCTION
def Detect_Eye(eye):
	poi_A = distance.euclidean(eye[1], eye[5])
	poi_B = distance.euclidean(eye[2], eye[4])
	poi_C = distance.euclidean(eye[0], eye[3])
	aspect_ratio_Eye = (poi_A+poi_B)/(2*poi_C)
	return aspect_ratio_Eye


# MAIN LOOP IT WILL RUN ALL THE UNLESS AND
# UNTIL THE PROGRAM IS BEING KILLED BY THE USER
while True:
	null, frame = cap.read()
	gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = face_detector(gray_scale)

	for face in faces:
		face_landmarks = dlib_facelandmark(gray_scale, face)
		
		for n in range(0, 68):
			x = face_landmarks.part(n).x
			y = face_landmarks.part(n).y
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
		
		leftEye = []
		rightEye = []
		
		# THESE ARE THE POINTS ALLOCATION FOR THE
		# LEFT EYES IN .DAT FILE THAT ARE FROM 42 TO 47
		for n in range(42, 48):
			x = face_landmarks.part(n).x
			y = face_landmarks.part(n).y
			rightEye.append((x, y))
			next_point = n+1
			if n == 47:
				next_point = 42
			x2 = face_landmarks.part(next_point).x
			y2 = face_landmarks.part(next_point).y
			cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

		# THESE ARE THE POINTS ALLOCATION FOR THE
		# RIGHT EYES IN .DAT FILE THAT ARE FROM 36 TO 41
		for n in range(36, 42):
			x = face_landmarks.part(n).x
			y = face_landmarks.part(n).y
			leftEye.append((x, y))
			next_point = n+1
			if n == 41:
				next_point = 36
			x2 = face_landmarks.part(next_point).x
			y2 = face_landmarks.part(next_point).y
			cv2.line(frame, (x, y), (x2, y2), (255, 255, 0), 1)

		# CALCULATING THE ASPECT RATIO FOR LEFT
		# AND RIGHT EYE
		right_Eye = Detect_Eye(rightEye)
		left_Eye = Detect_Eye(leftEye)
		Eye_Rat = (left_Eye+right_Eye)/2

		# NOW ROUND OF THE VALUE OF AVERAGE MEAN
		# OF RIGHT AND LEFT EYES
		Eye_Rat = round(Eye_Rat, 2)
		print(Eye_Rat)


		if Eye_Rat < 0.25:
			if closed_eyes_start_time is None:
				closed_eyes_start_time = time.time()
			else:
				elapsed_time = time.time() - closed_eyes_start_time
				#print(elapsed_time)
				if elapsed_time > DROWSINESS_THRESHOLD:
					cv2.putText(frame, "DROWSINESS DETECTED", (50, 100),
								cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 210), 3)
					cv2.putText(frame, "Alert!!!! WAKE UP DUDE", (50, 450),
								cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)
					# audio cues
					engine.say("Wake up bro, time to grind.")
					engine.runAndWait()
		else:
			closed_eyes_start_time = None

	cv2.imshow("Drowsiness DETECTOR IN OPENCV2", frame)
	#can make it another key so that it keeps looping lol
	if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

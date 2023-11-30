import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from joblib import Parallel, delayed
from tf_model import Model

#Site Packages Changes
'''
$HOME/pythonvenv/python310/lib/python3.10/site-packages/tensorflow/lite/python/interpreter.py
Stopped  from tensorflow.lite.python.interpreter_wrapper import _pywrap_tensorflow_interpreter_wrapper as _interpreter_wrapper

Reaon - tflite_runtime
'''

VISUAL_LANDMARKS=True

#Drawing Models
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


#Detection Model
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

#for Right hand max_num_hands should be 2
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Function to process an image and extract facial landmarks
def extract_facial_landmarks(image_rgb, frame):

	results = face_mesh.process(image_rgb)

	# Check if facial landmarks are detected
	face_x = []
	face_y = []
	face_z = []

	if results.multi_face_landmarks:

		for i, face_landmarks in enumerate(results.multi_face_landmarks):
			if VISUAL_LANDMARKS:
				mp_drawing.draw_landmarks(
					frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
					landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
					connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
				)
			for idx, landmark in enumerate(face_landmarks.landmark):
			
				# Get the x, y, and z coordinates of each landmark
				x, y, z = landmark.x, landmark.y, landmark.z

				face_x.append(x)
				face_y.append(y)
				face_z.append(z)
	
	return {'x':face_x, 'y':face_y, 'z':face_z}


def extract_hand_landmarks(image_rgb, frame):

	# Process the image with the hands model
	results = hands.process(image_rgb)

	hand_x = []
	hand_y = []
	hand_z = []

	left_hand_x = []
	left_hand_y = []
	left_hand_z = []

	right_hand_x = []
	right_hand_y = []
	right_hand_z = []
    
    # Check if hand landmarks are detected
	if results.multi_hand_landmarks:
		for hand_landmarks in results.multi_hand_landmarks:
			if VISUAL_LANDMARKS:
				mp_drawing.draw_landmarks(
					frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
					landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
					connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
				)
			for idx, landmark in enumerate(hand_landmarks.landmark):
				# Get the x, y, and z coordinates of each landmark
				x, y, z = landmark.x, landmark.y, landmark.z
				if idx == 0:
					thumb_x = landmark.x
				if idx == 17:
					pinky_x = landmark.x
				
				hand_x.append(x)
				hand_y.append(y)
				hand_z.append(z)
			
			if thumb_x < pinky_x:
				left_hand_x = hand_x
				left_hand_y = hand_y
				left_hand_z = hand_z
			
			else:
				right_hand_x = hand_x
				right_hand_y = hand_y
				right_hand_z = hand_z

	left_hand = {'x': left_hand_x, 'y': left_hand_y, 'z': left_hand_z}
	right_hand = {'x': right_hand_x, 'y': right_hand_y, 'z': right_hand_z}


	return {'left_hand':left_hand, 'right_hand':right_hand}


		
def extract_pose_landmarks(image_rgb):

	pose_x = []
	pose_y = []
	pose_z = []
	# Process the image with the pose model
	results = pose.process(image_rgb)

    # Check if pose landmarks are detected
	if results.pose_landmarks:
		for idx, landmark in enumerate(results.pose_landmarks.landmark):
			# Get the x, y, and z coordinates of each landmark
			x, y, z = landmark.x, landmark.y, landmark.z

			pose_x.append(x)
			pose_y.append(y)
			pose_z.append(z)
			
	return {'x':pose_x, 'y':pose_y, 'z':pose_z}


# Example usage
# image_path = "sample.jpg"
# extract_facial_landmarks(image_path)

def processingLandmarks(face_landmarks, hand_landmarks, pose_landmarks, Landmark_df):

	left_hand_landmarks = {}
	right_hand_landmarks = {}

	no_hand_flag = False

	if len(hand_landmarks['left_hand']['x']) < 1:
		left_hand_landmarks['x'] = [np.nan] * 21
		left_hand_landmarks['y'] = [np.nan] * 21
		left_hand_landmarks['z'] = [np.nan] * 21

		no_hand_flag = True
		
	else:
		left_hand_landmarks['x'] = hand_landmarks['left_hand']['x']
		left_hand_landmarks['y'] = hand_landmarks['left_hand']['y']
		left_hand_landmarks['z'] = hand_landmarks['left_hand']['z']


	if len(hand_landmarks['right_hand']['x']) < 1:
		right_hand_landmarks['x'] = [np.nan] * 21
		right_hand_landmarks['y'] = [np.nan] * 21
		right_hand_landmarks['z'] = [np.nan] * 21

		#if no_hand_flag:
		#	return landmark_df

	else:
		right_hand_landmarks['x'] = hand_landmarks['right_hand']['x']
		right_hand_landmarks['y'] = hand_landmarks['right_hand']['y']
		right_hand_landmarks['z'] = hand_landmarks['right_hand']['z']

	


	values = face_landmarks['x'] + face_landmarks['y'] + face_landmarks['z'] + left_hand_landmarks['x'] + left_hand_landmarks['y'] + left_hand_landmarks['z'] + right_hand_landmarks['x'] + right_hand_landmarks['y'] + right_hand_landmarks['z'] + pose_landmarks['x'] + pose_landmarks['y'] + pose_landmarks['z']

	try:
		df = pd.DataFrame([values], columns=face_cols_x + face_cols_y + face_cols_z + 
											left_hand_cols_x + left_hand_cols_y + left_hand_cols_z + 
											right_hand_cols_x + right_hand_cols_y + right_hand_cols_z + 
											pose_cols_x + pose_cols_y + pose_cols_z)
		
		Landmark_df = pd.concat([Landmark_df, df], ignore_index=True)
	except Exception as e:
		pass	
	return Landmark_df

def parallel_inference(_model, landmark_df):
	landmark_df = landmark_df.astype('float32')
	result = _model.Inference(landmark_df)
	landmark_df = landmark_df.drop(landmark_df.index)
	return result, landmark_df



if __name__=='__main__':

	face_cols_x = [f'x_face_{i}' for i in range(468)]
	face_cols_y = [f'y_face_{i}' for i in range(468)]
	face_cols_z = [f'z_face_{i}' for i in range(468)]

	pose_cols_x = [f'x_pose_{i}' for i in range(33)]
	pose_cols_y = [f'y_pose_{i}' for i in range(33)]
	pose_cols_z = [f'z_pose_{i}' for i in range(33)]

	left_hand_cols_x = [f'x_left_hand_{i}' for i in range(21)]
	left_hand_cols_y = [f'y_left_hand_{i}' for i in range(21)]
	left_hand_cols_z = [f'z_left_hand_{i}' for i in range(21)]

	right_hand_cols_x = [f'x_right_hand_{i}' for i in range(21)]
	right_hand_cols_y = [f'y_right_hand_{i}' for i in range(21)]
	right_hand_cols_z = [f'z_right_hand_{i}' for i in range(21)]

	landmark_df = pd.DataFrame(columns=face_cols_x + face_cols_y + face_cols_z + 
											left_hand_cols_x + left_hand_cols_y + left_hand_cols_z + 
											right_hand_cols_x + right_hand_cols_y + right_hand_cols_z + 
											pose_cols_x + pose_cols_y + pose_cols_z)
	
	tf_model = Model()
	
	#print(landmark_df)
	cap = cv2.VideoCapture(0)

	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	font = cv2.FONT_HERSHEY_SIMPLEX
	bottom_left_corner = (100, height - 10)
	font_scale = 0.8
	font_color = (255, 255, 255)  # White
	thickness = 2
	result = ''
	try:

		while cap.isOpened():
		# Read a frame from the camera
			ret, frame = cap.read()

			if not ret:
				print("Error reading frame")
				break

			# Convert the frame to RGB format (required by MediaPipe)
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			# Process the frame with the pose model
			pose_Landmarks = extract_pose_landmarks(frame_rgb)
			hands_Landmarks = extract_hand_landmarks(image_rgb=frame_rgb, frame=frame)
			face_Landmarks = extract_facial_landmarks(image_rgb=frame_rgb, frame=frame)


			landmark_df = processingLandmarks(face_landmarks=face_Landmarks, hand_landmarks=hands_Landmarks, pose_landmarks=pose_Landmarks, Landmark_df = landmark_df )

			if (len(landmark_df) > 128):
				#result = Parallel(n_jobs=4)([delayed(parallel_inference)(tf_model, landmark_df)])
				result, landmark_df = parallel_inference(tf_model, landmark_df)

			#Textual Rendering using Cv2
			text = result
			cv2.putText(frame, text, bottom_left_corner, font, font_scale, font_color, thickness, cv2.LINE_AA)
			cv2.imshow('Opencv', frame)


			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		
		cap.release()
		cv2.destroyAllWindows()
	except KeyboardInterrupt as e:
		landmark_df.to_csv('landmark.csv', index=False)

	landmark_df.to_csv('landmark.csv', index=False)
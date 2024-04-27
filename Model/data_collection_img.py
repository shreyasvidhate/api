import os  
import numpy as np 
import cv2 
from keras.utils import to_categorical
import pandas as pd
import math
import mediapipe as mp
import json

holistic = mp.solutions.pose
holis = holistic.Pose()
drawing = mp.solutions.drawing_utils

def get_updated_df_img(df,img):
	res = holis.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	lst = []
	if res.pose_landmarks:
    # coordinates data
		for i in res.pose_landmarks.landmark:
			lst.append(i.x-res.pose_landmarks.landmark[0].x)
			lst.append(i.y-res.pose_landmarks.landmark[0].y)
			lst.append(i.z-res.pose_landmarks.landmark[0].z)

    # angles data
	with open('reqd_angles.json','r') as f:
		data = json.load(f)
		print(data)
		for number,angles in data.items():
			lst.append(calculateAngles(res.pose_landmarks.landmark,angles[0],angles[1],angles[2]))

def calculateAngles(landmarks,a,b,c):		
	vector_ab = [landmarks[b].x - landmarks[a].x, landmarks[b].y - landmarks[a].y]
	vector_bc = [landmarks[c].x - landmarks[b].x, landmarks[c].y - landmarks[b].y]
	
	dot_product = vector_ab[0] * vector_bc[0] + vector_ab[1] * vector_bc[1]
	magnitude_ab = math.sqrt(vector_ab[0] ** 2 + vector_ab[1] ** 2)
	magnitude_bc = math.sqrt(vector_bc[0] ** 2 + vector_bc[1] ** 2)

	cosine_angle = dot_product / (magnitude_ab * magnitude_bc)

	angle_rad = math.acos(cosine_angle)
	angle_deg = math.degrees(angle_rad)
	return angle_deg

def func():
	for i in os.listdir(folder_path):
		if i=="xyz" or i=="test" or i=="Front Hands":
			continue
		asana_path = os.path.join(folder_path,i)
		images_folder_path = os.path.join(asana_path,'Vertical_sideways')      # Vertical_sideways
		X = []
		for j in os.listdir(images_folder_path):
			curr_img_folder = os.path.join(images_folder_path,j)    # v1
			key_counter = 0
			for k in os.listdir(curr_img_folder):
				if k.endswith(".jpg") or k.endswith(".jpeg") or k.endswith(".png"):
					img_path = os.path.join(curr_img_folder,k)
					img = cv2.imread(img_path)
					res = holis.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
					lst = []
					if res.pose_landmarks:
						for i in res.pose_landmarks.landmark:
							lst.append(i.x-res.pose_landmarks.landmark[0].x)
							lst.append(i.y-res.pose_landmarks.landmark[0].y)
							lst.append(i.z-res.pose_landmarks.landmark[0].z)
						with open('reqd_angles.json','r') as f:
							data = json.load(f)
						for number,angles in data.items():
							lst.append(calculateAngles(res.pose_landmarks.landmark,angles[0],angles[1],angles[2]))
						lst.append(key_counter)
						key_counter += 1
						X.append(lst)
		img_data_df = pd.DataFrame(X)
		img_data_df.columns = [i + 1 for i in range(116)]
		img_data_df = img_data_df.rename(columns={116: 'pose'})
		img_data_df.to_csv('img_data_df.csv',index=False)
		return img_data_df
			

def store_csv_asanawise():
	for trainer in os.listdir(folder_path):			
		trainer_path = os.path.join(folder_path,trainer)		# names of trainer
		for asana in os.listdir(trainer_path):
			asana_folder_path = os.path.join(trainer_path,asana)	# e1
			X = []
			for asana_version in os.listdir(asana_folder_path):
				asana_version_path = os.path.join(asana_folder_path,asana_version)		# v1
				frames_path = os.path.join(asana_version_path,'frames')
				key_counter = 0
				for k in os.listdir(frames_path):
					if k.endswith(".jpg") or k.endswith(".jpeg") or k.endswith(".png"):
						img_path = os.path.join(frames_path,k)
						img = cv2.imread(img_path)
						res = holis.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
						lst = []
						if res.pose_landmarks:
							for i in res.pose_landmarks.landmark:
								lst.append(i.x)
								lst.append(i.y)
								lst.append(i.z)
							# with open('reqd_angles.json','r') as f:
							# 	data = json.load(f)
							# for number,angles in data.items():
							# 	lst.append(calculateAngles(res.pose_landmarks.landmark,angles[0],angles[1],angles[2]))
							lst.append(key_counter)
							key_counter += 1
							X.append(lst)
			if len(X)==0:
				continue
			img_data_df = pd.DataFrame(X)
			col = 0
			img_data_df.columns = [str(col + 1) for col in range(100)]			# 100 or 116
			img_data_df = img_data_df.rename(columns={100: 'pose'})		# 100 or 116
			if os.path.exists(os.path.join(current_directory,'New_data','csvs',f'{asana}.csv')):
				curr_df = pd.read_csv(os.path.join(current_directory,'New_data','csvs',f'{asana}.csv'))
				updated_df = pd.concat([curr_df,img_data_df],ignore_index=True)
				print(updated_df.columns,"updated columns")
				print(curr_df.columns,"curr columns")
				print(curr_df.shape,"curr_shape")
				print(img_data_df.shape,"img shape")
				print(updated_df.shape,"updated shape")
				print(img_data_df.columns)
				# updated_df = updated_df.sample(frac=1.0, random_state=42)
				updated_df.to_csv(os.path.join(current_directory,'New_data','csvs',f'{asana}.csv'),index=False)
			else:
				img_data_df.to_csv(os.path.join(current_directory,'New_data','csvs',f'{asana}.csv'),index=False)

def store_csv_trainerwise():
	for trainer in os.listdir(folder_path):			
		trainer_path = os.path.join(folder_path,trainer)		# names of trainer
		for asana in os.listdir(trainer_path):
			asana_folder_path = os.path.join(trainer_path,asana)	# e1
			X = []
			for asana_version in os.listdir(asana_folder_path):
				asana_version_path = os.path.join(asana_folder_path,asana_version)		# v1
				frames_path = os.path.join(asana_version_path,'frames')
				key_counter = 0
				for k in os.listdir(frames_path):
					if k.endswith(".jpg") or k.endswith(".jpeg") or k.endswith(".png"):
						img_path = os.path.join(frames_path,k)
						img = cv2.imread(img_path)
						res = holis.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
						lst = []
						if res.pose_landmarks:
							for i in res.pose_landmarks.landmark:
								lst.append(i.x)
								lst.append(i.y)
								lst.append(i.z)
							# with open('reqd_angles.json','r') as f:
							# 	data = json.load(f)
							# for number,angles in data.items():
							# 	lst.append(calculateAngles(res.pose_landmarks.landmark,angles[0],angles[1],angles[2]))
							lst.append(key_counter)
							key_counter += 1
							X.append(lst)
			if len(X)==0:
				continue
			img_data_df = pd.DataFrame(X)
			img_data_df.columns = [col + 1 for col in range(100)]			# 100 or 116
			img_data_df = img_data_df.rename(columns={100: 'pose'})		# 100 or 116
			os.makedirs(os.path.join(current_directory,'New_data','csvs',f'{trainer}'),exist_ok=True)
			if os.path.exists(os.path.join(current_directory,'New_data','csvs',f'{trainer}',f'{asana}.csv')):
				curr_df = pd.read_csv(os.path.join(current_directory,'New_data','csvs',f'{trainer}',f'{asana}.csv'))
				updated_df = pd.concat([curr_df,img_data_df],ignore_index=True)
				print(curr_df.shape,"curr_shape")
				print(img_data_df.shape,"img shape")
				print(updated_df.shape,"updated shape")
				print(img_data_df.columns)
				# updated_df = updated_df.sample(frac=1.0, random_state=42)
				updated_df.to_csv(os.path.join(current_directory,'New_data','csvs',f'{trainer}',f'{asana}.csv'),index=False)
			else:
				img_data_df.to_csv(os.path.join(current_directory,'New_data','csvs',f'{trainer}',f'{asana}.csv'),index=False)
	

current_directory = os.getcwd()
# folder_path = os.path.join(current_directory, 'Asanas')
folder_path = os.path.join(current_directory, 'New_data','images')

# func()
store_csv_asanawise()
# store_csv_trainerwise()


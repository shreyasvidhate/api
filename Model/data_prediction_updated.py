import mediapipe as mp 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import cv2 
import os
import math
import json
from datetime import datetime
import time
import requests
import socket
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from Model.amendment import amendment_suggestion
from fastapi import FastAPI, Request
import json

app = FastAPI()

@app.post("/predict")
async def receive_landmarks(request: Request):
    data = await request.json()

    print("Received coordinates for prediction:")
    print(data,"from prediction")

    return {"message": "Prediction completed"}

coordinates_file_path = os.path.join(os.getcwd(),"coordinates.json")

def read_coordinates(file_path):
    with open(file_path, 'r') as file:
        coordinates = json.load(file)
    return coordinates

def monitor_coordinates():
    while True:
        if os.path.exists(coordinates_file_path):
            coordinates = read_coordinates(coordinates_file_path)
            print(coordinates)
        else:
            print("Coordinates file not found.")
        time.sleep(5)
    return coordinates

def send_output(output):
    url = 'https://api-mlkit.onrender.com/get_output'
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, json=output)
    if response.status_code == 200:
        print("Output sent successfully:", output)
    else:
        print("Failed to send output:", response.text)


def inFrame(lst):
	if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility>0.6 and lst[16].visibility>0.6:
		return True 
	return False

def number_of_poses(asana):
    n_poses = {"Tadasana":5,"Vrukshasana":7,"Utkatasana":3,"Veerbhadrasana":5,"Trikonasana":4}
    return n_poses[asana]

def asana_name(label):
    asanas = {"e1":"Tadasana","e2":"Vrukshasana","e3":"Utkatasana","e4":"Veerbhadrasana","e5":"Trikonasana"}
    return asanas[label]

def calculateAngles(landmarks,a,b,c):		
	# 11-13-15, 12-14-16, 13-11-23, 14-12-24, 23-25-27, 24-26-28, 11-23-25, 12-24-26, 7-11-13, 8-12-14, 25-27-31, 26-28-32, 13-15-17, 14-16-18, 23-24-26, 24-23-25
	vector_ab = [landmarks[b].x - landmarks[a].x, landmarks[b].y - landmarks[a].y]
	vector_bc = [landmarks[c].x - landmarks[b].x, landmarks[c].y - landmarks[b].y]
	
	dot_product = vector_ab[0] * vector_bc[0] + vector_ab[1] * vector_bc[1]
	magnitude_ab = math.sqrt(vector_ab[0] ** 2 + vector_ab[1] ** 2)
	magnitude_bc = math.sqrt(vector_bc[0] ** 2 + vector_bc[1] ** 2)

	cosine_angle = dot_product / (magnitude_ab * magnitude_bc)

	angle_rad = math.acos(cosine_angle)
	angle_deg = math.degrees(angle_rad)
	return angle_deg

current_directory = os.getcwd()
folder_path = os.path.join(current_directory, 'New_data')
models_path = os.path.join(folder_path, 'models')

holistic = mp.solutions.pose
holis = holistic.Pose()
drawing = mp.solutions.drawing_utils

frame_interval = 2
buffer_time = 10

def predict():
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = 0
    frame_extraction_count = 0

    print(fps)
        
    start_time = time.time()
    frame_extraction_time = time.time()

    X = []
    frame_data = []
    framewise_timestamps = {}
    amendment_required = False
    output = None
    results = []

    for current_model in os.listdir(models_path):
        if current_model.split('.')[-1] != "h5":
            continue
        model_path = os.path.join(os.path.join(models_path,current_model))
        print (current_model)
        model = load_model(model_path)

        start_time = time.time()
        while (time.time()-start_time)<buffer_time:
            _,frm = cap.read()
            if _:
                frm = cv2.flip(frm, 1)
                print(time.time()-start_time)
                cv2.putText(frm, f'{int((buffer_time-(time.time()-start_time))//1)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("window", frm)
            cv2.waitKey(1)


        while True:
            _,frm = cap.read()
            window = np.zeros((940,940,3), dtype="uint8")
            if _:
                frm = cv2.flip(frm, 1)
                res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
                drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS)
                frame_count += 1
                if (time.time() - frame_extraction_time) >= frame_interval:
                    print(time.time() - frame_extraction_time)

                    lst = []
                    if res.pose_landmarks:
                        # if not inFrame(res.pose_landmarks.landmark):
                        #     cv2.putText(window,"Make sure your whole body is visible!" , (180,180),cv2.FONT_ITALIC, 1.3, (0,0,255),2)
                        #     continue

                        # coordinates data
                        for i in res.pose_landmarks.landmark:
                            lst.append(i.x)
                            lst.append(i.y)
                            lst.append(i.z)

                        # angles data
                        # with open('reqd_angles.json','r') as f:
                        #     data = json.load(f)
                        # for number,angles in data.items():
                        #     lst.append(calculateAngles(res.pose_landmarks.landmark,angles[0],angles[1],angles[2]))
                        
                        frame_extraction_count += 1	

                        # frame logging
                        timestamp = datetime.now().isoformat()
                        frame_info = {
                            "frame_number": frame_extraction_count,
                            "timestamp": timestamp
                        }
                        frame_data.append(frame_info)
                        with open('frame_data.json', 'w') as json_file:
                            json.dump(frame_data, json_file)

                        # for current_model in os.listdir(models_path):
                        #     if current_model.split('.')[-1] != "h5":
                        #         continue
                        #     model_path = os.path.join(os.path.join(models_path,current_model))
                        #     print (current_model)
                        #     model = load_model(model_path)

                        lst = np.array(lst)
                        lst = lst.reshape(1, -1)
                        print(lst)
                        result = model.predict(lst)
                        print(result,"res")
                        output = np.argmax(result)
                        print(output,"op")
                        amendment_required = False
                        if result[0][output]<0.85:
                            amendment_required = True
                            message = amendment_suggestion(current_model.split('.')[0],res.pose_landmarks.landmark,output)
                        results.append(output)

                        if output==number_of_poses(asana_name(current_model.split('.')[0]))-1:
                            print("Completed "+asana_name(current_model.split('.')[0])+" asana")
                            break

                        # framewise_timestamps[output] = timestamp
                        # for pose in framewise_timestamps:
                        #     if (pose==output) or (pose<output and framewise_timestamps[pose]<timestamp) or (pose>output and framewise_timestamps[pose]>timestamp):
                        #         continue
                        #     else:
                        #         amendment_required = True
                        #         break
                            
                        # print(framewise_timestamps)
                        # if amendment_required:
                        #     print("amendment required")
                            # posture amendment call
                                
                    frame_extraction_time = time.time()
            if amendment_required:
                cv2.putText(window,message , (180,180),cv2.FONT_ITALIC, 1.3, (0,0,255),2)
                # amendment_required = False
            else:
                cv2.putText(window,asana_name(current_model.split('.')[0])+" - "+str(output)+"/"+str(number_of_poses(asana_name(current_model.split('.')[0])))+" step"+str(round(result[0][output]*100,2))+r'% correct', (180,180),cv2.FONT_ITALIC, 1.3, (0,255,0),2)
            # cv2.putText(frm, str(int((time.time()-start_time)//1))+" sec" , (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
            window[220:700, 170:810, :] = cv2.resize(frm, (640, 480))
            cv2.imshow("window", window)

            if cv2.waitKey(1)==27:
                cv2.destroyAllWindows()
                cap.release()
                break

# def main():
#     coordinates = monitor_coordinates()
#     send_output("from ml")

# if __name__ == "__main__":
#     main()
    

# predict()
# receive_landmarks()
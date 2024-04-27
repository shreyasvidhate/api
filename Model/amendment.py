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
from keras.models import load_model
from sklearn.metrics import confusion_matrix

def arms_slopes(keypoints):
    left_arm_slope1 = (keypoints['left_wrist']['y']-keypoints['left_elbow']['y'])/(keypoints['left_wrist']['x']-keypoints['left_elbow']['x'])
    left_arm_slope2 = (keypoints['left_elbow']['y']-keypoints['left_shoulder']['y'])/(keypoints['left_elbow']['x']-keypoints['left_shoulder']['x'])
    right_arm_slope1 = (keypoints['right_wrist']['y']-keypoints['right_elbow']['y'])/(keypoints['right_wrist']['x']-keypoints['right_elbow']['x'])
    right_arm_slope2 = (keypoints['right_elbow']['y']-keypoints['right_shoulder']['y'])/(keypoints['right_elbow']['x']-keypoints['right_shoulder']['x'])
    
    return [left_arm_slope1,left_arm_slope2,right_arm_slope1,right_arm_slope2]

def arm_amendments(current_keypoints, ideal_keypoints):
    current_slopes = arms_slopes(current_keypoints)
    ideal_slopes = arms_slopes(ideal_keypoints)

    if abs(current_slopes[1]-ideal_slopes[1])>0.15:
        return (False,'right','upward')
    elif abs(current_slopes[3]-ideal_slopes[3])>0.15:
        return (False,'left','upward')
    elif abs(current_slopes[0]-ideal_slopes[0])>0.25:
        return (False,'right','downward')
    elif abs(current_slopes[2]-ideal_slopes[2])>0.25:
        return (False,'left','downward')
    return (True)

def legs_slopes(keypoints):
    left_leg_slope1 = (keypoints['left_toe']['y']-keypoints['left_knee']['y'])/(keypoints['left_toe']['x']-keypoints['left_knee']['x'])
    left_leg_slope2 = (keypoints['left_knee']['y']-keypoints['left_hip']['y'])/(keypoints['left_knee']['x']-keypoints['left_hip']['x'])
    right_leg_slope1 = (keypoints['right_toe']['y']-keypoints['right_knee']['y'])/(keypoints['right_toe']['x']-keypoints['right_knee']['x'])
    right_leg_slope2 = (keypoints['right_knee']['y']-keypoints['right_hip']['y'])/(keypoints['right_knee']['x']-keypoints['right_hip']['x'])
    
    return [left_leg_slope1,left_leg_slope2,right_leg_slope1,right_leg_slope2]

def leg_amendments(current_keypoints, ideal_keypoints):
    current_slopes = legs_slopes(current_keypoints)
    ideal_slopes = legs_slopes(ideal_keypoints)

    if abs(current_slopes[1]-ideal_slopes[1])>0.15:
        return (False,'right','upward')
    elif abs(current_slopes[3]-ideal_slopes[3])>0.15:
        return (False,'left','upward')
    elif abs(current_slopes[0]-ideal_slopes[0])>0.25:
        return (False,'right','downward')
    elif abs(current_slopes[2]-ideal_slopes[2])>0.25:
        return (False,'left','downward')
    return (True)

def head_torso_slopes(keypoints):
    head_torso_slope = (keypoints['left_hip']['y']-keypoints['nose']['y'])/(keypoints['left_hip']['x']-keypoints['nose']['x'])
    return head_torso_slope

def head_amendments(current_keypoints, ideal_keypoints):
    current_slope = head_torso_slopes(current_keypoints)
    ideal_slope = head_torso_slopes(ideal_keypoints)

    if current_slope>(ideal_slope+0.15):        # conditions
        return (False)

def tadasana_amendments(current_keypoints,ideal_keypoints):
    arm_amendments_reqd = arm_amendments(current_keypoints,ideal_keypoints)

    if arm_amendments_reqd[0]==False:
        side = arm_amendments_reqd[1]
        direction = arm_amendments_reqd[2]
        message = f'Move your {side} hand {direction}!'
        return message
    
def vrukshasana_amendments(current_keypoints,ideal_keypoints,current_pose):
    arm_amendments_reqd = arm_amendments(current_keypoints,ideal_keypoints)
    leg_amendments_reqd = leg_amendments(current_keypoints,ideal_keypoints)

    if arm_amendments_reqd[0]==False:
        side = arm_amendments_reqd[1]
        direction = arm_amendments_reqd[2]
        message = f'Move your {side} hand {direction}!'

        if current_pose==4:
            message = "Touch your palms!"
        return message
    
    if leg_amendments_reqd[0]==False:
        side = arm_amendments_reqd[1]
        direction = arm_amendments_reqd[2]
        message = f'Move your {side} leg {direction}!'
        if current_pose==5:
            message = "Place your foot on the knee of other leg sideways!"

def utkataasana_amendment(current_keypoints,ideal_keypoints,current_pose):
    arm_amendments_reqd = arm_amendments(current_keypoints,ideal_keypoints)
    leg_amendments_reqd = leg_amendments(current_keypoints,ideal_keypoints)

    if current_pose>=1:
        message = f'Bend your body in your knees!'
        return message

def trikonasana_amendment(current_keypoints,ideal_keypoints):
    arm_amendments_reqd = arm_amendments(current_keypoints,ideal_keypoints)
    leg_amendments_reqd = leg_amendments(current_keypoints,ideal_keypoints)
    head_amendments_reqd = head_amendments(current_keypoints,ideal_keypoints)

    if arm_amendments_reqd[0]==False:
        side = arm_amendments_reqd[1]
        direction = arm_amendments_reqd[2]
        message = f'Move your {side} hand {direction}!'
        return message
    
    if head_amendments_reqd[0]==False:
        message = f'Bend more and touch your foot with your palm!'
    
    # leg spread condition

def veerbhadrasana_amendment(current_keypoints,ideal_keypoints,current_pose):
    arm_amendments_reqd = arm_amendments(current_keypoints,ideal_keypoints)
    leg_amendments_reqd = leg_amendments(current_keypoints,ideal_keypoints)
    head_amendments_reqd = head_amendments(current_keypoints,ideal_keypoints)

    if arm_amendments_reqd[0]==False:
        side = arm_amendments_reqd[1]
        direction = arm_amendments_reqd[2]
        message = f'Move your {side} hand {direction}!'
        return message
    
    if current_pose>1 and current_keypoints['left_ear']['visibility']>0.5 and current_keypoints['right_ear']['visibility']>0.5:
        message = f'Twist your body sideways!'
        return message

    if current_pose>3:
        message = f'Bend your body in your knees!'
        return message
    
    # leg spread condition

def amendment_suggestion(current_asana,current_landmarks,current_pose):
    current_keypoints = {
        'nose' : {'x':current_landmarks[0].x,'y':current_landmarks[0].y,'z':current_landmarks[0].z},
        'left_ear' : {'x':current_landmarks[7].x,'y':current_landmarks[7].y,'z':current_landmarks[7].z,'visibility':current_landmarks[7].visibility},
        'right_ear' : {'x':current_landmarks[8].x,'y':current_landmarks[8].y,'z':current_landmarks[8].z,'visibility':current_landmarks[8].visibility},
        'left_finger' : {'x':current_landmarks[19].x,'y':current_landmarks[19].y,'z':current_landmarks[19].z},
        'right_finger' : {'x':current_landmarks[20].x,'y':current_landmarks[20].y,'z':current_landmarks[20].z},
        'left_wrist' : {'x':current_landmarks[15].x,'y':current_landmarks[15].y,'z':current_landmarks[15].z},
        'right_wrist' : {'x':current_landmarks[16].x,'y':current_landmarks[16].y,'z':current_landmarks[16].z},
        'left_elbow' : {'x':current_landmarks[13].x,'y':current_landmarks[13].y,'z':current_landmarks[13].z},
        'right_elbow' : {'x':current_landmarks[14].x,'y':current_landmarks[14].y,'z':current_landmarks[14].z},
        'left_shoulder' : {'x':current_landmarks[11].x,'y':current_landmarks[11].y,'z':current_landmarks[11].z},
        'right_shoulder' : {'x':current_landmarks[12].x,'y':current_landmarks[12].y,'z':current_landmarks[12].z},
        'left_toe' : {'x':current_landmarks[31].x,'y':current_landmarks[31].y,'z':current_landmarks[31].z},
        'right_toe' : {'x':current_landmarks[32].x,'y':current_landmarks[32].y,'z':current_landmarks[32].z},
        'left_knee' : {'x':current_landmarks[25].x,'y':current_landmarks[25].y,'z':current_landmarks[25].z},
        'right_knee' : {'x':current_landmarks[26].x,'y':current_landmarks[26].y,'z':current_landmarks[26].z},
        'left_hip' : {'x':current_landmarks[23].x,'y':current_landmarks[23].y,'z':current_landmarks[23].z},
        'right_hip' : {'x':current_landmarks[24].x,'y':current_landmarks[24].y,'z':current_landmarks[24].z},
    }

    current_directory = os.getcwd()
    folder_path = os.path.join(current_directory, 'New_data')
    csv_folder_path = os.path.join(folder_path, 'csvs')
    df = pd.read_csv(os.path.join(csv_folder_path,f'{current_asana}.csv'))
    df = df[df['100']==current_pose]
    

    ideal_keypoints = {
        'nose' : {'x':df['1'].mean(),'y':df['2'].mean(),'z':df['3'].mean()},
        'left_ear' : {'x':df['22'].mean(),'y':df['23'].mean(),'z':df['24'].mean()},
        'right_ear' : {'x':df['25'].mean(),'y':df['26'].mean(),'z':df['27'].mean()},
        'left_finger' : {'x':df['58'].mean(),'y':df['59'].mean(),'z':df['60'].mean()},
        'right_finger' : {'x':df['61'].mean(),'y':df['62'].mean(),'z':df['63'].mean()},
        'left_wrist' : {'x':df['46'].mean(),'y':df['47'].mean(),'z':df['48'].mean()},
        'right_wrist' : {'x':df['49'].mean(),'y':df['50'].mean(),'z':df['51'].mean()},
        'left_elbow' : {'x':df['40'].mean(),'y':df['41'].mean(),'z':df['42'].mean()},
        'right_elbow' : {'x':df['43'].mean(),'y':df['44'].mean(),'z':df['45'].mean()},
        'left_shoulder' : {'x':df['34'].mean(),'y':df['35'].mean(),'z':df['36'].mean()},
        'right_shoulder' : {'x':df['37'].mean(),'y':df['38'].mean(),'z':df['39'].mean()},
        'left_toe' : {'x':df['94'].mean(),'y':df['95'].mean(),'z':df['96'].mean()},
        'right_toe' : {'x':df['97'].mean(),'y':df['98'].mean(),'z':df['99'].mean()},
        'left_knee' : {'x':df['76'].mean(),'y':df['77'].mean(),'z':df['78'].mean()},
        'right_knee' : {'x':df['79'].mean(),'y':df['80'].mean(),'z':df['81'].mean()},
        'left_hip' : {'x':df['70'].mean(),'y':df['71'].mean(),'z':df['72'].mean()},
        'right_hip' : {'x':df['73'].mean(),'y':df['74'].mean(),'z':df['75'].mean()},
    }

    if current_asana=='e1':
        message = tadasana_amendments(current_keypoints,ideal_keypoints)
    elif current_asana=='e2':
        message = vrukshasana_amendments(current_keypoints,ideal_keypoints,current_pose)
    elif current_asana=='e3':
        message = utkataasana_amendment(current_keypoints,ideal_keypoints,current_pose)
    elif current_asana=='e4':
        message = veerbhadrasana_amendment(current_keypoints,ideal_keypoints,current_pose)
    elif current_asana=='e5':
        message = trikonasana_amendment(current_keypoints,ideal_keypoints)

    return message


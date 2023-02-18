#size = 1/distance.
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import noduro
import time
from csv import writer
import pandas as pd
import noduro_code.read_settings as read_settings
import noduro
import os
try:
    SCALE = read_settings.get_scale_factor()
except:
    print("no scale")

DEFAULT_FILE,_ = read_settings.get_settings()
DEFAULT_FILE = DEFAULT_FILE["filesystem"]["pose_standardization"] #in settings
DICT = noduro.read_json(DEFAULT_FILE, True)
def convert_holistic_to_dict(holistic_values):
    return {"face" : holistic_values.face_landmarks, "pose" : holistic_values.pose_landmarks, "left_hand" : holistic_values.left_hand_landmarks, "right_hand" : holistic_values.right_hand_landmarks}

def iter_landmarks(landmark_list,feature_dict):
        new = []
        for index, (key, value) in enumerate(feature_dict.items()):
            #check if dict landmark_list has the key, fill it with sklearn imputers
            if landmark_list is None:
                value = [(np.nan,np.nan,np.nan) for v in value]
            else:
                value = [(landmark_list[val].x,landmark_list[val].y,landmark_list[val].z) for val in value]
            new.extend(value)
        return new

def filter_body_part(landmarks, ref_dict : dict):
    if landmarks is not None:
        landmarks = landmarks.landmark
    val = iter_landmarks(landmarks,ref_dict)
    return val

def filter_body_parts(landmarks : dict, ref : dict):
    ret = []
    for key, value in ref.items():
        ret.append(filter_body_part(landmarks[key],value))
    return ret

def convert_holistic_to_parts(landmarks,ref_dict):
    return filter_body_parts(convert_holistic_to_dict(landmarks), ref_dict)

def display_pose_direct(points,gesture_point_dict):
    fig = plt.figure()
    points = convert_holistic_to_parts(points, gesture_point_dict)
    ax = plt.axes(projection='3d')
    for point in points:
        point = np.array(point)
        ax.scatter3D(point[:,0],point[:,2]-1,point[:,1]*-1)
    ax.set_xlim([0,1])
    ax.set_ylim([-1,0])
    ax.set_zlim([-1,0])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

def distance(point1,point2):
    dist = point2-point1
    return np.sqrt(dist[0]**2 + dist[1]**2 + dist[2]**2)

def derive_calibration(values):
    # file = np.array(pd.read_csv(file, header = None))
    ret = [np.min(values), np.mean(values), np.max(values), np.std(values), len(values)]
    return ret
def calibrate_pose_using_eye_and_chest(points): #use the eye for scaling, and the chest as a centering value
    points["face"] = points["face"]
    left = DICT["scalar"]["face"]["left_iris"]; left = [(points["face"].landmark[l].x,points["face"].landmark[l].y,points["face"].landmark[l].z) for l in left]
    right =  DICT["scalar"]["face"]["right_iris"]; right = [(points["face"].landmark[r].x,points["face"].landmark[r].y,points["face"].landmark[r].z) for r in right]
    left = np.mean(left, axis = 0)
    right = np.mean(right, axis = 0)
    dist = distance(left,right)

    chest = DICT["center"]["pose"]["chest"]; center = np.mean([(points["pose"].landmark[c].x,points["pose"].landmark[c].y,points["pose"].landmark[c].z) for c in chest],axis = 0)
    return dist/SCALE,center #return ratio between size of current image and SCALE value. 
def center_and_scale_from_raw(points, gpdict): #derived point values and the gesture point dictionary
    curr_scale,center = calibrate_pose_using_eye_and_chest(points)
    ret = []
    s = time.time()
    for key, value in gpdict.items():
        value = [x for v in value.values() for x in v]
        if points[key] == None:
            value = [(np.nan,np.nan,np.nan) for v in value]
        else:
            value = [((points[key].landmark[val].x,points[key].landmark[val].y,points[key].landmark[val].z)-center)/curr_scale for val in value]
        ret.append(value)
    return ret
def flatten_gesture_point_dict_to_list(gpdict):
    ret = []
    for key, value in gpdict.items():
        value = [x for v in value.values() for x in v]
        ret.append(value)
    return ret
def flatten_3d_to_1d(points : list):
    return np.array([x for v in points for x in v]).flatten()
# def standardize_pose(points):

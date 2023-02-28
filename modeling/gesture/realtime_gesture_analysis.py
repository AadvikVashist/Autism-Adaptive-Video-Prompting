# import necessary packages
import time
import csv
import numpy as np
import os
import pandas as pd
#for training
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle 
import cv2
from gesture_tracker import gesture_tracker
import noduro_code.read_settings as read_settings
import noduro
from sklearn.impute import SimpleImputer
import modeling.gesture.pose_manipulation.pose_standardizer as pose_standardizer

"""the realtime_gesture_analysis class is a subclass of gesture tracker.
The class uses the noduro gesture models trainined in gesture_ingestion to analyze gestures. 
The class can take any of the four existing models, at any point in their trained time, and can apply analysis based on the frame_skip value. 
The frameskip value can be trained using frame_skip.py in the noduro_code folder."""
class realtime_gesture_analysis(gesture_tracker):
    def __init__(self,use_newest : str = None):
        super().__init__(eye = True, face = True, hand = True, pose = True, eye_confidence = 0.7, face_confidence= 0.7, hand_confidence = 0.7, pose_confidence = 0.7,number_of_hands = 2,  frameskip = True)
        self.settings, _ = read_settings.get_settings()
        self.default_folder = self.settings["filesystem"]["model_storage"] #in settings
        self.default_folder = noduro.subdir_path(self.default_folder)
        self.models = sorted(os.listdir(self.default_folder))
        self.gpd = pose_standardizer.flatten_gesture_point_dict_to_list(self.gesture_point_dict)
        if use_newest is None:
            selected_model = noduro.get_dir_files(noduro.join(self.default_folder,self.models[-1]))
        else:
            try:
                ind = self.models.index(use_newest)
                selected_model = noduro.get_dir_files(noduro.join(self.default_folder,self.models[ind]))
            except:
                raise ValueError("folder not found in the models. Please try again.")
        self.selected_model = {
            "lr" : [pickle.load(open(i,"rb")) for i in selected_model if "lr.pkl" in i][0],
            "rc" : [pickle.load(open(i,"rb")) for i in selected_model if "rc.pkl" in i][0],
            "rf" : [pickle.load(open(i,"rb")) for i in selected_model if "rf.pkl" in i][0],
            "gb" : [pickle.load(open(i,"rb")) for i in selected_model if "gb.pkl" in i][0]
        }


    def realtime_analysis(self, capture_index : int = 0, save_vid_file  : str = None, save_results_vid_file : str = None, frame_skip = None):
        if capture_index == None:
            capture_index = self.camera_selector() #select camera
        self.capture = cv2.VideoCapture(capture_index, cv2.CAP_DSHOW) #cap_show makes startup alot faster. Starts camera
        first_frame = True  
        landmarks = None
        self.looping_analysis(videoCapture = self.capture, video_shape = None, fps = None, result_vid = save_results_vid_file, starting_vid = save_vid_file, frame_skip = frame_skip, save_pose = False, standardize_pose = True,save_frames = False)    

    
    def while_processing(self, frame):
        stand, distance = pose_standardizer.center_and_scale_from_raw(pose_standardizer.convert_holistic_to_dict(self.processed_frame["holistic"]), self.gesture_point_dict,self.moving_average)
        row = pose_standardizer.fill_nans_with_imputer_for_sklearn_regression(stand, False)
        gest = None
        proba = None
        for name, model in self.selected_model.items():
            body_language_class = model.predict(row)[0]
            try:
                body_language_prob = model.predict_proba(row)[0]
                if proba is None:
                    proba = body_language_prob
                    gest = list(np.zeros(proba.shape))
                else:
                    proba += body_language_prob
                gest[np.argmax(proba)] = body_language_class
            except:
                pass
            # if g[0] not in gest:
            #     gest[g[0]] = g
            # else:
            #     gest[g[0]][1] += g[1]
        self.etc["gesture"] = [gest[np.argmax(proba)],np.max(proba)]  # gest[list(gest.keys())[np.argmax(np.array(list(gest.values()))[:,1] )]]
        return frame
a = realtime_gesture_analysis()
a.realtime_analysis()
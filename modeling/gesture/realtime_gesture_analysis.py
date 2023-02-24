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
class realtime_gesture_analysis(gesture_tracker):
    def __init__(self):
        super().__init__(eye = True, face = True, hand = True, pose = True, eye_confidence = 0.7, face_confidence= 0.7, hand_confidence = 0.7, pose_confidence = 0.7,number_of_hands = 2,  frameskip = True)
        self.settings, _ = read_settings.get_settings()
        self.default_folder = self.settings["filesystem"]["model_storage"] #in settings
        self.default_folder = noduro.subdir_path(self.default_folder)
        self.models = sorted(os.listdir(self.default_folder))
        self.gpd = pose_standardizer.flatten_gesture_point_dict_to_list(self.gesture_point_dict)
        newest_model_files = noduro.get_dir_files(noduro.join(self.default_folder,self.models[-1]))
        self.newest_models = {
            "lr" : [pickle.load(open(i,"rb")) for i in newest_model_files if "lr.pkl" in i][0],
            "rc" : [pickle.load(open(i,"rb")) for i in newest_model_files if "rc.pkl" in i][0],
            "rf" : [pickle.load(open(i,"rb")) for i in newest_model_files if "rf.pkl" in i][0],
            "gb" : [pickle.load(open(i,"rb")) for i in newest_model_files if "gb.pkl" in i][0]
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
        for name, model in self.newest_models.items():
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
    # def existing_read(self, classification : str, video_file):
    #     result_video_file = os.path.splitext(video_file); results_csv = result_video_file[0] + ".csv"
    #     result_video_file[0] += "_results"; result_video_file = ''.join(result_video_file)#remove file extension and add results to the end
    #     csv_data = self.video_analysis(video = video_file, 
    #                                                     result_video = result_video_file,
    #                                                     classification = classification)
    #     self.write_csv(results_csv,csv_data, self.number_of_coordinates)
    # def frame_by_frame_check(self, frame, row, trys : bool = True):
    #     X = pd.DataFrame([row])
    #     for model in self.models[1::]:
    #         model = model[0]
    #         if trys:
    #             body_language_class = model.predict(X)[0]
    #             try:
    #                 body_language_prob = model.predict_proba(X)[0]
    #                 print(body_language_class, body_language_prob)
    #             except:
    #                 print(body_language_class)
    #             try:
    #                 cv2.putText(frame, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    #                 cv2.putText(frame, body_language_class.split(' ')[0], (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
    #                 # Display Probability
    #                 cv2.putText(frame, 'PROB', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    #                 cv2.putText(frame, str(round(body_language_prob[np.argmax(body_language_prob)],2)), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    #                 return frame
    #             except:
    #                 pass
    #         else:
    #             body_language_class = model.predict(X)[0]
    #             body_language_prob = model.predict_proba(X)[0]
    #             print(body_language_class, body_language_prob)
    #             cv2.putText(frame, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    #             cv2.putText(frame, body_language_class.split(' ')[0], (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
    #             # Display Probability
    #             cv2.putText(frame, 'PROB', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    #             cv2.putText(frame, str(round(body_language_prob[np.argmax(body_language_prob)],2)), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    #             return model
a = realtime_gesture_analysis()
# a.get_pickle_files("C:/Users/aadvi/Desktop/Tester")
# a.live_read()
a.realtime_analysis()
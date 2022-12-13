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
class realtime_usage:
    def __init__(self):
        x =0
    def get_pickle_files(self, base_dir):
        everything = [os.path.join(dp, f) for dp, dn, fn in os.walk(base_dir) for f in fn]
        self.lr_model = [i for i in everything if "_lr.pkl" in i]
        self.rc_model = [i for i in everything if "_rc.pkl" in i]
        self.rf_model = [i for i in everything if "_rf.pkl" in i]
        self.gn_model = [i for i in everything if "_gb.pkl" in i]
        self.models = [self.lr_model,self.rc_model,self.rf_model,self.gn_model]
    
    
    def live_read(self, classification : str, save_vid_file  : str, capture_index : int = 0):
            save_results_file = list(os.path.splitext(save_vid_file)); results_csv = save_results_file[0] + ".csv"
            save_results_file[0] += "_results"; save_results_file = ''.join(save_results_file)#remove file extension and add results to the end
            csv_data = self.gesture_model.realtime_analysis(capture_index = capture_index,
                                                    save_vid_file = save_vid_file,
                                                    save_results_vid_file = save_results_file, classification = classification)
            self.write_csv(results_csv,csv_data, self.gesture_model.number_of_coordinates)
            return results_csv
    
    
    def existing_read(self, classification : str, video_file):
        result_video_file = os.path.splitext(video_file); results_csv = result_video_file[0] + ".csv"
        result_video_file[0] += "_results"; result_video_file = ''.join(result_video_file)#remove file extension and add results to the end
        csv_data = self.gesture_model.video_analysis(video = video_file, 
                                                        result_video = result_video_file,
                                                        classification = classification)
        self.write_csv(results_csv,csv_data, self.gesture_model.number_of_coordinates)
    def frame_by_frame_check(self, frame, row, trys : bool = True):
        X = pd.DataFrame([row])
        for model in self.models[0:1]:
            if trys:
                try:
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]
                    print(body_language_class, body_language_prob)
                    cv2.putText(frame, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(frame, body_language_class.split(' ')[0], (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Display Probability
                    cv2.putText(frame, 'PROB', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(frame, str(round(body_language_prob[np.argmax(body_language_prob)],2)), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    return model
                except:
                    pass
            else:
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
                print(body_language_class, body_language_prob)
                cv2.putText(frame, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, body_language_class.split(' ')[0], (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display Probability
                cv2.putText(frame, 'PROB', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, str(round(body_language_prob[np.argmax(body_language_prob)],2)), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                return model

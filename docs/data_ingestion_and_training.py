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
from gesture_tracker import gesture_tracker
# initialize mediapipe
class data_ingestion:
    def __init__(self):
        self.gesture_model = gesture_tracker(True, True, True, 0.7, 0.7, 0.7, 2)

    
    def write_csv(self, filename, rows, num_coords):                        
        with open(filename, mode='w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            landmarks = ['class']
            for val in range(1, num_coords+1):
                landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
            csv_writer.writerow(landmarks)
            for row in rows:
                csv_writer.writerow(row)
    
    
    def live_train(self, classification : str, save_vid_file  : str, capture_index : int = 0):
        save_results_file = list(os.path.splitext(save_vid_file)); results_csv = save_results_file[0] + ".csv"
        save_results_file[0] += "_results"; save_results_file = ''.join(save_results_file)#remove file extension and add results to the end
        csv_data = self.gesture_model.realtime_analysis(capture_index = capture_index,
                                                save_vid_file = save_vid_file,
                                                save_results_vid_file = save_results_file, classification = classification)
        self.write_csv(results_csv,csv_data, self.gesture_model.number_of_coordinates)
        return results_csv
    
    def existing_training(self, classification : str, video_file):
        result_video_file = os.path.splitext(video_file); results_csv = result_video_file[0] + ".csv"
        result_video_file[0] += "_results"; result_video_file = ''.join(result_video_file)#remove file extension and add results to the end
        csv_data = self.gesture_model.video_analysis(video = video_file, 
                                                        result_video = result_video_file,
                                                        classification = classification)
        self.write_csv(results_csv,csv_data, self.gesture_model.number_of_coordinates)
    
    
    def read_collected_data(self, filename, classification):
        df = pd.read_csv(filename)
        df.head()
        df.tail()
        X = df.drop('class', axis=1) # features
        y = df['class'] # target value
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
        return X_train, X_test, y_train, y_test
    
    
    def model_pipeline(self,classification, input_csv_file,output_csv_file = None):
        if output_csv_file is None:
            output_csv_file = list(os.path.splitext(input_csv_file)); results_csv = input_csv_file[0] + "_analyzed.csv"
        pipelines = {
            'lr':make_pipeline(StandardScaler(), LogisticRegression()),
            'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
            'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
            'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
        }
        fit_models = {}
        X_train, X_test, y_train, y_test = self.read_collected_data(input_csv_file, classification)
        for algo, pipeline in pipelines.items():
            model = pipeline.fit(X_train, y_train)
            fit_models[algo] = model
        fit_models['rc'].predict(X_test)
        
        #stored predictions
        for algo, model in fit_models.items():
            yhat = model.predict(X_test)
            print(algo, accuracy_score(y_test, yhat)) #predicted algorithm accuracy_score
        
        #store predicted values
        with open((output_csv_file[0] + '_lr.pkl'), 'wb') as f: #.pkl file
            pickle.dump(fit_models['lr'], f)
        with open((output_csv_file[0] + '_rc.pkl'), 'wb') as f: #.pkl file
            pickle.dump(fit_models['rc'], f)
        with open((output_csv_file[0] + '_rf.pkl'), 'wb') as f: #.pkl file
            pickle.dump(fit_models['rf'], f)
        with open((output_csv_file[0] + '_gb.pkl'), 'wb') as f: #.pkl file
            pickle.dump(fit_models['gb'], f)
a = data_ingestion()
# results_csv = a.live_train("ymca", "C:/Users/aadvi/Desktop/Testing/video1.mp4",0)
a.model_pipeline("ymca", "C:/Users/aadvi/Desktop/Testing/video1.csv")
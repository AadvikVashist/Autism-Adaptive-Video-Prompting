# import necessary packages
import csv
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
import init
from datetime import datetime
import cv2
# initialize mediapipe
class data_ingestion:
    def __init__(self, repo_writeable : bool = False):
        self.gesture_model = gesture_tracker(True, True, True, 0.7, 0.7, 0.7, 2)
        self.repo_writeable = repo_writeable
    def make_training_set(self, master_model : bool = False):
        folder_name = init.folder_selector()
        results = []
        index = 0
        time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        
        while True:
            file_name = os.path.join(folder_name,  "capture_") + time + ".mp4"
            results_file_name = os.path.join(folder_name,  "capture_") + time + ".csv"
            classification = input("what do you want to call this classifier: ")
            classification = classification.lower().strip()
            results_csv_file,csv_data = self.live_train(classification, file_name,0)
            results.extend(csv_data)
            check = input("would you like to make another classifier?")
            if "y" in check.lower():
                time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                continue
            else:
                break
        self.write_csv(results_file_name,results, self.gesture_model.number_of_coordinates)
        self.model_pipeline(results_file_name, os.path.join(folder_name, "results"),master_model)
    def write_csv(self, filename, rows, num_coords):                        
        with open(filename, mode='w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            landmarks = ['class']
            for val in range(1, num_coords+1):
                landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
            csv_writer.writerow(landmarks)
            for row in rows:
                csv_writer.writerow(row)
    
    
    def live_train(self, classification : str, save_vid_file  : str, capture_index : int = 1):
        capture = cv2.VideoCapture(capture_index, cv2.CAP_DSHOW)
        while True:
            _, frame = capture.read()
            cv2.imshow("type q once framing is right", cv2.flip(frame,1))
            if cv2.waitKey(1) == ord('q'):
                capture.release()
                cv2.destroyAllWindows()
                break
        save_results_file = list(os.path.splitext(save_vid_file)); results_csv = save_results_file[0] + ".csv"
        save_results_file[0] += "_results"; save_results_file = ''.join(save_results_file)#remove file extension and add results to the end
        csv_data, elapsed_time = self.gesture_model.realtime_analysis(capture_index = capture_index,
                                                save_vid_file = save_vid_file,
                                                save_results_vid_file = save_results_file, classification = classification)
        self.write_csv(results_csv,csv_data, self.gesture_model.number_of_coordinates)
        return results_csv,csv_data
    
    def existing_training(self, classification : str, video_file):
        result_video_file = os.path.splitext(video_file); results_csv = result_video_file[0] + ".csv"
        result_video_file[0] += "_results"; result_video_file = ''.join(result_video_file)#remove file extension and add results to the end
        csv_data = self.gesture_model.video_analysis(video = video_file, 
                                                        result_video = result_video_file,
                                                        classification = classification)
        self.write_csv(results_csv,csv_data, self.gesture_model.number_of_coordinates)
    
    
    def read_collected_data(self, filename):
        df = pd.read_csv(filename)
        df.head()
        df.tail()
        X = df.drop('class', axis=1) # features
        y = df['class'] # target value
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
        return X_train, X_test, y_train, y_test
    
    
    def model_pipeline(self, input_csv_file, output_file_base: str = None, master_model : bool = False):
        if output_file_base is None:
            output_file_base = list(os.path.splitext(input_csv_file))
            output_file_base = output_file_base[0]
        pipelines = {
            'lr':make_pipeline(StandardScaler(), LogisticRegression()),
            'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
            'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
            'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
        }
        fit_models = {}
        X_train, X_test, y_train, y_test = self.read_collected_data(input_csv_file)
        for algo, pipeline in pipelines.items():
            model = pipeline.fit(X_train, y_train)
            fit_models[algo] = model
        fit_models['rc'].predict(X_test)
        
        #stored predictions
        for algo, model in fit_models.items():
            yhat = model.predict(X_test)
            print(algo, accuracy_score(y_test, yhat)) #predicted algorithm accuracy_score
        
        #store predicted values
        with open((output_file_base + '_lr.pkl'), 'wb') as f: #.pkl file
            pickle.dump(fit_models['lr'], f)
        with open((output_file_base + '_rc.pkl'), 'wb') as f: #.pkl file
            pickle.dump(fit_models['rc'], f)
        with open((output_file_base + '_rf.pkl'), 'wb') as f: #.pkl file
            pickle.dump(fit_models['rf'], f)
        with open((output_file_base + '_gb.pkl'), 'wb') as f: #.pkl file
            pickle.dump(fit_models['gb'], f) 
        if self.repo_writeable and master_model:
            output_file_base = os.path.join(os.getcwd(), "model", "model_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            with open((output_file_base + '_lr.pkl'), 'wb') as f: #.pkl file
                pickle.dump(fit_models['lr'], f)
            with open((output_file_base + '_rc.pkl'), 'wb') as f: #.pkl file
                pickle.dump(fit_models['rc'], f)
            with open((output_file_base + '_rf.pkl'), 'wb') as f: #.pkl file
                pickle.dump(fit_models['rf'], f)
            with open((output_file_base + '_gb.pkl'), 'wb') as f: #.pkl file
                pickle.dump(fit_models['gb'], f) 
        elif self.repo_writeable:
            raise ValueError ("This object does not have write access to master. Please make sure to set the __init__ value is_writeable to True. ")
a = data_ingestion(True)
# results_csv = a.make_training_set(True)
a.model_pipeline("C:/Users/aadvi/Desktop/Tester/capture_2022-12-15-17-30-14.csv", master_model =True)
# a.model_pipeline("high five", 'C:/Users/aadvi/Desktop/Tester\\capture_2022-12-13-16-40-11.csv')
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
from modeling.gesture.gesture_tracker import gesture_tracker
import noduro
from datetime import datetime
import cv2
import numpy as np
from  fuzzywuzzy import fuzz as fz
import shutil
import time
# initialize mediapipe
from sklearn.impute import SimpleImputer
import modeling.gesture.pose_manipulation.pose_standardizer as pose_standardizer
import noduro_code.read_settings as read_settings
DEFAULT_FOLDER,_ = read_settings.get_settings()
DEFAULT_FOLDER = DEFAULT_FOLDER["filesystem"]["gesture_paths"] #in settings
DEFAULT_FOLDER = noduro.subdir_path(DEFAULT_FOLDER)
class gesture_data_ingestion(gesture_tracker):
    def __init__(self,):
        super().__init__(eye = True, face = True, hand = True, pose = True, eye_confidence = 0.7, face_confidence= 0.7, hand_confidence = 0.7, pose_confidence = 0.7,number_of_hands = 2,  frameskip = True)
        self.gestures = os.listdir(DEFAULT_FOLDER)
        self.gpd = pose_standardizer.flatten_gesture_point_dict_to_list(self.gesture_point_dict)
    def compare_input_and_folders(self, options : str):
        while True:
                inp = input("found a folder in directory with similar same name called '" + options + "', is this the right one? if cancel, type 'c' ") #input to check if value is correct
                if inp.strip().lower() == "c":
                    return None
                try:
                    inp = noduro.check_boolean_input(inp)
                    print("registered", inp)
                    break
                except:
                    print("try again please")
                    pass
        return inp
    def check_if_gesture_exists(self, gesture_name : str) -> str:
        if any([gesture_name.lower().strip() == a.lower().strip() for a in self.gestures]): #if there is a direct match
            string = [a for a in self.gestures if gesture_name.lower().strip() == a.lower().strip()][0]
            return noduro.join(DEFAULT_FOLDER,string)
        if any([True if fz.ratio(gesture_name.lower().strip(),a.lower().strip()) > 70 else False for a in self.gestures]): #check if 70% chance or greater
            string = [fz.ratio(gesture_name.lower().strip(),a.lower().strip()) for a in self.gestures] # get the value
            a = np.argsort(string) # get the indexes from least to greatest based on match percentages
            for index in range(len(a)-1,len(a)-4,-1): #check the top 3 options
                string = self.gestures[a[index]] #current string option
                bools = self.compare_input_and_folders(string) # check whether the user confirms
                if bools is True:
                    return noduro.join(DEFAULT_FOLDER,string)
        return None
    
    def fill_nans_with_imputer_for_sklearn_regression(self, list_2d : list) -> list:
        #check if the class has a variable named self.imputer
        if not hasattr(self, 'imputer'):
            self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        list_1d = [pose_standardizer.flatten_3d_to_1d(d) for d in list_2d]
        return self.imputer.fit_transform(list_1d)
    
    def save_points_as_csv(self, file : str, points : list) -> None:
        val = 0
        keys = list(self.gesture_point_dict.keys())
        points = list(points)
        if not os.path.exists(file):
            #iterate over self.gpd, formatting each index as key + "_" + index + "_" + X/Y/Z
            row = [keys[index] + "_" + str(v) + "_" + alpha for index,value in enumerate(self.gpd) for v in value for alpha in ["x","y","z"]]
            points.insert(0,row)
        
        noduro.write_csv(file,points,False,False,True)
    
    def intitialize_current_gesture_set(self, inputted_value : str = None, existing_files : bool = None): #take a single gesture string, analyze the directory, and train get the csv's for those videos. 
        if inputted_value is None:
            inputted_value = input("what is the name of the gesture you want to train? ")
        dir = self.check_if_gesture_exists(inputted_value)
        if dir is None: #if the directory doesn't exist
            noduro.make_dir_if_not_exist(noduro.join(DEFAULT_FOLDER,inputted_value))
        self.current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        cwd = noduro.join(DEFAULT_FOLDER,inputted_value, self.current_time)
        noduro.make_dir_if_not_exist(cwd)
        
        if existing_files:
            folder_path = noduro.folder_selector()
            folder = os.listdir(folder_path)
            for f in folder:
                send_folder = noduro.join(folder_path,f)
                recieve_folder = noduro.join(cwd,f)
                shutil.move(send_folder,recieve_folder)
                print("moved", send_folder, "to", recieve_folder)
        elif existing_files == False:
            self.train_multiple_videos_of_same_gesture_using_camera(save_vid_folder = cwd, certain = False)
        if existing_files is not None:
            self.train_gesture_using_folder_videos_recurse(cwd)

    def convert_video_to_csv_files(self, file : str, result_file_names : str) -> None:   
        self.video_analysis(file,None,None, 1, standardize_pose = True) #analyze the video and get a list of the results
        #fill nanes with imputer for self.save_pose and self.save_calibrated_pose
        self.save_pose = self.fill_nans_with_imputer_for_sklearn_regression(self.save_pose)
        self.save_calibrated_pose = self.fill_nans_with_imputer_for_sklearn_regression(self.save_calibrated_pose)
        self.save_points_as_csv(result_file_names[0],self.save_pose) # write raw results
        self.save_points_as_csv(result_file_names[1],self.save_calibrated_pose)#write actual results
    
    def train_gesture_using_folder_videos_recurse(self, folder_path) -> None:
        folder_files = noduro.get_dir_files(folder_path, False,False)
        for file in folder_files:
            f, ext= os.path.splitext(file)
            results = [f + "_results.csv",f + "_standardized_results.csv"]
            if ext in [".mp4", ".m4a"] and all(results not in folder_files):
                self.convert_video_to_csv_files(self, file,results)
    # def make_training_set(self, master_model : bool = False):
    #     folder_name = noduro.folder_selector()
    #     results = []
        
    #     while True:
    #         file_name = noduro.join(folder_name,  "capture_") + time + ".mp4"
    #         results_file_name = os.path.join(folder_name,  "capture_") + time + ".csv"
    #         classification = input("what do you want to call this classifier: ")
    #         classification = classification.lower().strip()
    #         results_csv_file,csv_data = self.train_using_camera(classification, file_name,0)
    #         results.extend(csv_data)
    #         check = input("would you like to make another classifier?")
    #         if "y" in check.lower():
    #             time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    #             continue
    #         else:
    #             break
    #     self.write_csv(results_file_name,results, self.gesture_model.number_of_coordinates)
    #     self.model_pipeline(results_file_name, os.path.join(folder_name, "results"),master_model)

    def train_multiple_videos_of_same_gesture_using_camera(self,save_vid_folder : str, capture_index : int = 0, certain : bool = False):
        index = 0
        while True:
            self.train_using_camera(save_vid_folder, capture_index, certain)
            print(f"trained {index} video succesfuly")
            index+=1
            while True:
                try: 
                    train_again = noduro.check_boolean_input(input("would you like to train one more video?"))
                    break
                except:
                    print("please try again")
            if train_again: #user decided to not train another model. 
                continue
            else:
                break
        print(f"trained a total of {index} videos for gesture")
    def train_using_camera(self, save_vid_folder : str, capture_index : int = 0, certain : bool = False):
        if certain is False:
            while True: #check if user wants to livetrain
                inp = input("would you like to livetrain using camera? if cancel, type 'c' ") #input to check if value is correct
                if inp.strip().lower() == "c":
                    return None
                try:
                    inp = noduro.check_boolean_input(inp)
                    print("registered", inp)
                    break
                except:
                    print("try again please")
                    pass
        
        capture = cv2.VideoCapture(capture_index, cv2.CAP_DSHOW)
        start = time.time()
        showTime = True
        font = cv2.FONT_HERSHEY_SIMPLEX
        text1 = "type q once you have the framing"
        text2 = "you have 5 seconds to get ready"
        # get boundary of this text
        text1size = cv2.getTextSize(text1, font, 1, 2)[0]
        text2size = cv2.getTextSize(text2, font, 1, 2)[0]


        while True: #get the framing of the video right so that data is accurate
            _, frame = capture.read()
            frame = cv2.flip(frame,1)
            if showTime:
                start = time.time()
                text1X = int((frame.shape[1] - text1size[0]) / 2)
                text2X = int((frame.shape[1] - text2size[0]) / 2)
                cv2.putText(frame, text1, (text1X,int(frame.shape[1]/15)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                cv2.putText(frame, text2, (text2X ,int(frame.shape[1]/15+text1size[1]*1.25)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
            else:
                #use cv2.puttext to display text in frame with number of integer seconds left using time.time() - start. 
                tim = str(int(5 - (time.time() - start)))
                textsize = cv2.getTextSize(tim, font, 15, 25)[0]
                textX = int((frame.shape[1] - textsize[0]) / 2)
                textY = int((frame.shape[0] + textsize[1]) / 2)
                cv2.putText(frame, tim, (textX, textY ), cv2.FONT_HERSHEY_SIMPLEX, 15, (255,255,255), 25)
            if cv2.waitKey(1) == ord('q'):
                showTime = False
            if time.time() - start >= 5:
                break
            cv2.imshow("type q once framing is right, you will get 5 seconds before it cancels", frame)
        capture.release()
        cv2.destroyAllWindows()
        
        
        file_name = noduro.join(save_vid_folder + "capture_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".mp4")
        f, ext= os.path.splitext(file_name)
        results = [f + "_results.csv",f + "_standardized_results.csv"]
        self.realtime_analysis(capture_index = capture_index,
                                                save_vid_file = file_name,
                                                save_results_vid_file = None,
                                                frame_skip=0)
        self.convert_video_to_csv_files(file_name,results)

    def sort_existing_gesture_data(self, folder : str = None) -> dict: #use noduro.get_dir_files to get all files in folder. If folder isn't provided use DEFAULT_FOLDER. Put all of the files into their own key in a dictionary, where the ones with the same root folder are together.
        if folder is None:
            folder = DEFAULT_FOLDER
        files = noduro.get_dir_files(folder)
        data = {}
        for file in files:
            f, ext = os.path.splitext(file)
            gest = noduro.join(os.path.relpath(f,folder)).split('/')[0]
            if ext == ".csv":
                if gest in data:
                    data[gest].append(file)
                else:
                    data[gest] = [file]
        return data

    def read_existing_gesture_data(self, folder : str = None) -> tuple:
        if folder is None:
            folder = DEFAULT_FOLDER
        data = self.sort_existing_gesture_data(folder)
        ret = {}; point_names = None
        for key in data:
            k = []
            for file in data[key]:
                csv_data = noduro.read_csv(file,False)
                k.extend(np.array(csv_data[1::],dtype = np.float16))
                point_names = csv_data[0]
            ret[key] = k
        ret = self.add_class_to_data(ret)
        return ret, point_names
    
    def add_class_to_data(self, data : dict): #insert a class column at index 0in  all of the lists of data in the dictionary key with the class being called the key 
        for k in data:
            datum = np.array(data[k], dtype = object)
            datum = np.insert(datum,0,[k for i in range(datum.shape[0])], axis = 1)
            data[k] = datum
        return data
        
    def seed_data_from_dict(self, data : dict):
        datum = np.array([x for v in data.values() for x in v])
        X = datum[:,0]
        Y = datum[:,1::]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)
        return X_train, X_test, y_train, y_test
   
    def model_pipeline(self, input_folder : str = None, output_file_base: str = None): # , master_model : bool = False
        if input_folder is None:
            input_folder = DEFAULT_FOLDER
        if output_file_base is None:
            output_file_base = DEFAULT_FOLDER + "_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        pipelines = {
            'lr':make_pipeline(StandardScaler(), LogisticRegression()),
            'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
            'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
            'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
        }
        fit_models = {}
        data, display = self.read_existing_gesture_data(input_folder)
        X_train, X_test, y_train, y_test = self.seed_data_from_dict(data)
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
if __name__ == "__main__":
    #make gesture data ingestion object and retrain the existing data using DEFAULT_FOLDER
    a = gesture_data_ingestion()
    a.model_pipeline()
# a = gesture_data_ingestion()
# a.intitialize_current_gesture_set("cutting", existing_files = False)
# a.save_points_as_csv("xyz.csv", [[1,2,2,3,4,1,41,2,2,2,1],[1,2,2,2,2,312,31,23,123,1,1,1]])
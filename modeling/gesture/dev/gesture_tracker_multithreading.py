# import necessary packages
import cv2
import numpy as np
import mediapipe as mp
import os
import platform
from tkinter import filedialog as fd
import tkinter as tk
import time
import math
import matplotlib.cm 
import mediapipe.python.solutions.face_mesh_connections as mp_eyes
import matplotlib.pyplot as plt
import modeling.gesture.mp_points.pose_manipulation as points
from modeling import init
import multiprocessing
from typing import Union
from multiprocessing import Process
# from realtime_usage import realtime_usage
# initialize mediapipe
from gesture_tracker import gesture_tracker
import logging
class Logger():
    @property
    def logger(self):
        component = "{}.{}".format(type(self).__module__, type(self).__name__)
        #default log handler to dump output to console

        return logging.getLogger(component)
class VideoGet(Logger):

    def __init__(self, src):
        self.src = src
        self.capture = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        _, self.frame = self.capture.read()
        del self.capture
        self.index= 0
        self.process = Process(target = self.run, args=()).start()

    def run(self):
        self.capture = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        while True:
            _, self.frame = self.capture.read()
            self.index+=1
class VideoShow(Logger):

    def __init__(self, frame):
        self.frame = frame
        self.process = Process(target = self.run, args=()).start()
        self.index = 0
        try:
           cv2.imshow("video", self.frame)
        except:
            pass
        if cv2.waitKey(1) == ord("q"):
            self.stopped = True     
    def run(self):
        while True:
            cv2.imshow("video", self.frame)
            self.index+=1   

    def shutdown(self):
        print("Shutdown initiated")
        self.exit.set()

class gesture_tracker_multithreaded(gesture_tracker):
    def realtime_analysis(self, capture_index : int = 0, save_vid_file  : str = None, save_results_vid_file : str = None,   classification : str = None):
        if capture_index == None:
            capture_index = self.camera_selector()
        self.capture = VideoGet(capture_index)
        video_shower = VideoShow(self.capture.frame)
        first_frame = True  
        landmarks = None
        if classification:
            saved = []
        start_time = time.time()
        self.last_time = time.time()
        self.processed_frame = {}
        self.visibilty_dict = {}
        while True:
            frame = self.capture.frame
            if first_frame and save_vid_file is not None:
                curr = cv2.VideoWriter(save_vid_file, 
                            fourcc = self.etc["video_codec"],
                            fps = self.capture.get(cv2.CAP_PROP_FPS),
                            frameSize = (frame.shape[1], frame.shape[0]),
                            isColor = True)
            if first_frame and save_results_vid_file is not None:
                result = cv2.VideoWriter(save_results_vid_file, 
                            fourcc = self.etc["video_codec"],
                            fps = self.capture.get(cv2.CAP_PROP_FPS),
                            frameSize = (frame.shape[1], frame.shape[0]),
                            isColor = True)
            if first_frame:
                first_frame = False
                
            if save_vid_file:
                curr.write(frame)
            if save_results_vid_file:
                result.write(frame)
            try:
                landmarks = self.extract_landmarks(self.processed_frame["holistic"], classification)
                if classification is not None:
                    saved.append(landmarks)
            except:
                pass
            if landmarks is not None: 
                frame = self.frame_by_frame_check(frame, landmarks, True)
            frame = self.per_frame_analysis(frame, True)
            # video_shower.frame = self.process_frame
            video_shower.frame = self.process_frame
            if cv2.waitKey(1) == ord('q'):
                self.capture.stop()
                video_shower.stop()
                if save_results_vid_file:
                    result.release()
                    print("result_Release")
                if save_vid_file:
                    curr.release()
                    print("curr_Release")
                cv2.destroyAllWindows()
                break
            self.capture_index+=1
        if classification:
            return saved, time.time() - start_time
        return None, time.time() - start_time
    def per_frame_analysis(self, frame, show_final : bool = True):
        frame.flags.writeable = False
        # frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = True
        self.number_of_coordinates = 0
        
        if self.tracking_or_not["hand"] and self.tracking_or_not["pose"] and self.tracking_or_not["face"]: 
            a = time.time()           
            self.processed_frame["holistic"] = self.model_and_solution["holistic_model"].process(framergb)
            # print(1/(time.time() - a))
            self.visibilty_dict = {"left_hand" : self.processed_frame["holistic"].left_hand_landmarks is not None, "right_hand" : self.processed_frame["holistic"].right_hand_landmarks is not None, "pose": self.processed_frame["holistic"].pose_landmarks is not None, "face" : self.processed_frame["holistic"].face_landmarks is not None}
            try:
                self.number_of_coordinates = len(self.processed_frame["holistic"].pose_landmarks.landmark)+len(self.processed_frame["holistic"].face_landmarks.landmark)
            except:
                pass
            frame = self.draw_holistic(frame, self.processed_frame["holistic"])
        else:
            if self.tracking_or_not["hand"]:
                self.processed_frame["hand"] = self.model_and_solution["hand_model"].process(framergb)
                self.visibilty_dict["left_hand"] = self.processed_frame["hand"].left_hand_landmarks is not None
                self.visibilty_dict["right_hand"] = self.processed_frame["right_hand"].right_hand_landmarks is not None

                try:
                    self.number_of_coordinates += sum([len(hand.landmark) for hand in self.processed_frame["hand"].multi_hand_landmarks])
                    hand_landmarks = self.draw_hand(frame, self.processed_frame["hand"])
                except:
                    pass
            if self.tracking_or_not["pose"]:
                self.processed_frame["pose"] = self.model_and_solution["pose_model"].process(framergb)
                self.visibilty_dict["pose"] = self.processed_frame["pose"].pose_landmarks is not None
                try:
                    self.number_of_coordinates += len(self.processed_frame["pose"].pose_landmarks.landmark)
                    self.draw_pose(frame, self.processed_frame["pose"])
                except:
                    pass
            if self.tracking_or_not["face"]:
                self.processed_frame["face"] = self.model_and_solution["face_model"].process(framergb)
                self.visibilty_dict["face"] = self.processed_frame["face"].face_landmarks is not None
                try:
                    self.number_of_coordinates += len(self.processed_frame["face"].face_landmarks.landmark)
                    hand_landmarks = self.draw_face(frame, self.processed_frame["face"])
                except:
                    pass
        self.get_timer()
        if "fps" in self.etc:
            cv2.putText(frame, "FPS: " + str(self.etc["fps"]), (10,15), cv2.FONT_HERSHEY_PLAIN, fontScale = 1, thickness= 2, color = (0,0,0))
        self.process_frame = frame
        return frame
    
if __name__ == '__main__':
    a = gesture_tracker_multithreaded()
    a.realtime_analysis()
# print(np.mean(a.timer,axis = 1))
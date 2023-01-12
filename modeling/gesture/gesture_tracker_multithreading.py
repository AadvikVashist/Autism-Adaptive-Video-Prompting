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
import modeling.gesture.mp_points.gesture_tracking_points as points
from modeling import init
from typing import Union
from threading import Thread
# from realtime_usage import realtime_usage
# initialize mediapipe
from gesture_tracker import gesture_tracker
class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):    
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()
            print("hi")

    def stop(self):
        self.stopped = True
class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """
    def __init__(self, frame=None, frame_name = None):
        self.frame = frame
        self.stopped = False
        if frame_name is None:
            self.frame_name = "Video"
        else:
            self.frame_name = frame_name
    def start(self):
        Thread(target=self.show, args=()).start()
        return self
    def show(self):
        while not self.stopped:
            cv2.imshow(self.frame_name, self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True        
    def stop(self):
        self.stopped = True
class gesture_tracker_multithreaded(gesture_tracker):
    def realtime_analysis(self, capture_index : int = 0, save_vid_file  : str = None, save_results_vid_file : str = None,   classification : str = None):
        if capture_index == None:
            capture_index = self.camera_selector()
        self.capture = VideoGet(capture_index).start()
        video_shower = VideoShow(self.capture.read()[1]).start()
        first_frame = True  
        landmarks = None
        if classification:
            saved = []
        start_time = time.time()
        self.last_time = time.time()
        self.processed_frame = {}
        self.visibilty_dict = {}
        while True:
            _, frame = self.capture.frame
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
            Thread(target= self.per_frame_analysis, args=(frame, True)).start()
            video_shower.frame = self.process_frame
            cv2.imshow("Gesture tracked. Press Q to exit", self.process_frame)
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
    
a = gesture_tracker_multithreaded()
a.realtime_analysis()
# print(np.mean(a.timer,axis = 1))
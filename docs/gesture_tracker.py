# TechVidvan hand Gesture Recognizer

# import necessary packages

import cv2
import numpy as np
import mediapipe as mp
import dlib
import os
import pandas as pd
import platform
from tkinter import filedialog as fd
# initialize mediapipe

class gesture_tracker:
    def __init__(self, face : bool = True, hand : bool = True, pose : bool = True, face_confidence : float = 0.7, hand_confidence : float = 0.7, pose_confidence : float  = 0.7, number_of_hands : int = 2):
        self.hand = hand
        self.face = face
        self.pose = pose
        self.face_confidence = face_confidence
        self.hand_confidence = hand_confidence
        self.pose_confidence = pose_confidence
        self.hand_quantity = number_of_hands
        self.mediapipe_drawing = mp.solutions.drawing_utils
        self.mediapipe_drawing_styles = mp.solutions.drawing_styles
        if hand and pose and face:
            self.holistic_solution= mp.solutions.holistic
            self.holistic_model = mp.solutions.holistic.Holistic(static_image_mode=False,model_complexity=2,enable_segmentation=True,refine_face_landmarks=True) 
        else:
            if hand:#intialize the hand gesture tracker
                self.hand_solution = mp.solutions.hands
                self.hand_model = mp.solutions.hands.Hands(max_num_hands=self.hand_quantity, min_detection_confidence=hand_confidence) 
                
            if pose:
                self.pose_solution = mp.solutions.pose
                self.pose_model = 0
            if face:
                self.face_solution = mp.solutions.face_mesh 
                self.face_model = self.face_solution.FaceMesh(static_image_mode=False,max_num_faces=1,refine_landmarks=True,min_detection_confidence=face_confidence) #static_image_mode might need to change            
    def camera_selector(self) -> int:
        def camera_tester():
            camera_index = 0
            cameras = []
            captures = []
            while True:
                try:
                    cap =cv2.VideoCapture(camera_index)
                    ret, frame = cap.read()
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cameras.append(camera_index)
                    captures.append(cap)
                    camera_index+=1
                except:
                    break  
            return captures, cameras
        captures, cameras = camera_tester()
        if len(captures) > 1:
            x = 0
            y = 0
            for cap in cameras: 
                while True:
                    _, frame =  captures[cap].read()
                    cv2.imshow("You are currently viewing capture " + str(cap) + ". Press 'c' to continue",frame)  
                    if cv2.waitKey(1) == ord('c'):
                        captures[cap].release()
                        cv2.destroyAllWindows()
                        break
            while True:
                ind = input("which camera would you like to capture? (0 is default): ")
                try:
                    ind = int(ind)
                    return ind 
                except:
                    print("please try again")
        elif len(captures) == 0:
            return 0
        else:
            raise Exception("You do not have an accesible camera. Please try again")
    def file_selector(self, root = None):
        if platform.system() == 'Windows': # this hasn't been tested 
            desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop') #windows 
        elif platform.system() == 'Darwin':
            desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop') #mac 
        elif platform.system() == 'Linux':
            desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop') #linux
        else:
            raise Exception("Unsupported operating system: " + platform.system())
        filename = fd.askopenfilename(title = "select the video file you want", filetypes=[("video files", ("*.mp4", "*.m4p", "*.m4v","*.avi", "*.mov","*.mkv","*.wmv","*.webm"))], initialdir = desktop) #,
        resulting_file = os.path.splitext(filename)
        csv_file = resulting_file[0] + "_aadvikified.csv"
        resulting_file = resulting_file[0] + "_aadvikified.mp4"  #implement custom file return types laterresulting_file[1]
        return filename, resulting_file, csv_file
    def draw_holistic(self, frame, results):
        self.mediapipe_drawing.draw_landmarks(
            frame,
            results.face_landmarks,
            self.holistic_solution.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mediapipe_drawing_styles.get_default_face_mesh_tesselation_style())
        self.mediapipe_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.holistic_solution.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mediapipe_drawing_styles.get_default_pose_landmarks_style())
    def draw_pose(self, frame, results):
        self.mediapipe_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.pose_solution.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mediapipe_drawing_styles.get_default_pose_landmarks_style())
    def draw_face(self, frame, results):
        for face_landmarks in results.multi_face_landmarks:
            self.mediapipe_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.face_solution.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mediapipe_drawing_styles.get_default_pose_landmarks_style())
            self.mediapipe_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.face_solution.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mediapipe_drawing_styles.get_default_pose_landmarks_style())
            self.mediapipe_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.face_solution.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mediapipe_drawing_styles.get_default_pose_landmarks_style())
    def draw_hand(self, frame, results): 
        for hand_world_landmarks in results.multi_hand_world_landmarks:
            self.mediapipe_drawing.plot_landmarks(
                hand_world_landmarks, self.hand_solution.HAND_CONNECTIONS, azimuth=5)
    def per_frame_analysis(self, frame, show_final : bool = True, save_results : bool = True):
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    
        if self.hand and self.pose and self.face:
            print("processing")
            self.frame_holistic = self.holistic_model.process(framergb)
            print("drawing")
            self.draw_holistic(frame, self.frame_holistic)
        else:
            if self.hand:#intialize the hand gesture tracker
                self.frame_hand = self.hand_model.process(framergb)
                self.draw_hand(frame, self.frame_hand)
            if self.pose:
                self.frame_pose = self.pose_model.process(framergb)
                self.draw_pose(frame, self.frame_pose)
            if self.face:
                self.frame_face = self.face_model.process(framergb)
                self.draw_face(frame, self.frame_face)
        cv2.imshow("Gesture tracked", frame)
        return frame
    def realtime_analysis(self, capture_index = 0):
        if capture_index == None:
            capture_index = self.camera_selector()
        self.capture = cv2.VideoCapture(capture_index)
        while True:
            _, frame = self.capture.read()
            self.per_frame_analysis(frame, True, True)
            if cv2.waitKey(1) == ord('q'):
                self.capture.release()
                cv2.destroyAllWindows()
                break
    def create_csv(self, filename, filetype : str, recorded_values: list):
        s = 0
        
    def video_analysis_demo(self, video = None):
        def video_dimensions_fps(videofile):
            vid = cv2.VideoCapture(videofile)
            _,frame = vid.read()
            height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            fps = vid.get(cv2.CAP_PROP_FPS)
            vid.release()
            return int(width),int(height),fps
        if not video:
            video,resulting_video, save_csv = self.file_selector()
        self.capture = cv2.VideoCapture(video)
        self.vid_info = video_dimensions_fps(video)
        result = cv2.VideoWriter(resulting_video, 
                        fourcc = cv2.VideoWriter_fourcc(*'h264'),
                        fps = self.vid_info[2], frameSize= self.vid_info[0:2], isColor = True)    
        while True:
            _, frame = self.capture.read()
            if frame is None:
                self.capture.release()
                result.release()
                cv2.destroyAllWindows()
                break
            frame =self.per_frame_analysis(frame, True, True)
            result.write(frame)
a = gesture_tracker()
a.video_analysis_demo()
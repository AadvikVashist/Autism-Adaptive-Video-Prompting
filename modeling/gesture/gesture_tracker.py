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
import threading
from sklearn.linear_model import LinearRegression

# from realtime_usage import realtime_usage
# initialize mediapipe
class gesture_tracker:
    def __init__(self, eye : bool = True, face : bool = True, hand : bool = True, pose : bool = True,
                eye_confidence : float = 0.7, face_confidence : float = 0.7,
                hand_confidence : float = 0.7, pose_confidence : float  = 0.7,
                number_of_hands : int = 2):
        self.landmark_px = {
                            "face" : {},
                            "left_hand" : {},
                            "right_hand": {},
                            "pose" : {},
                            }
        
        self.etc = {
            "timers" : {
                "face" : {},
                "left_hand" : {},
                "right_hand": {},
                "pose" : {},
            },
            "timer" : time.time(),
            "colormap" : {
                "cmap" : matplotlib.cm.get_cmap('hsv')
            },
            "video_codec" : cv2.VideoWriter_fourcc(*'mp4v'),
            "max_wait_time" : 1
        }
        self.capture_index = 0
        self.tracking_or_not = {"eye" : eye,
                                "hand" : hand,
                                "face": face,
                                "pose": pose,
                                "eye_confidence" : eye_confidence,
                                "face_confidence" : face_confidence,
                                "hand_confidence" : hand_confidence,
                                "pose_confidence" : pose_confidence,
                                "hand_quantity" : number_of_hands,
                                }
        
        self.super_resolution = {"2" : cv2.dnn_superres.DnnSuperResImpl_create(),
                                "3" : cv2.dnn_superres.DnnSuperResImpl_create(),
                                "4" : cv2.dnn_superres.DnnSuperResImpl_create(),
                                }
        for i in range(2,5):
            self.super_resolution[str(i)].readModel("models/upscaling/ESPCN_x" + str(i) + ".pb")
            self.super_resolution[str(i)].setModel("espcn", i)
        
        self.mediapipe_drawing = mp.solutions.drawing_utils
        self.mediapipe_drawing_styles = mp.solutions.drawing_styles
        self.gesture_point_dict = {}
        self.model_and_solution = {}
        self.linear_model = LinearRegression()
        if self.tracking_or_not["hand"] and self.tracking_or_not["pose"] and self.tracking_or_not["face"]:
            self.gesture_point_dict["pose"] = points.get_pose_dict()
            self.gesture_point_dict["left_hand"] = points.get_hand_dict()["left_hand"]
            self.gesture_point_dict["right_hand"] = points.get_hand_dict()["right_hand"]
            self.gesture_point_dict["face"] = points.get_face_dict()
            self.etc["colormap"]["cmap_spacing"] = 1/(len(self.gesture_point_dict["face"].keys())-1)*0.8
            self.model_and_solution["holistic_solution"] = mp.solutions.holistic
            self.model_and_solution["holistic_model"] = self.model_and_solution["holistic_solution"].Holistic(static_image_mode=False,model_complexity=2,
                                                                    enable_segmentation=True,
                                                                    refine_face_landmarks=True,
                                                                    min_tracking_confidence=max(self.tracking_or_not["hand_confidence"],self.tracking_or_not["face_confidence"], self.tracking_or_not["pose_confidence"]),
                                                                    min_detection_confidence=max(self.tracking_or_not["hand_confidence"],self.tracking_or_not["face_confidence"], self.tracking_or_not["pose_confidence"]))
        else:
            if self.tracking_or_not["hand"]: #intialize the hand gesture tracker
                self.gesture_point_dict["left_hand"] = points.get_hand_dict()["left_hand"]
                self.gesture_point_dict["right_hand"] = points.get_hand_dict()["right_hand"]
                self.model_and_solution["hand_solution"] = mp.solutions.hands
                self.model_and_solution["hand_model"] = mp.solutions.hands.Hands(static_image_mode = False,
                                                            max_num_hands = self.self.tracking_or_not["hand_quantity"],
                                                            min_detection_confidence = self.tracking_or_not["hand_confidence"],
                                                            min_tracking_confidence = self.tracking_or_not["hand_confidence"]) 
            if self.tracking_or_not["pose"]:
                self.gesture_point_dict["pose"] = points.get_pose_dict()
                self.model_and_solution["pose_solution"] = mp.solutions.pose
                self.model_and_solution["pose_model"] = self.model_and_solution["pose_solution"].Pose(static_image_mode = False,
                                                        model_complexity = 2,
                                                        enable_segmentation = True,
                                                        min_detection_confidence = self.tracking_or_not["pose_confidence"],
                                                        min_tracking_confidence = self.tracking_or_not["pose_confidence"])
            if self.tracking_or_not["face"]:
                self.gesture_point_dict["face"] = points.get_face_dict()
                self.etc["colormap"]["cmap_spacing"] = 1/(len(self.gesture_point_dict["face"].keys())-1)*0.8
                self.model_and_solution["face_solution"] = mp.solutions.face_mesh 
                self.model_and_solution["face_model"] = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                                                    max_num_faces=1,
                                                                    refine_landmarks=True,
                                                                    min_detection_confidence = self.tracking_or_not["face_confidence"],
                                                                    min_tracking_confidence = self.tracking_or_not["face_confidence"]) #static_image_mode might need to change            
    
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
        filename = init.file_selector(None, filetypes=[("video files", ("*.mp4", "*.m4p", "*.m4v","*.avi", "*.mov","*.mkv","*.wmv","*.webm"))])
        resulting_file = os.path.splitext(filename)
        csv_file = resulting_file[0] + "_aadvikified.csv"
        resulting_file = resulting_file[0] + "_aadvikified.mp4"  #implement custom file return types laterresulting_file[1]
        return filename, resulting_file, csv_file
    
    def is_valid_normalized_value(self,value):
        return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))
    def _normalized_to_pixel_coordinates(self, normalized_x: float, normalized_y: float, image_width: int,image_height: int):
        """Converts normalized value pair to pixel coordinates."""
        # Checks if the float value is between 0 and 1.
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))
        # if not (is_valid_normalized_value(normalized_x) and
        #         is_valid_normalized_value(normalized_y)):
        #     # TODO: Draw coordinates even if it's outside of the image bounds.
        #     return None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return int(x_px),int(y_px), 
    
    def updated_landmarks(self, landmark_list, feature : Union[list,str]):
        def iter_landmarks(feat):
            new = {}
            for index, (key, value) in enumerate(self.gesture_point_dict[feat].items()):
                value = [landmark_list.landmark[val] for val in value]
                new[key] = value
            return new
        if type(feature) == list:
            ret ={}
            for key,feat in feature.items():
                val = iter_landmarks(feat)
                ret[key] = val
            return ret
        else:
            val = iter_landmarks(feature)
            return val

    def draw_face_proprietary(self,frame, landmark_list, individual_show = False):
        framed = None
        if landmark_list is None and len(self.landmark_px["face"]) != 0:
            if not("previous_elapsed" in self.etc["timers"]["face"] and self.etc["timers"]["face"]["previous_elapsed"] > self.etc["max_wait_time"]):
                for index, (key, value) in enumerate(self.landmark_px["face"].items()):
                    for val in value:
                        cv2.circle(frame, [val[0], val[1]], 1, np.array(self.etc["colormap"]["cmap"](index*self.etc["colormap"]["cmap_spacing"]))*255, 1)
        
        elif landmark_list is not None:
            image_rows, image_cols, _ = frame.shape
            face_dict = []
            for index, (key, value) in enumerate(self.gesture_point_dict["face"].items()):
                value = [self._normalized_to_pixel_coordinates(landmark_list.landmark[val].x,  landmark_list.landmark[val].y, image_cols, image_rows) for val in value]
                self.landmark_px["face"][key] = value
                for val in value:
                    cv2.circle(frame, [val[0], val[1]], 1, np.array(self.etc["colormap"]["cmap"](index*self.etc["colormap"]["cmap_spacing"]))*255, 1)
            
            lister = np.array(self.landmark_px["face"]["face_outline"])
            mins = np.min(lister, axis = 0)
            maxs = np.max(lister, axis = 0)
            framed = frame[int(mins[1]*0.8):int(maxs[1]*1.25), int(mins[0]*0.8):int(maxs[0]*1.25)]
            
        if individual_show and framed is not None and framed.shape[0] != 0:
            cv2.imshow("facial_points_drawn", framed)
            cv2.waitKey(0)
        return frame, framed
    
    def draw_pose_proprietary(self, frame: np.array, pose_landmark_list, individual_show = False):
        framed = None
        if pose_landmark_list == None and len(self.landmark_px["pose"]) != 0:
            if not("previous_elapsed" in self.etc["timers"]["pose"] and self.etc["timers"]["pose"]["previous_elapsed"] > self.etc["max_wait_time"]):
                for index, (key, value) in enumerate(self.landmark_px["pose"].items()):
                    for val in value:
                        cv2.circle(frame, [val[0], val[1]], 2, np.array(self.etc["colormap"]["cmap"](index*self.etc["colormap"]["cmap_spacing"]))*255, 2)
        elif pose_landmark_list is not None:
            
            image_rows, image_cols, _ = frame.shape
            for index, (key, value) in enumerate(self.gesture_point_dict["pose"].items()):
                value = [self._normalized_to_pixel_coordinates(pose_landmark_list.landmark[val].x,  pose_landmark_list.landmark[val].y, image_cols, image_rows) for val in value]
                self.landmark_px["pose"][key] = value
                for val in value:
                    cv2.circle(frame, [val[0], val[1]], 2, np.array(self.etc["colormap"]["cmap"](index*self.etc["colormap"]["cmap_spacing"]))*255, 2)
            
            lister = np.array(points.flatten(self.landmark_px["pose"]))
            mins = np.min(lister, axis = 0)
            maxs = np.max(lister, axis = 0)
            framed = frame[int(mins[1]*0.8):int(maxs[1]*1.25), int(mins[0]*0.8):int(maxs[0]*1.25)]
        if individual_show and framed is not None:
            cv2.imshow("facial_points_drawn", framed)               
            cv2.waitKey(0)
        return frame, framed
    
    def draw_hands_proprietary(self, frame: np.array, left_hand_landmarks, right_hand_landmarks, individual_show = False):
        frame_left = None; frame_right = None
        if left_hand_landmarks == None and len(self.landmark_px["left_hand"]) != 0:
            if not("previous_elapsed" in self.etc["timers"]["left_hand"] and self.etc["timers"]["left_hand"]["previous_elapsed"] > self.etc["max_wait_time"]):
                for index, (key, value) in enumerate(self.landmark_px["left_hand"].items()):
                    for val in value:
                        cv2.circle(frame, [val[0], val[1]], 2, np.array(self.etc["colormap"]["cmap"](index*self.etc["colormap"]["cmap_spacing"]))*255, 2)
        elif left_hand_landmarks is not None:
            
            image_rows, image_cols, _ = frame.shape
            for index, (key, value) in enumerate(self.gesture_point_dict["left_hand"].items()):
                value = [self._normalized_to_pixel_coordinates(left_hand_landmarks.landmark[val].x,  left_hand_landmarks.landmark[val].y, image_cols, image_rows) for val in value]
                self.landmark_px["left_hand"][key] = value
                for val in value:
                    cv2.circle(frame, [val[0], val[1]], 2, np.array(self.etc["colormap"]["cmap"](index*self.etc["colormap"]["cmap_spacing"]))*255, 2)
            
            lister = np.array(points.flatten(self.landmark_px["left_hand"]))
            mins = np.min(lister, axis = 0)
            maxs = np.max(lister, axis = 0)
            frame_left = frame[int(mins[1]*0.8):int(maxs[1]*1.25), int(mins[0]*0.8):int(maxs[0]*1.25)]
                    
        if right_hand_landmarks == None and len(self.landmark_px["right_hand"]) != 0:
            if not("previous_elapsed" in self.etc["timers"]["right_hand"] and self.etc["timers"]["right_hand"]["previous_elapsed"] > self.etc["max_wait_time"]):
                for index, (key, value) in enumerate(self.landmark_px["right_hand"].items()):
                    for val in value:
                        cv2.circle(frame, [val[0], val[1]], 2, np.array(self.etc["colormap"]["cmap"](index*self.etc["colormap"]["cmap_spacing"]))*255, 2)
        elif right_hand_landmarks is not None:
            
            image_rows, image_cols, _ = frame.shape
            for index, (key, value) in enumerate(self.gesture_point_dict["right_hand"].items()):
                value = [self._normalized_to_pixel_coordinates(right_hand_landmarks.landmark[val].x,  right_hand_landmarks.landmark[val].y, image_cols, image_rows) for val in value]
                self.landmark_px["right_hand"][key] = value
                for val in value:
                    cv2.circle(frame, [val[0], val[1]], 2, np.array(self.etc["colormap"]["cmap"](index*self.etc["colormap"]["cmap_spacing"]))*255, 2)
            
            lister = np.array(points.flatten(self.landmark_px["right_hand"]))
            mins = np.min(lister, axis = 0)
            maxs = np.max(lister, axis = 0)
            frame_right = frame[int(mins[1]*0.8):int(maxs[1]*1.25), int(mins[0]*0.8):int(maxs[0]*1.25)]
        if individual_show:
            if frame_left is not None:
                cv2.imshow("left_hand_points_drawn", frame_left)
                cv2.waitKey(0)
            if frame_right is not None:
                cv2.imshow("right_hand_points_drawn", frame_right)
                cv2.waitKey(0)
        return frame
    
    def face_pose(self, frame, landmark_list):
        nose_list = [33, 263, 1, 61, 291, 199]
        height, width, _ = frame.shape
        nose_listed = np.array([list(self._normalized_to_pixel_coordinates(land.x, land.y, width, height)) for index, land in enumerate(landmark_list.landmark) if index in nose_list])
        for i in nose_listed:
            try:
                cv2.circle(frame,i,2,(255,255,255),2)
            except:
                pass
        face_2d = [(lm.x * width, lm.y * height) for idx, lm in enumerate(landmark_list.landmark) if idx in nose_list]
        face_3d = [(lm.x * width, lm.y * height, lm.z ) for idx, lm in enumerate(landmark_list.landmark) if idx in nose_list]
        nose_3d = [landmark_list.landmark[1].x*width, landmark_list.landmark[1].y*height,landmark_list.landmark[1].z*3000]
        nose_2d = [landmark_list.landmark[1].x*width, landmark_list.landmark[1].y*height]
        
        # Convert to NumPy array
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        # The camera matrix
        focal_length = 1 * width

        cam_matrix = np.array([ [focal_length, 0, height / 2],
                                [0, focal_length, width / 2],
                                [0, 0, 1]])

        # The Distance Matrix
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        # Get rotational matrix
        rmat, jac = cv2.Rodrigues(rot_vec)

        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # Get the y rotation degree
        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360
        print(np.around((x,y,z),2))
        return frame
    
    def eyes(self, frame, landmark_list):
        return frame, landmark_list

    def draw_holistic(self, frame, results):
        self.draw_face_proprietary(frame, results.face_landmarks, False)
        self.draw_pose_proprietary(frame, results.pose_landmarks, False)
        self.draw_hands_proprietary(frame, results.left_hand_landmarks, results.right_hand_landmarks, False)
        if results.face_landmarks is not None:
            frame =  self.eyes(frame, results)
        return frame
        # self.mediapipe_drawing.draw_landmarks(
        #     frame,
        #     results.pose_landmarks,
        #     self.model_and_solution["holistic_solution"].POSE_CONNECTIONS,
        #     landmark_drawing_spec=self.mediapipe_drawing_styles.get_default_pose_landmarks_style())
    
    def draw_pose(self, frame, results):
        self.mediapipe_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.model_and_solution["pose_solution"].POSE_CONNECTIONS,
            landmark_drawing_spec=self.mediapipe_drawing_styles.get_default_pose_landmarks_style())
    
    def draw_face(self, frame, results):
        for face_landmarks in results.multi_face_lsandmarks:
            self.mediapipe_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.model_and_solution["face_solution"].FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mediapipe_drawing_styles.get_default_face_mesh_contours_style())
            self.mediapipe_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.model_and_solution["face_solution"].FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mediapipe_drawing_styles.get_default_face_mesh_iris_connections_style())
    
    def draw_hand(self, frame, results): 
        hands = []
        for hand_landmarks in results.multi_hand_landmarks:
            self.mediapipe_drawing.draw_landmarks(frame,
                hand_landmarks , self.model_and_solution["hand_solution"].HAND_CONNECTIONS,
                self.mediapipe_drawing_styles.get_default_hand_landmarks_style(),
                self.mediapipe_drawing_styles.get_default_hand_connections_style())
            landmarks = []
            for landmark in hand_landmarks:
                x,y,z = landmark.x,landmark.y,landmark.z
                landmarks.append(x,y,z)
            hands.append(landmarks)
        return np.array(hands, dtype = np.float32)
    
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
    
    def realtime_analysis(self, capture_index : int = 0, save_vid_file  : str = None, save_results_vid_file : str = None,   classification : str = None):
        if capture_index == None:
            capture_index = self.camera_selector()
        self.capture = cv2.VideoCapture(capture_index, cv2.CAP_DSHOW)
        first_frame = True  
        landmarks = None
        if classification:
            saved = []
        start_time = time.time()
        self.last_time = time.time()
        self.processed_frame = {}
        self.visibilty_dict = {}
        while True:
            _, frame = self.capture.read()
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
            cv2.imshow("Gesture tracked. Press Q to exit", frame)    
            if cv2.waitKey(1) == ord('q'):
                self.capture.release()
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
    
    
    def frame_by_frame_check(self, frame,landmarks, bool):
        return frame
    
    
    def video_analysis(self, video = None, result_video = None, classification = None):
        def video_dimensions_fps(videofile):
            vid = cv2.VideoCapture(videofile)
            _,frame = vid.read()
            height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            fps = vid.get(cv2.CAP_PROP_FPS)
            vid.release()
            return int(width),int(height),fps
        
        if not video:
            video,result_video, = self.file_selector()
        self.capture = cv2.VideoCapture(video)
        self.vid_info = video_dimensions_fps(video)
        result = cv2.VideoWriter(result_video, 
                        fourcc = self.etc["video_codec"],
                        fps = self.vid_info[2],
                        frameSize= self.vid_info[0:2],
                        isColor = True)   
        if classification:
            saved = []
        self.processed_frame = {}
        while True:
            _, frame = self.capture.read()
            if frame is None:
                self.capture.release()
                result.release()
                cv2.destroyAllWindows()
                break
            frame = self.per_frame_analysis(frame, True)
            if classification is not None:
                landmarks = self.extract_landmarks(self.processed_frame["holistic"], classification)
                saved.append(landmarks)
            result.write(frame)
        if classification:
            return saved
    
    
    def extract_landmarks(self, results, class_name = None):
        face =  self.updated_landmarks(results.face_landmarks.landmark, "face")
        pose = self.updated_landmarks(results.pose_landmarks.landmark, "pose")
        left_hand = self.updated_landmarks(results.left_hand_landmarks.landmark, "left_hand")
        right_hand = self.updated_landmarks(results.right_hand_landmarks.landmark, "right_hand")

        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z,] for landmark in pose]).flatten()) #removed  landmark.visibility from list, but if necessary, must add back
        face_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in face]).flatten())#removed  landmark.visibility from list, but if necessary, must add back
        left_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in left_hand]).flatten())#removed  landmark.visibility from list, but if necessary, must add back
        right_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in right_hand]).flatten())#removed  landmark.visibility from list, but if necessary, must add back

        # Concate rowsq
        row = pose_row+face_row+left_hand_row+right_hand_row
        # Append class name
        if class_name is not None:
            row.insert(0, class_name)
        return row
    
    def get_timer(self):
        curr_reference = time.time()
        self.etc["fps"]= np.around(1/(curr_reference - self.etc["timer"]), 2)
        self.etc["timer"] = curr_reference
        for feature, value in self.visibilty_dict.items():
            if not value:
                if len(self.etc["timers"][feature]) == 0:
                    self.etc["timers"][feature]["start"] = time.time()
                    self.etc["timers"][feature]["previous_elapsed"] = 0 
                elif int(curr_reference - self.etc["timers"][feature]["start"]) > self.etc["timers"][feature]["previous_elapsed"]:
                    self.etc["timers"][feature]["previous_elapsed"] = int(curr_reference - self.etc["timers"][feature]["start"])
                    print(feature,"time elapsed =", self.etc["timers"][feature]["previous_elapsed"] )
            else:
                if "start" in self.etc["timers"][feature]: #cleans up if face is found after camera starts
                    print(feature, "detected; breaking wait time for", feature)
                    del self.etc["timers"][feature]["start"]
                    del self.etc["timers"][feature]["previous_elapsed"]
# a = gesture_tracker()
# a.realtime_analysis()
# print(np.mean(a.timer,axis = 1))
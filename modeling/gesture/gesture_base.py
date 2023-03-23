# import necessary packages
import cv2
import numpy as np
import mediapipe as mp
import os

import time
import math
import matplotlib.cm 
import modeling.gesture.pose_manipulation.training_points as points
import modeling.gesture.eye_tracking as eye_tracking
import noduro
import noduro_code.read_settings as read_settings
# main class for all gesture tracking related things
import modeling.gesture.pose_manipulation.pose_standardizer as standardize
import matplotlib.pyplot as plt
import modeling.gesture.check_distance as check_distance
"""a base class that takes videos, as either a video file or through a capture. 
Base gestures are derived from mediapipe's Holistic, Hand, Pose, and Face Mesh modules. 
The gesture class doesn't need to use holistic, but it is strongle recommended.
The class takes a frameskip feature, that skips analysis to maximize frame per second whilst maintaining gesture analysis.
By default, propeitary points are tracked as opposed to all of them, with selected values being chosen to maximize runtime viability whilst maninting accuracy.
The start, stop, and while procesing functions all serve as placeholders for subclasses to allow for added functionality within the gesture tracking base, as it is a closed loop process without such options"""

class gesture_base:
    def __init__(self, eye : bool = True, face : bool = True, hand : bool = True, pose : bool = True,
                eye_confidence : float = 0.7, face_confidence : float = 0.7,
                hand_confidence : float = 0.7, pose_confidence : float  = 0.7,
                number_of_hands : int = 2, frameskip = False):
        self.track = { "start" : time.time() }
        self.draw = True
        self.landmark_px = { 
                            "face" : {},
                            "left_hand" : {},
                            "right_hand": {},
                            "pose" : {},
                            } # for drawing body parts using custom pointsets

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
            "max_wait_time" : 1,
            "optimal_fps" : 20,
            "distance" : 1.0
        } #various values that don't need their own class variables
        if frameskip == True:
            self.etc["frame_skip"] = read_settings.get_points()
        else:
            self.etc["frame_skip"] = 1
        self.tracking_or_not = {"eye" : eye,
                                "hand" : hand,
                                "face": face,
                                "pose": pose,
                                "eye_confidence" : eye_confidence,
                                "face_confidence" : face_confidence,
                                "hand_confidence" : hand_confidence,
                                "pose_confidence" : pose_confidence,
                                "hand_quantity" : number_of_hands,
                                } #basic information to cover whats being tracked, what isn't, and any other values
        
        # self.super_resolution = {"2" : cv2.dnn_superres.DnnSuperResImpl_create(),
        #                         "3" : cv2.dnn_superres.DnnSuperResImpl_create(),
        #                         "4" : cv2.dnn_superres.DnnSuperResImpl_create(),
        #                         }
        # for i in range(2,5):
        #     self.super_resolution[str(i)].readModel(noduro.subdir_path("data/analyzed/ESPCN_x" + str(i) + ".pb"))
        #     self.super_resolution[str(i)].setModel("espcn", i)
        #     a = noduro.subdir_path("data/analyzed/ESPCN_x" + str(i) + ".pb")
        
        self.mediapipe_drawing = mp.solutions.drawing_utils #setup
        self.mediapipe_drawing_styles = mp.solutions.drawing_styles
        self.gesture_point_dict = {} #values determined in .json file
        self.model_and_solution = {} #store the models
        self.track["init to model and solution"] = time.time() - self.track["start"]; self.track["start"] = time.time()
        if self.tracking_or_not["hand"] and self.tracking_or_not["pose"] and self.tracking_or_not["face"]: #holistic
            self.gesture_point_dict["pose"] = points.get_pose_dict()
            self.gesture_point_dict["left_hand"] = points.get_hand_dict()["left_hand"]
            self.gesture_point_dict["right_hand"] = points.get_hand_dict()["right_hand"]
            self.gesture_point_dict["face"] = points.get_face_dict()
            self.etc["colormap"]["cmap_spacing"] = 1/(len(self.gesture_point_dict["face"].keys())-1)*0.8
            self.model_and_solution["holistic_solution"] = mp.solutions.holistic
            self.model_and_solution["holistic_model"] = self.model_and_solution["holistic_solution"].Holistic(static_image_mode=False,model_complexity=1,
                                                                    refine_face_landmarks=True,
                                                                    min_tracking_confidence=max(self.tracking_or_not["hand_confidence"],self.tracking_or_not["face_confidence"], self.tracking_or_not["pose_confidence"]),
                                                                    min_detection_confidence=max(self.tracking_or_not["hand_confidence"],self.tracking_or_not["face_confidence"], self.tracking_or_not["pose_confidence"]))
        else: #missing 1+ body part(s)
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
        self.track["model and solution to end of solution"] = time.time() - self.track["start"]; self.track["start"] = time.time()
        self.start() #filler func for sub classes
    
    def camera_selector(self) -> int: #select camera 
        def camera_tester(): #test a camera
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
    
    def file_finder(self, root = None): #use classs file selector but also get return files
        filename = noduro.file_selector(None, filetypes=[("video files", ("*.mp4", "*.m4p", "*.m4v","*.avi", "*.mov","*.mkv","*.wmv","*.webm"))])
        resulting_file = os.path.splitext(filename)
        csv_file = resulting_file[0] + "_aadvikified.csv"
        resulting_file = resulting_file[0] + "_aadvikified.mp4"  #implement custom file return types laterresulting_file[1]
        return filename, resulting_file, csv_file
    
    def is_valid_normalized_value(self,value): #check if value is normalized
        return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

    def _normalized_to_pixel_coordinates(self, normalized_x: float, normalized_y: float, image_width: int,image_height: int): #Converts normalized value pair to pixel coordinates.
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return int(x_px),int(y_px), 

    def draw_face_proprietary(self,frame, landmark_list, individual_show = False, scalar = 1): #draw face using gesture points from .json 
        framed = None
        scale = int(scalar * 1.25 * self.etc["points_size"])
        if scale < 1:
            scale = 1
            
        if landmark_list is None and len(self.landmark_px["face"]) != 0: #if there has been a previous comparison point set, and less than a second has passed since that time. No landmarks found
            if not("previous_elapsed" in self.etc["timers"]["face"] and self.etc["timers"]["face"]["previous_elapsed"] > self.etc["max_wait_time"]):
                for index, (key, value) in enumerate(self.landmark_px["face"].items()):
                    if "eye" in key or "iris" in key:
                        scaled = int(scale/1.5)
                    else:
                        scaled = scale
                    for val in value:
                        cv2.circle(frame, [val[0], val[1]],0, np.array(self.etc["colormap"]["cmap"](index*self.etc["colormap"]["cmap_spacing"]))*255, scaled)
        
        elif landmark_list is not None: #if landmarks found
            image_rows, image_cols, _ = frame.shape
            face_dict = []
            for index, (key, value) in enumerate(self.gesture_point_dict["face"].items()):
                value = [self._normalized_to_pixel_coordinates(landmark_list.landmark[val].x,  landmark_list.landmark[val].y, image_cols, image_rows) for val in value]
                self.landmark_px["face"][key] = value
                if "eye" in key or "iris" in key:
                    scaled = int(scale/1.5)
                else:
                    scaled = scale

                for val in value:
                    cv2.circle(frame, [val[0], val[1]], 0, np.array(self.etc["colormap"]["cmap"](index*self.etc["colormap"]["cmap_spacing"]))*255, scaled)
            
            lister = np.array(self.landmark_px["face"]["face_outline"])
            mins = np.min(lister, axis = 0)
            maxs = np.max(lister, axis = 0)
            framed = frame[int(mins[1]*0.8):int(maxs[1]*1.25), int(mins[0]*0.8):int(maxs[0]*1.25)]
            
        if individual_show and framed is not None and framed.shape[0] != 0: #if you want to show the point drawing indvidually. 
            cv2.imshow("facial_points_drawn", framed)
            cv2.waitKey(0)
        return frame, framed
    
    def draw_pose_proprietary(self, frame: np.array, pose_landmark_list, individual_show = False, scalar = 1): #draw points for pose using .json file 
        framed = None
        scaled = int(scalar * 2 * self.etc["points_size"])
        if scaled < 1:
            scaled = 1
        if pose_landmark_list == None and len(self.landmark_px["pose"]) != 0:  #if there has been a previous comparison point set, and less than a second has passed since that time. No landmarks found
            if not("previous_elapsed" in self.etc["timers"]["pose"] and self.etc["timers"]["pose"]["previous_elapsed"] > self.etc["max_wait_time"]):
                for index, (key, value) in enumerate(self.landmark_px["pose"].items()):
                    for val in value:
                        cv2.circle(frame, [val[0], val[1]],0, np.array(self.etc["colormap"]["cmap"](index*self.etc["colormap"]["cmap_spacing"]))*255, scaled)
        elif pose_landmark_list is not None: #if landmarks found
            
            image_rows, image_cols, _ = frame.shape
            for index, (key, value) in enumerate(self.gesture_point_dict["pose"].items()):
                value = [self._normalized_to_pixel_coordinates(pose_landmark_list.landmark[val].x,  pose_landmark_list.landmark[val].y, image_cols, image_rows) for val in value]
                self.landmark_px["pose"][key] = value
                for val in value:
                    cv2.circle(frame, [val[0], val[1]], 0, np.array(self.etc["colormap"]["cmap"](index*self.etc["colormap"]["cmap_spacing"]))*255, scaled)
            
            lister = np.array(points.flatten(self.landmark_px["pose"]))
            mins = np.min(lister, axis = 0)
            maxs = np.max(lister, axis = 0)
            framed = frame[int(mins[1]*0.8):int(maxs[1]*1.25), int(mins[0]*0.8):int(maxs[0]*1.25)]
        if individual_show and framed is not None: #if isolated point display is required
            cv2.imshow("facial_points_drawn", framed)               
            cv2.waitKey(0)
        return frame, framed
    
    def draw_hands_proprietary(self, frame: np.array, left_hand_landmarks, right_hand_landmarks, individual_show = False, scalar = 1): #draws each hand seperately, as one may be tracked while the other isn't. Uses .json file
        frame_left = None; frame_right = None
        scaled = int(scalar * 2 * self.etc["points_size"])
        if scaled < 1:
            scaled = 1

        if left_hand_landmarks == None and len(self.landmark_px["left_hand"]) != 0:  #if there has been a previous comparison point set, and less than a second has passed since that time. No landmarks found
            if not("previous_elapsed" in self.etc["timers"]["left_hand"] and self.etc["timers"]["left_hand"]["previous_elapsed"] > self.etc["max_wait_time"]):
                for index, (key, value) in enumerate(self.landmark_px["left_hand"].items()):
                    for val in value:
                        cv2.circle(frame, [val[0], val[1]], 0, np.array(self.etc["colormap"]["cmap"](index*self.etc["colormap"]["cmap_spacing"]))*255, scaled)
        elif left_hand_landmarks is not None: #landmarks found
            image_rows, image_cols, _ = frame.shape
            for index, (key, value) in enumerate(self.gesture_point_dict["left_hand"].items()):
                value = [self._normalized_to_pixel_coordinates(left_hand_landmarks.landmark[val].x,  left_hand_landmarks.landmark[val].y, image_cols, image_rows) for val in value]
                self.landmark_px["left_hand"][key] = value
                for val in value:
                    cv2.circle(frame, [val[0], val[1]], 0, np.array(self.etc["colormap"]["cmap"](index*self.etc["colormap"]["cmap_spacing"]))*255, scaled)
            
            lister = np.array(points.flatten(self.landmark_px["left_hand"]))
            mins = np.min(lister, axis = 0)
            maxs = np.max(lister, axis = 0)
            frame_left = frame[int(mins[1]*0.8):int(maxs[1]*1.25), int(mins[0]*0.8):int(maxs[0]*1.25)]
                    
        if right_hand_landmarks == None and len(self.landmark_px["right_hand"]) != 0:  #if there has been a previous comparison point set, and less than a second has passed since that time. No landmarks found
            if not("previous_elapsed" in self.etc["timers"]["right_hand"] and self.etc["timers"]["right_hand"]["previous_elapsed"] > self.etc["max_wait_time"]):
                for index, (key, value) in enumerate(self.landmark_px["right_hand"].items()):
                    for val in value:
                        cv2.circle(frame, [val[0], val[1]], 0, np.array(self.etc["colormap"]["cmap"](index*self.etc["colormap"]["cmap_spacing"]))*255, scaled)
        elif right_hand_landmarks is not None: #landmarks found
            image_rows, image_cols, _ = frame.shape
            for index, (key, value) in enumerate(self.gesture_point_dict["right_hand"].items()):
                right_hand_scale = np.mean([right_hand_landmarks.landmark[val].z for val in value])
                value = [self._normalized_to_pixel_coordinates(right_hand_landmarks.landmark[val].x,  right_hand_landmarks.landmark[val].y, image_cols, image_rows) for val in value]
                self.landmark_px["right_hand"][key] = value
                for val in value:
                    cv2.circle(frame, [val[0], val[1]], 0, np.array(self.etc["colormap"]["cmap"](index*self.etc["colormap"]["cmap_spacing"]))*255, scaled)
            
            lister = np.array(points.flatten(self.landmark_px["right_hand"]))
            mins = np.min(lister, axis = 0)
            maxs = np.max(lister, axis = 0)
            frame_right = frame[int(mins[1]*0.8):int(maxs[1]*1.25), int(mins[0]*0.8):int(maxs[0]*1.25)]
        if individual_show: #if specialized display required
            if frame_left is not None:
                cv2.imshow("left_hand_points_drawn", frame_left)
                cv2.waitKey(0)
            if frame_right is not None:
                cv2.imshow("right_hand_points_drawn", frame_right)
                cv2.waitKey(0)
        return frame
    
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
    
    def frame_by_frame_check(self, frame,landmarks, bool):
        return frame
    

    def draw_on_screen(self,frame, texts : list):
        scalar = int(np.round(frame.shape[0]/1000))
        _, h = cv2.getTextSize("a", cv2.FONT_HERSHEY_PLAIN, 2*scalar, scalar)[0]
        h1 = 1.4*h
        w1 = h/2
        for text in texts:
            cv2.putText(frame, text, np.int32([w1,h1]), cv2.FONT_HERSHEY_PLAIN, 2*scalar, (255, 255, 255), scalar, cv2.LINE_AA)
            h1 += 1.3*h
        return frame

    def start(self): #filler func for sub classes
        return None
    def while_processing(self,frame,process):
        return frame
    def end(self):
        return None

if __name__ == '__main__':
    a = gesture_base(frameskip = True)
    # a.video_analysis("C:/Users/aadvi/Desktop/IMG_1004.mp4", result_video = "C:/Users/aadvi/Desktop/vid.mp4")
    a.realtime_analysis()
    # a.realtime_analysis(save_vid_file="C:/Users/aadvi/Desktop/Autism/Autism-Adaptive-Video-Prompting/data/raw/gestures/happy/2023-02-21-18-09-10/capture.mp4") #("C:/Users/aadvi/Desktop/Movie on 2-8-23 at 9.43 AM.mov")
#     a.video_analysis("C:/Users/aadvi/Desktop/Autism/Autism-Adaptive-Video-Pcerompting/data/raw/gestures/cutting/2023-02-18-10-14-06/test1.mp4")
# print(np.mean(a.timer,axis = 1))
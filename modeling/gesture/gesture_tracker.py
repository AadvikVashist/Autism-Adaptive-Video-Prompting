# import necessary packages
import cv2
import numpy as np
import mediapipe as mp
import os

import time
import math
import matplotlib.cm 
import modeling.gesture.pose_manipulation.training_points as points
from typing import Union
import modeling.gesture.eye_tracking as eye_tracking
import noduro
import noduro_code.read_settings as read_settings
# main class for all gesture tracking related things
import modeling.gesture.pose_manipulation.pose_standardizer as standardize
import matplotlib.pyplot as plt
import modeling.gesture.check_distance as check_distance
global track
track = {}
class gesture_tracker:
    def __init__(self, eye : bool = True, face : bool = True, hand : bool = True, pose : bool = True,
                eye_confidence : float = 0.7, face_confidence : float = 0.7,
                hand_confidence : float = 0.7, pose_confidence : float  = 0.7,
                number_of_hands : int = 2, frameskip = False):
        track["start"] = time.time()
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
        track["init to model and solution"] = time.time() - track["start"]; track["start"] = time.time()
        if self.tracking_or_not["hand"] and self.tracking_or_not["pose"] and self.tracking_or_not["face"]: #holistic
            self.gesture_point_dict["pose"] = points.get_pose_dict()
            self.gesture_point_dict["left_hand"] = points.get_hand_dict()["left_hand"]
            self.gesture_point_dict["right_hand"] = points.get_hand_dict()["right_hand"]
            self.gesture_point_dict["face"] = points.get_face_dict()
            self.etc["colormap"]["cmap_spacing"] = 1/(len(self.gesture_point_dict["face"].keys())-1)*0.8
            self.model_and_solution["holistic_solution"] = mp.solutions.holistic
            self.model_and_solution["holistic_model"] = self.model_and_solution["holistic_solution"].Holistic(static_image_mode=False,model_complexity=1,
                                                                    enable_segmentation =False,
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
        track["model and solution to end of solution"] = time.time() - track["start"]; track["start"] = time.time()
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

    def draw_face_proprietary(self,frame, landmark_list, individual_show = False): #draw face using gesture points from .json 
        framed = None
        if landmark_list is None and len(self.landmark_px["face"]) != 0: #if there has been a previous comparison point set, and less than a second has passed since that time. No landmarks found
            if not("previous_elapsed" in self.etc["timers"]["face"] and self.etc["timers"]["face"]["previous_elapsed"] > self.etc["max_wait_time"]):
                for index, (key, value) in enumerate(self.landmark_px["face"].items()):
                    for val in value:
                        cv2.circle(frame, [val[0], val[1]], 1, np.array(self.etc["colormap"]["cmap"](index*self.etc["colormap"]["cmap_spacing"]))*255, 1)
        
        elif landmark_list is not None: #if landmarks found
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
            
        if individual_show and framed is not None and framed.shape[0] != 0: #if you want to show the point drawing indvidually. 
            cv2.imshow("facial_points_drawn", framed)
            cv2.waitKey(0)
        return frame, framed
    
    def draw_pose_proprietary(self, frame: np.array, pose_landmark_list, individual_show = False): #draw points for pose using .json file 
        framed = None
        if pose_landmark_list == None and len(self.landmark_px["pose"]) != 0:  #if there has been a previous comparison point set, and less than a second has passed since that time. No landmarks found
            if not("previous_elapsed" in self.etc["timers"]["pose"] and self.etc["timers"]["pose"]["previous_elapsed"] > self.etc["max_wait_time"]):
                for index, (key, value) in enumerate(self.landmark_px["pose"].items()):
                    for val in value:
                        cv2.circle(frame, [val[0], val[1]], 2, np.array(self.etc["colormap"]["cmap"](index*self.etc["colormap"]["cmap_spacing"]))*255, 2)
        elif pose_landmark_list is not None: #if landmarks found
            
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
        if individual_show and framed is not None: #if isolated point display is required
            cv2.imshow("facial_points_drawn", framed)               
            cv2.waitKey(0)
        return frame, framed
    
    def draw_hands_proprietary(self, frame: np.array, left_hand_landmarks, right_hand_landmarks, individual_show = False): #draws each hand seperately, as one may be tracked while the other isn't. Uses .json file
        frame_left = None; frame_right = None
        if left_hand_landmarks == None and len(self.landmark_px["left_hand"]) != 0:  #if there has been a previous comparison point set, and less than a second has passed since that time. No landmarks found
            if not("previous_elapsed" in self.etc["timers"]["left_hand"] and self.etc["timers"]["left_hand"]["previous_elapsed"] > self.etc["max_wait_time"]):
                for index, (key, value) in enumerate(self.landmark_px["left_hand"].items()):
                    for val in value:
                        cv2.circle(frame, [val[0], val[1]], 2, np.array(self.etc["colormap"]["cmap"](index*self.etc["colormap"]["cmap_spacing"]))*255, 2)
        elif left_hand_landmarks is not None: #landmarks found
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
                    
        if right_hand_landmarks == None and len(self.landmark_px["right_hand"]) != 0:  #if there has been a previous comparison point set, and less than a second has passed since that time. No landmarks found
            if not("previous_elapsed" in self.etc["timers"]["right_hand"] and self.etc["timers"]["right_hand"]["previous_elapsed"] > self.etc["max_wait_time"]):
                for index, (key, value) in enumerate(self.landmark_px["right_hand"].items()):
                    for val in value:
                        cv2.circle(frame, [val[0], val[1]], 2, np.array(self.etc["colormap"]["cmap"](index*self.etc["colormap"]["cmap_spacing"]))*255, 2)
        elif right_hand_landmarks is not None: #landmarks found
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
        if individual_show: #if specialized display required
            if frame_left is not None:
                cv2.imshow("left_hand_points_drawn", frame_left)
                cv2.waitKey(0)
            if frame_right is not None:
                cv2.imshow("right_hand_points_drawn", frame_right)
                cv2.waitKey(0)
        return frame
    
    def eyes(self, frame, landmarks):
        image_rows, image_cols, _ = frame.shape
        face_list, pose_list = eye_tracking.landmarks_to_lists(landmarks)
        right_iris = self.gesture_point_dict["face"]["right_iris"]
        left_iris = self.gesture_point_dict["face"]["left_iris"]
        chest = self.gesture_point_dict["pose"]["chest"]
        nose_line = self.gesture_point_dict["face"]["nose_line"]
        right_iris_list = np.array([list(self._normalized_to_pixel_coordinates(land.x, land.y, image_cols, image_rows)) for index, land in enumerate(face_list.landmark) if index in right_iris]) #rewrite to index with right iris
        left_iris_list = np.array([list(self._normalized_to_pixel_coordinates(land.x, land.y, image_cols, image_rows)) for index, land in enumerate(face_list.landmark) if index in left_iris]) #rewrite
        chest_list = np.array([list(self._normalized_to_pixel_coordinates(land.x, land.y, image_cols, image_rows)) for index, land in enumerate(pose_list.landmark) if index in chest]) #rewrite
        nose_list = np.array([list(self._normalized_to_pixel_coordinates(land.x, land.y, image_cols, image_rows)) for index, land in enumerate(face_list.landmark) if index in nose_line]) #rewrite

        king_joshua_ratio, nose_angle,angle_diff, new_point, body_center, eye_center, _, _ = eye_tracking.calculate_eye_ratio(left_iris_list,right_iris_list,chest_list,nose_list)
        frame = eye_tracking.draw_eye_calculations(frame,eye_center,angle_diff,king_joshua_ratio, body_center, nose_angle, chest_list, new_point)
        return frame

    def draw_holistic(self, frame, results): #run the drawing algorithms
        self.draw_face_proprietary(frame, results.face_landmarks, False)
        self.draw_pose_proprietary(frame, results.pose_landmarks, False)
        self.draw_hands_proprietary(frame, results.left_hand_landmarks, results.right_hand_landmarks, False)
        if results.face_landmarks is not None:
            frame =  self.eyes(frame, results)
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
    
    def per_frame_analysis(self, frame, show_final : bool = True):
        track["loop to start of per frame"] = time.time() - track["start"]; track["start"] = time.time()
        # frame.flags.writeable = False #save speed
        # frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame.flags.writeable = True #so you can fix image
        start = time.time() # check time
        if self.tracking_or_not["hand"] and self.tracking_or_not["pose"] and self.tracking_or_not["face"]: #holistic
            self.processed_frame["holistic"] = self.model_and_solution["holistic_model"].process(framergb) #run holistic model
            track["per frame to process"] = time.time() - track["start"]; track["start"] = time.time()
            # print(1/(time.time() - a))
            self.visibilty_dict = {"left_hand" : self.processed_frame["holistic"].left_hand_landmarks is not None, "right_hand" : self.processed_frame["holistic"].right_hand_landmarks is not None, "pose": self.processed_frame["holistic"].pose_landmarks is not None, "face" : self.processed_frame["holistic"].face_landmarks is not None} #check if any is none
        else:
            if self.tracking_or_not["hand"]:
                self.processed_frame["hand"] = self.model_and_solution["hand_model"].process(framergb)
                self.visibilty_dict["left_hand"] = self.processed_frame["hand"].left_hand_landmarks is not None
                self.visibilty_dict["right_hand"] = self.processed_frame["right_hand"].right_hand_landmarks is not None
            if self.tracking_or_not["pose"]:
                self.processed_frame["pose"] = self.model_and_solution["pose_model"].process(framergb)
                self.visibilty_dict["pose"] = self.processed_frame["pose"].pose_landmarks is not None
            if self.tracking_or_not["face"]:
                self.processed_frame["face"] = self.model_and_solution["face_model"].process(framergb)
                self.visibilty_dict["face"] = self.processed_frame["face"].face_landmarks is not None
        self.get_timer()
        track["per frame to timer"] = time.time() - track["start"]; track["start"] = time.time()
        self.process_frame = frame #frame post processing(could have drawing points in it as well)
        _ = self.while_processing(self.process_frame) #filler for any subclasses
        if _ is not None:
            frame = _
        track["timer to end of per frame"] = time.time() - track["start"]; track["start"] = time.time()
        return frame

    def frame_by_frame_check(self, frame,landmarks, bool):
        return frame
    
    def looping_analysis(self, videoCapture : object, video_shape = None, fps = None,  result_vid : str = None, starting_vid: str = None, frame_skip : int = None, save_pose : bool = False, standardize_pose : bool = False, save_frames = False):
        self.processed_frame = {}
        self.visibilty_dict = {}
        
        if fps is None:
            fps = 30
        _, frame = videoCapture.read()

        #frame_skip
        if frame_skip is None:
            frame_skip = self.etc["frame_skip"] #recommended frame skip vs set value 
        if video_shape is None:
            video_shape = (frame.shape[1], frame.shape[0])
        #video writer objects

        if starting_vid is not None: #whether to make a new video writer
            curr = cv2.VideoWriter(starting_vid, 
                fourcc = self.etc["video_codec"],
                fps = fps,
                frameSize = video_shape,
                isColor = True)
        if result_vid is not None: #whether to make a result video writer
            result = cv2.VideoWriter(result_vid, 
                fourcc = self.etc["video_codec"],
                fps = fps,
                frameSize = video_shape,
                isColor = True)
        #loops
        self.etc["frame_index"] = 0 
        if save_pose:
            self.save_pose = []
        if standardize_pose:
            self.save_calibrated_pose = []
            self.moving_average = []
            try:
                self.etc["moving_average_length"] = int(30/frame_skip)
            except:
                self.etc["moving_average_length"] = 5
        track["capture to b4 loop"] = time.time() - track["start"]; track["start"] = time.time()        
        while True:
            _, frame = videoCapture.read()
            
            if frame is None:
                break
            if starting_vid: #write frame to the file
                curr.write(frame)
            
            if frame_skip != 0 and self.etc["frame_index"] % self.etc["frame_skip"] == 0 : #if you want to skip frames
                frame = self.per_frame_analysis(frame, True) #run frame analysis
                
                if save_pose or standardize_pose:
                    gesture_dic = standardize.convert_holistic_to_dict(self.processed_frame["holistic"])
                if save_pose and save_frames:
                    self.save_pose.append(standardize.filter_body_parts(gesture_dic, self.gesture_point_dict))
                elif save_pose:
                    self.save_pose = standardize.filter_body_parts(gesture_dic, self.gesture_point_dict)
                # closer_or_farther = check_distance.closer_or_farther(_)
                # print(closer_or_farther)
                if standardize_pose:
                    stand, distance = standardize.center_and_scale_from_raw(gesture_dic, self.gesture_point_dict,self.moving_average)
                    #if the array isn't long enough, force add
                    if len(self.moving_average) < self.etc["moving_average_length"]:
                        self.moving_average.append(distance)
                    else:
                        self.moving_average.append(distance)
                        del self.moving_average[0]
                    if save_frames:
                        self.save_calibrated_pose.append(stand)
                    else:
                        self.save_calibrated_pose = stand
            
            if self.tracking_or_not["hand"] and self.tracking_or_not["pose"] and self.tracking_or_not["face"]:
                frame = self.draw_holistic(frame, self.processed_frame["holistic"])
            
            if "fps" in self.etc:
                cv2.putText(frame, "FPS: " + str(self.etc["fps"]), (10,15), cv2.FONT_HERSHEY_PLAIN, fontScale = 1, thickness= 2, color = (0,0,0))
            if "gesture" in self.etc:
                cv2.putText(frame, self.etc["gesture"][0] + ", " + self.etc["gesture"][1], (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            if result_vid: #write the results
                result.write(frame)
            
            self.etc["frame_index"] += 1
            
            #Frame displays
            cv2.imshow("Gesture tracked. Press Q to exit", frame) #show tracking    
            if cv2.waitKey(1) == ord('q'): #stop everything
                videoCapture.release()
                if result_vid:
                    result.release()
                    print("result_Release")
                if starting_vid:
                    curr.release()
                    print("curr_Release")
                cv2.destroyAllWindows()
                break
        self.end() #ending

    def realtime_analysis(self, capture_index : int = 0, save_vid_file  : str = None, save_results_vid_file : str = None, frame_skip = None, analyze = True):
        if capture_index == None:
            capture_index = self.camera_selector() #select camera
        track["end of init to capture"] = time.time() - track["start"]; track["start"] = time.time()
        self.capture = cv2.VideoCapture(capture_index, cv2.CAP_DSHOW) #cap_show makes startup alot faster. Starts camera
        first_frame = True  
        landmarks = None
        self.looping_analysis(videoCapture = self.capture, video_shape = None, fps = None, result_vid = save_results_vid_file, starting_vid = save_vid_file, frame_skip = frame_skip, save_pose = analyze, standardize_pose = analyze, save_frames = analyze)    
    
    def video_dimensions_fps(self,videofile):
        vid = cv2.VideoCapture(videofile) #vid capture object
        _,frame = vid.read()
        height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        fps = int(np.round(vid.get(cv2.CAP_PROP_FPS),0))
        vid.release()
        return int(width),int(height),fps
    
    def video_analysis(self, video = None, result_video = None, frame_skip = 1, standardize_pose = True):
        if not video:
            video,result_video, = self.file_finder() #get file if not provided
        self.capture = cv2.VideoCapture(video)
        self.vid_info = self.video_dimensions_fps(video)
        self.video_file = video
        self.looping_analysis(self.capture, self.vid_info[0:2], self.vid_info[2], result_video, None, frame_skip = 1, standardize_pose=standardize_pose)

    def get_timer(self): #timers to check tracking data
        curr_reference = time.time()
        self.etc["fps"] = np.around(self.etc["frame_skip"]/(curr_reference - self.etc["timer"]), 2)
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
    
    def start(self): #filler func for sub classes
        return None
    def while_processing(self,frame):
        return None
    def end(self):
        return None

if __name__ == '__main__':
    a = gesture_tracker(frameskip = True)
    a.realtime_analysis(save_vid_file="C:/Users/aadvi/Desktop/Autism/Autism-Adaptive-Video-Prompting/data/raw/gestures/happy/2023-02-21-18-09-10/capture.mp4") #("C:/Users/aadvi/Desktop/Movie on 2-8-23 at 9.43 AM.mov")
#     a.video_analysis("C:/Users/aadvi/Desktop/Autism/Autism-Adaptive-Video-Prompting/data/raw/gestures/cutting/2023-02-18-10-14-06/test1.mp4")
# print(np.mean(a.timer,axis = 1))
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
from modeling.gesture.gesture_base import gesture_base
"""a base class that takes videos, as either a video file or through a capture. 
Base gestures are derived from mediapipe's Holistic, Hand, Pose, and Face Mesh modules. 
The gesture class doesn't need to use holistic, but it is strongle recommended.
The class takes a frameskip feature, that skips analysis to maximize frame per second whilst maintaining gesture analysis.
By default, propeitary points are tracked as opposed to all of them, with selected values being chosen to maximize runtime viability whilst maninting accuracy.
The start, stop, and while procesing functions all serve as placeholders for subclasses to allow for added functionality within the gesture tracking base, as it is a closed loop process without such options"""

class gesture_tracker(gesture_base):
    def draw_holistic(self, frame, results, scalar = 1): #run the drawing algorithms
        self.draw_face_proprietary(frame, results.face_landmarks, False,scalar)
        self.draw_pose_proprietary(frame, results.pose_landmarks, False,scalar)
        self.draw_hands_proprietary(frame, results.left_hand_landmarks, results.right_hand_landmarks, False,scalar)
        if results.face_landmarks is not None:
            frame,displays =  self.eyes(frame, results)
            if displays is not None:
                self.etc["screen_elements"] = displays
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
        frame = eye_tracking.draw_eye_calculations(frame,eye_center,angle_diff,king_joshua_ratio, body_center, nose_angle, chest_list,new_point)
        return frame

    def per_frame_analysis(self, frame, show_final : bool = True):
        self.track["loop to start of per frame"] = time.time() - self.track["start"]; self.track["start"] = time.time()
        # frame.flags.writeable = False #save speed
        # frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame.flags.writeable = True #so you can fix image
        start = time.time() # check time
        if self.tracking_or_not["hand"] and self.tracking_or_not["pose"] and self.tracking_or_not["face"]: #holistic
            self.processed_frame["holistic"] = self.model_and_solution["holistic_model"].process(framergb) #run holistic model
            self.track["per frame to process"] = time.time() - self.track["start"]; self.track["start"] = time.time()
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
        self.track["per frame to timer"] = time.time() - self.track["start"]; self.track["start"] = time.time()
        self.process_frame = frame #frame post processing(could have drawing points in it as well)
        return frame

    def draw_on_screen(self,frame, texts : list):
        if "h" not in self.etc:
            self.etc["scalar"] =  int(np.round(frame.shape[0]/800))
            _, self.etc["h"] = cv2.getTextSize("a", cv2.FONT_HERSHEY_PLAIN, 2*self.etc["scalar"], self.etc["scalar"])[0]
        h1 = 1.4*self.etc["h"]
        w = self.etc["h"]/2
        for text in texts:
            cv2.putText(frame, text, np.int32([self.etc["h"]/2,h1]), cv2.FONT_HERSHEY_PLAIN, 2*self.etc["scalar"], (255, 255, 255), self.etc["scalar"], cv2.LINE_AA)
            h1 += 1.3*self.etc["h"]
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
        
        self.track["capture to b4 loop"] = time.time() - self.track["start"]; self.track["start"] = time.time()        
        while True:
            _, frame = videoCapture.read()
            self.track["capture_read"] = time.time()-self.track["start"];self.track["start"] = time.time()

            if frame is None:
                videoCapture.release()
                if result_vid:
                    result.release()
                    print("result_Release")
                if starting_vid:
                    curr.release()
                    print("curr_Release")
                cv2.destroyAllWindows()
                break
            if starting_vid: #write frame to the file
                curr.write(frame)
            if "points_size" not in self.etc:
                self.etc["points_size"] = int(np.mean(frame.shape[:2])/150)
            
            if frame_skip != 0 and self.etc["frame_index"] % self.etc["frame_skip"] == 0 : #if you want to skip frames
                frame = self.per_frame_analysis(frame, True) #run frame analysis
                _ = self.while_processing(frame,True) #filler for any subclasses
                if _ is not None:
                    frame = _
                if save_pose or standardize_pose:
                    gesture_dic = standardize.convert_holistic_to_dict(self.processed_frame["holistic"])
                if save_pose and save_frames:
                    self.save_pose.append(standardize.filter_body_parts_faster(gesture_dic, self.gpdict_flatten))
                elif save_pose:
                    self.save_pose = standardize.filter_body_parts_faster(gesture_dic, self.gpdict_flatten)
                # closer_or_farther = check_distance.closer_or_farther(_)
                # print(closer_or_farther)s
                if standardize_pose:
                    try:
                        stand, self.etc["distance"] = standardize.center_and_scale_from_given(gesture_dic, self.gpdict_flatten,self.moving_average)
                        # standardize.display_pose_direct(stand)
                        #if the array isn't long enough, force add
                        if save_pose and save_frames:
                            self.save_calibrated_pose.append(stand)
                        elif save_pose:
                            self.save_calibrated_pose = stand
                        if len(self.moving_average) < self.etc["moving_average_length"]:
                            self.moving_average.append(self.etc["distance"])
                        else:
                            self.moving_average.append(self.etc["distance"])
                            del self.moving_average[0]
                    except:
                        pass
            else:
                _ = self.while_processing(frame,False) #filler for any subclasses
                if _ is not None:
                    frame = _
            start = time.time()
            if self.tracking_or_not["hand"] and self.tracking_or_not["pose"] and self.tracking_or_not["face"] and self.draw == True:
                frame = self.draw_holistic(frame, self.processed_frame["holistic"],self.etc["distance"])
            displays = []
            if "fps" in self.etc:
                displays.append("FPS: " + str(self.etc["fps"]))
            if "gesture" in self.etc:
                displays.append(str(self.etc["gesture"][0]) + ", " + str(self.etc["gesture"][1]))
            if "screen_elements" in self.etc:
                displays.extend(self.etc["screen_elements"])
            frame = self.draw_on_screen(frame, displays)
            if result_vid: #write the results
                result.write(frame)
            self.etc["frame_index"] += 1
            # if "width" not in self.etc:
            #     abc,self.etc["width"],self.etc["height"] = noduro.scale_image_to_window(frame)
            # else:
            #     abc, _, _ =noduro.scale_image_to_window(frame,self.etc["width"],self.etc["height"])
            #Frame displays
            cv2.imshow("Gesture tracked. Press Q to exit", frame) #show tracking    
            # cv2.imshow("Gesture tracked. Press Q to exit", cv2.resize(frame,(int(frame.shape[1]/3), int(frame.shape[0]/3)))) #show tracking    
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
        self.track["end of init to capture"] = time.time() - self.track["start"]; self.track["start"] = time.time()
        self.capture = cv2.VideoCapture(capture_index, cv2.CAP_DSHOW) #cap_show makes startup alot faster. Starts camera
        self.capture = noduro.set_resolution(self.capture, 1920,1080)
        first_frame = True  
        landmarks = None
        self.looping_analysis(videoCapture = self.capture, video_shape = None, fps = None, result_vid = save_results_vid_file, starting_vid = save_vid_file, frame_skip = frame_skip, save_pose = analyze, standardize_pose = analyze, save_frames = analyze)    

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
    def start(self):
        self.gpdict_flatten = standardize.flatten_gesture_point_dict_keys_to_list(self.gesture_point_dict)
if __name__ == '__main__':
    a = gesture_tracker(frameskip = True)
    # a.video_analysis("C:/Users/aadvi/Desktop/IMG_1004.mp4", result_video = "C:/Users/aadvi/Desktop/vid.mp4")
    a.realtime_analysis()
    # a.realtime_analysis(save_vid_file="C:/Users/aadvi/Desktop/Autism/Autism-Adaptive-Video-Prompting/data/raw/gestures/happy/2023-02-21-18-09-10/capture.mp4") #("C:/Users/aadvi/Desktop/Movie on 2-8-23 at 9.43 AM.mov")
#     a.video_analysis("C:/Users/aadvi/Desktop/Autism/Autism-Adaptive-Video-Pcerompting/data/raw/gestures/cutting/2023-02-18-10-14-06/test1.mp4")
# print(np.mean(a.timer,axis = 1))

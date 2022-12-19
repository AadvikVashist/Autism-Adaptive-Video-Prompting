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
# from realtime_usage import realtime_usage
# initialize mediapipe
class gesture_tracker:
    def __init__(self, eye : bool = True, face : bool = True, hand : bool = True, pose : bool = True, eye_confidence : float = 0.7, face_confidence : float = 0.7, hand_confidence : float = 0.7, pose_confidence : float  = 0.7, number_of_hands : int = 2):
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.optimal_y = 80
        
        self.eye = eye
        self.hand = hand
        self.face = face
        self.pose = pose
        self.eye = eye
        self.eye_confidence = self.eye
        self.face_confidence = face_confidence
        self.hand_confidence = hand_confidence
        self.pose_confidence = pose_confidence
        self.hand_quantity = number_of_hands
        self.mediapipe_drawing = mp.solutions.drawing_utils
        self.mediapipe_drawing_styles = mp.solutions.drawing_styles
        
        self.super_resolution2 = cv2.dnn_superres.DnnSuperResImpl_create()
        self.super_resolution3 = cv2.dnn_superres.DnnSuperResImpl_create()
        self.super_resolution4 = cv2.dnn_superres.DnnSuperResImpl_create()
        
        self.super_resolution2.readModel("model/upscaling/ESPCN_x2.pb")
        self.super_resolution2.setModel("espcn",2)
        
        self.super_resolution3.readModel("model/upscaling/ESPCN_x3.pb")
        self.super_resolution3.setModel("espcn",3)
        
        self.super_resolution4.readModel("model/upscaling/ESPCN_x3.pb")
        self.super_resolution4.setModel("espcn",4)
        if hand and pose and face:
            self.holistic_solution= mp.solutions.holistic
            self.holistic_model = self.holistic_solution.Holistic(static_image_mode=False,model_complexity=2,
                                                                    enable_segmentation=True,
                                                                    refine_face_landmarks=True,
                                                                    min_tracking_confidence=max(hand_confidence,face_confidence, pose_confidence),
                                                                    min_detection_confidence=max(hand_confidence,face_confidence, pose_confidence))
        else:
            if hand:#intialize the hand gesture tracker
                self.hand_solution = mp.solutions.hands
                self.hand_model = mp.solutions.hands.Hands(static_image_mode = False,
                                                            max_num_hands = self.hand_quantity,
                                                            min_detection_confidence = hand_confidence,
                                                            min_tracking_confidence = hand_confidence) 
            if pose:
                self.pose_solution = mp.solutions.pose
                self.pose_model = self.pose_solution.Pose(static_image_mode = False,
                                                        model_complexity = 2,
                                                        enable_segmentation = True,
                                                        min_detection_confidence = pose_confidence,
                                                        min_tracking_confidence = pose_confidence)
            if face:
                self.face_solution = mp.solutions.face_mesh 
                self.face_model = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                                                    max_num_faces=1,
                                                                    refine_landmarks=True,
                                                                    min_detection_confidence = face_confidence,
                                                                    min_tracking_confidence = face_confidence) #static_image_mode might need to change            
    
    
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
        window = tk.Tk()
        window.wm_attributes('-topmost', 1)
        window.withdraw()
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
    def _normalized_to_pixel_coordinates(self, normalized_x: float, normalized_y: float, image_width: int,image_height: int):
        """Converts normalized value pair to pixel coordinates."""
        # Checks if the float value is between 0 and 1.
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))
        if not (is_valid_normalized_value(normalized_x) and
                is_valid_normalized_value(normalized_y)):
            # TODO: Draw coordinates even if it's outside of the image bounds.
            return None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return int(x_px),int(y_px)
        
        
    def draw_points(self,frame, landmark_list):
        _VISIBILITY_THRESHOLD = 0.5
        _PRESENCE_THRESHOLD = 0.5
        image_rows, image_cols, _ = frame.shape
        landmark_listed = []
        cmap = matplotlib.cm.get_cmap('Spectral')
        for idx, landmark in enumerate(landmark_list.landmark):
            if ((landmark.HasField('visibility') and landmark.visibility < _VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                landmark.presence < _PRESENCE_THRESHOLD)):
                continue
            landmark_px = self._normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                        image_cols, image_rows)
            landmark_listed.append(landmark_px)
            # color = list(cmap(idx/len(landmark_list.landmark))[0:3]); color[0] *= 255; color[1] *= 255; color[2] *= 255
            color = np.random.choice(range(0,150), size=3)
            color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 
            
            cv2.circle(frame, landmark_px, 1, color, 4)
            cv2.imshow("xyz",frame)
            print(idx) # 7 22 23 24 25 26 27 28 29 30 33 56
            cv2.waitKey(0)
            
            time.sleep(0.1)
        return frame, landmark_listed
    
    
    def eyes(self, frame, landmark_list):
        mask = np.zeros_like(frame)

        right_eye = list(set(np.array(list(mp_eyes.FACEMESH_RIGHT_EYE)).flatten()))
        left_eye = list(set(np.array(list(mp_eyes.FACEMESH_LEFT_EYE)).flatten()))
        right_iris = list(set(np.array(list(mp_eyes.FACEMESH_RIGHT_IRIS)).flatten()))
        left_iris = list(set(np.array(list(mp_eyes.FACEMESH_LEFT_IRIS)).flatten()))
        image_rows, image_cols, _ = frame.shape    
        if any((right_eye is None, left_eye is None, right_iris is None, left_iris is None)):
            return
        right_eye_list = np.array([list(self._normalized_to_pixel_coordinates(land.x, land.y, image_cols, image_rows)) for index, land in enumerate(landmark_list.landmark) if index in right_eye])
        left_eye_list = np.array([list(self._normalized_to_pixel_coordinates(land.x, land.y, image_cols, image_rows)) for index, land in enumerate(landmark_list.landmark) if index in left_eye])
        right_iris_list = np.array([list(self._normalized_to_pixel_coordinates(land.x, land.y, image_cols, image_rows)) for index, land in enumerate(landmark_list.landmark) if index in right_iris])
        left_iris_list = np.array([list(self._normalized_to_pixel_coordinates(land.x, land.y, image_cols, image_rows)) for index, land in enumerate(landmark_list.landmark) if index in left_iris])
        
        cv2.fillPoly(mask, pts = [right_eye_list], color = (255,255,255))
        cv2.fillPoly(mask, pts = [left_eye_list], color = (255,255,255))
        right_eye_list = np.array([list(self._normalized_to_pixel_coordinates(land.x, land.y, image_cols, image_rows)) for index, land in enumerate(landmark_list.landmark) if index in right_eye])
        left_eye_list = np.array([list(self._normalized_to_pixel_coordinates(land.x, land.y, image_cols, image_rows)) for index, land in enumerate(landmark_list.landmark) if index in left_eye])
        
        # right_iris_list = np.array([list(self._normalized_to_pixel_coordinates(land.x, land.y, image_cols, image_rows)) for index, land in enumerate(landmark_list.landmark) if index in right_iris])
        # left_iris_list = np.array([list(self._normalized_to_pixel_coordinates(land.x, land.y, image_cols, image_rows)) for index, land in enumerate(landmark_list.landmark) if index in left_iris])
        # cv2.fillPoly(mask, pts = [right_iris_list], color = (255,255,255))
        # cv2.fillPoly(mask, pts = [left_iris_list], color = (255,255,255))

        kernel = np.ones((2,2),np.uint8)
        mask = cv2.dilate(mask,kernel,iterations = 5)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        x,y,w,h = cv2.boundingRect(np.concatenate(contours))
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        masked_img = cv2.bitwise_and(frame, mask)[y:y+h,x:x+w]
        while masked_img.shape[0] < self.optimal_y:
            scalar = round(masked_img.shape[0] / self.optimal_y)
            if scalar >= 4:
                masked_img = self.super_resolution4.upsample(masked_img)
            elif scalar >= 3:
                masked_img = self.super_resolution3.upsample(masked_img)
            else:
                masked_img = self.super_resolution2.upsample(masked_img)
        # cv2.resize(masked_img, (int(masked_img.shape[1]*6),int(masked_img.shape[0]*6)), interpolation = cv2.INTER_AREA)
        cv2.imshow('Masked Image Upscale', masked_img)
        # contours, hierarchy = cv2.findContours(cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(masked_img, contours, -1, (0, 255, 0), 3)
        # # masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
        # # masked_img = cv2.resize(masked_img, (int(masked_img.shape[1]*6),int(masked_img.shape[0]*6)), interpolation = cv2.INTER_AREA)
        # cv2.imshow('Masked Image', masked_img)
        # gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
        # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
        # # ensure at least some circles were found
        # if circles is not None:
        #     # convert the (x, y) coordinates and radius of the circles to integers
        #     circles = np.round(circles[0, :]).astype("int")
        #     # loop over the (x, y) coordinates and radius of the circles
        #     for (x, y, r) in circles:
        #         # draw the circle in the output image, then draw a rectangle
        #         # corresponding to the center of the circle
        #         cv2.circle(masked_img, (x, y), r, (0, 255, 0), 4)
        # cv2.imshow('Masked Image', masked_img)
        # cv2.waitKey(0)
    def draw_holistic(self, frame, results):
        if results.face_landmarks is not None:
            self.eyes(frame, results.face_landmarks)

    def draw_pose(self, frame, results):
        self.mediapipe_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.pose_solution.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mediapipe_drawing_styles.get_default_pose_landmarks_style())
    
    
    def draw_face(self, frame, results):
        for face_landmarks in results.multi_face_lsandmarks:
            self.mediapipe_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.face_solution.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mediapipe_drawing_styles.get_default_face_mesh_contours_style())
            self.mediapipe_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.face_solution.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mediapipe_drawing_styles.get_default_face_mesh_iris_connections_style())
    
    
    def draw_hand(self, frame, results): 
        hands = []
        for hand_landmarks in results.multi_hand_landmarks:
            self.mediapipe_drawing.draw_landmarks(frame,
                hand_landmarks , self.hand_solution.HAND_CONNECTIONS,
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
        
        if self.hand and self.pose and self.face:
            self.frame_holistic = self.holistic_model.process(framergb)
            try:
                self.number_of_coordinates = len(self.frame_holistic.pose_landmarks.landmark)+len(self.frame_holistic.face_landmarks.landmark)
            except:
                pass
            self.draw_holistic(frame, self.frame_holistic)
        else:
            if self.hand:#intialize the hand gesture tracker
                self.frame_hand = self.hand_model.process(framergb)
                try:
                    self.number_of_coordinates += sum([len(hand.landmark) for hand in self.frame_hand.multi_hand_landmarks])
                    hand_landmarks = self.draw_hand(frame, self.frame_hand)
                except:
                    pass
                    # print("hand not found")
            if self.pose:
                self.frame_pose = self.pose_model.process(framergb)
                try:
                    self.number_of_coordinates += len(self.frame_pose.pose_landmarks.landmark)
                    self.draw_pose(frame, self.frame_pose)
                except:
                    pass
                    #print("pose not found")
            if self.face:
                self.frame_face = self.face_model.process(framergb)
                try:
                    self.number_of_coordinates += len(self.frame_face.face_landmarks.landmark)
                    hand_landmarks = self.draw_face(frame, self.frame_face)
                except:
                    pass
                    #print("face not found")
                    
        return frame
    
    
    def realtime_analysis(self, capture_index : int = 0, save_vid_file  : str = None, save_results_vid_file : str = None, classification : str = None):
        if capture_index == None:
            capture_index = self.camera_selector()
        self.capture = cv2.VideoCapture(capture_index, cv2.CAP_DSHOW)
        first_frame = True  
        landmarks = None
        if classification:
            saved = []
        start_time = time.time()    
        while True:
            _, frame = self.capture.read()
            if first_frame and save_vid_file is not None:
                curr = cv2.VideoWriter(save_vid_file, 
                            fourcc = self.fourcc,
                            fps = self.capture.get(cv2.CAP_PROP_FPS),
                            frameSize = (frame.shape[1], frame.shape[0]),
                            isColor = True)
            if first_frame and save_results_vid_file is not None:
                result = cv2.VideoWriter(save_results_vid_file, 
                            fourcc = self.fourcc,
                            fps = self.capture.get(cv2.CAP_PROP_FPS),
                            frameSize = (frame.shape[1], frame.shape[0]),
                            isColor = True)
            first_frame = False
                
            if save_vid_file:
                curr.write(frame)
            frame = self.per_frame_analysis(frame, True)
            if save_results_vid_file:
                result.write(frame)
            try:
                landmarks = self.extract_landmarks(self.frame_holistic, classification)
                if classification is not None:
                    saved.append(landmarks)
            except:
                pass
            if landmarks is not None:
                frame = self.frame_by_frame_check(frame, landmarks, True)
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
                        fourcc = self.fourcc,
                        fps = self.vid_info[2],
                        frameSize= self.vid_info[0:2],
                        isColor = True)   
        if classification:
            saved = []  
        while True:
            _, frame = self.capture.read()
            if frame is None:
                self.capture.release()
                result.release()
                cv2.destroyAllWindows()
                break
            frame = self.per_frame_analysis(frame, True, True)
            if classification is not None:
                landmarks = self.extract_landmarks(self.frame_holistic, classification)
                saved.append(landmarks)
            result.write(frame)
        if classification:
            return saved
    
    def extract_landmarks(self, results, class_name = None):
        pose = results.pose_landmarks.landmark
        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
        
        # Extract Face landmarks
        face = results.face_landmarks.landmark
        face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
        
        # Concate rows
        row = pose_row+face_row
        
        # Append class name
        if class_name is not None:
            row.insert(0, class_name)
        return row
a = gesture_tracker()
a.realtime_analysis()
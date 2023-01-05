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
import tracking_points as points
# from realtime_usage import realtime_usage
# initialize mediapipe
class gesture_tracker:
    def __init__(self, eye : bool = True, face : bool = True, hand : bool = True, pose : bool = True, eye_confidence : float = 0.7, face_confidence : float = 0.7, hand_confidence : float = 0.7, pose_confidence : float  = 0.7, number_of_hands : int = 2):
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.optimal_y = 80
        self.landmark_px = []
        self.odd_values = {}
        
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
        
        self.super_resolution2.readModel("models/upscaling/ESPCN_x2.pb")
        self.super_resolution2.setModel("espcn",2)
        
        self.super_resolution3.readModel("models/upscaling/ESPCN_x3.pb")
        self.super_resolution3.setModel("espcn",3)
        
        self.super_resolution4.readModel("models/upscaling/ESPCN_x3.pb")
        self.super_resolution4.setModel("espcn",4)
        
        self.hsv = matplotlib.cm.get_cmap('hsv')
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
        # if not (is_valid_normalized_value(normalized_x) and
        #         is_valid_normalized_value(normalized_y)):
        #     # TODO: Draw coordinates even if it's outside of the image bounds.
        #     return None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return int(x_px),int(y_px)
    
    
    def draw_points(self,frame, landmark_list):
        if landmark_list is None and len(self.landmark_px) != 0:
            for px in self.landmark_px:
                cv2.circle(frame, [px[0]*3, px[1]*3], 2, np.array(self.hsv(0.9))*255, 2)
            return frame
        elif landmark_list is not None:
            if "start_wait_time" in self.odd_values:
                print("face_detected")
                del self.odd_values["start_wait_time"]
                del self.odd_values["wait_time"]
            self.pose_dict = points.get_pose_dict()
            self.face_dict = points.get_face_dict()
            iteration = 1/(len(self.face_dict.keys())-1)*0.8
            image_rows, image_cols, _ = frame.shape
            face_dict = []
            face_list =  list(self.face_dict.values())
            # Mask of valid places in each row
            for key, dictionary in self.face_dict.items():
                if dictionary is not None:
                    face_dict.extend(dictionary)
            frame = cv2.resize(frame, [frame.shape[1]*3, frame.shape[0]*3])
            self.landmark_px = []
            for idx, landmark in enumerate(landmark_list.landmark):
                if idx in face_dict:
                    landmark_px = self._normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                                image_cols, image_rows)

                    index = [index for index, i in enumerate(face_list) if idx in i][0]
                    cv2.circle(frame, [landmark_px[0]*3, landmark_px[1]*3], 2, np.array(self.hsv(index*iteration))*255, 2)
                    self.landmark_px.append(landmark_px)
            lister = np.array(self.landmark_px)
            mins = np.min(lister, axis = 0)
            maxs = np.max(lister, axis = 0)
            frame = frame[mins[1]*3-50:maxs[1]*3+50, mins[0]*3-50:maxs[0]*3+50]
        else:
            curr_time = time.time()
            if "start_wait_time" not in self.odd_values:
                self.odd_values["start_wait_time"] = curr_time
                self.odd_values["wait_time"] = curr_time
            if int(self.odd_values["wait_time"] - self.odd_values["start_wait_time"]) < int(curr_time - self.odd_values["start_wait_time"]):
                print("waiting for face detection for " + str(int(self.odd_values["wait_time"] - self.odd_values["start_wait_time"])) + " second")
            self.odd_values["wait_time"] = curr_time
        cv2.imshow("frame", frame)
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
        self.draw_points(frame, results.face_landmarks)
        if results.face_landmarks is not None:
            self.eyes(frame, results.face_landmarks)
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
            curr_time = time.time(); self.timer[5].append(curr_time - self.last_time); self.last_time = curr_time #5
            self.frame_holistic = self.holistic_model.process(framergb)
            try:
                self.number_of_coordinates = len(self.frame_holistic.pose_landmarks.landmark)+len(self.frame_holistic.face_landmarks.landmark)
            except:
                pass
            curr_time = time.time(); self.timer[6].append(curr_time - self.last_time); self.last_time = curr_time #6
            self.draw_holistic(frame, self.frame_holistic)
            curr_time = time.time(); self.timer[7].append(curr_time - self.last_time); self.last_time = curr_time #7
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
        self.last_time = time.time()
        self.timer = [[],[],[],[],[],[],[],[],[],[],[],[]]
        while True:
            self.timer[0].append(0); self.last_time = time.time() # 0
            _, frame = self.capture.read()
            curr_time = time.time(); self.timer[1].append(curr_time - self.last_time); self.last_time = curr_time # 1
            if first_frame and save_vid_file is not None:
                curr = cv2.VideoWriter(save_vid_file, 
                            fourcc = self.fourcc,
                            fps = self.capture.get(cv2.CAP_PROP_FPS),
                            frameSize = (frame.shape[1], frame.shape[0]),
                            isColor = True)
            curr_time = time.time(); self.timer[2].append(curr_time - self.last_time); self.last_time = curr_time #2
            if first_frame and save_results_vid_file is not None:
                result = cv2.VideoWriter(save_results_vid_file, 
                            fourcc = self.fourcc,
                            fps = self.capture.get(cv2.CAP_PROP_FPS),
                            frameSize = (frame.shape[1], frame.shape[0]),
                            isColor = True)
            curr_time = time.time(); self.timer[3].append(curr_time - self.last_time); self.last_time = curr_time #3
            if first_frame:
                first_frame = False
                
            if save_vid_file:
                curr.write(frame)
            
            curr_time = time.time(); self.timer[4].append(curr_time - self.last_time); self.last_time = curr_time #4
            frame = self.per_frame_analysis(frame, True)
            curr_time = time.time(); self.timer[8].append(curr_time - self.last_time); self.last_time = curr_time #8
            if save_results_vid_file:
                result.write(frame)
            try:
                curr_time = time.time(); self.timer[9].append(curr_time - self.last_time); self.last_time = curr_time #9
                landmarks = self.extract_landmarks(self.frame_holistic, classification)
                if classification is not None:
                    saved.append(landmarks)
                curr_time = time.time(); self.timer[10].append(curr_time - self.last_time); self.last_time = curr_time #10
            except:
                pass
            if landmarks is not None:
                frame = self.frame_by_frame_check(frame, landmarks, True)
            cv2.imshow("Gesture tracked. Press Q to exit", frame)
            curr_time = time.time(); self.timer[11].append(curr_time - self.last_time); self.last_time = curr_time #11
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
        
        # Concate rowsq
        row = pose_row+face_row
        
        # Append class name
        if class_name is not None:
            row.insert(0, class_name)
        return row
a = gesture_tracker()
a.realtime_analysis()
# print(np.mean(a.timer,axis = 1))
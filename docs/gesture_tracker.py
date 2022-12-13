# import necessary packages
import cv2
import numpy as np
import mediapipe as mp
import os
import platform
from tkinter import filedialog as fd
import tkinter as tk
# from realtime_usage import realtime_usage
# initialize mediapipe
class gesture_tracker:
    def __init__(self, face : bool = True, hand : bool = True, pose : bool = True, face_confidence : float = 0.7, hand_confidence : float = 0.7, pose_confidence : float  = 0.7, number_of_hands : int = 2):
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
    
    
    def draw_holistic(self, frame, results):
        # self.mediapipe_drawing.draw_landmarks(frame, results.face_landmarks, self.holistic_solution.FACEMESH_CONTOURS, 
        #                         self.mediapipe_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        #                         self.mediapipe_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        #                         )
        
        # 2. Right hand
        self.mediapipe_drawing.draw_landmarks(frame, results.right_hand_landmarks, self.holistic_solution.HAND_CONNECTIONS, 
                                self.mediapipe_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                self.mediapipe_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                )

        # 3. Left Hand
        self.mediapipe_drawing.draw_landmarks(frame, results.left_hand_landmarks, self.holistic_solution.HAND_CONNECTIONS, 
                                self.mediapipe_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                self.mediapipe_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                )

        # 4. Pose Detections
        self.mediapipe_drawing.draw_landmarks(frame, results.pose_landmarks, self.holistic_solution.POSE_CONNECTIONS, 
                                self.mediapipe_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                self.mediapipe_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                )
    
    
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
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = True
        self.number_of_coordinates = 0
        
        if self.hand and self.pose and self.face:
            self.frame_holistic = self.holistic_model.process(framergb)
            try:
                self.number_of_coordinates = len(self.frame_holistic.pose_landmarks.landmark)+len(self.frame_holistic.face_landmarks.landmark)
                self.draw_holistic(frame, self.frame_holistic)
            except:
                pass
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
        while True:
            _, frame = self.capture.read()
            if first_frame and save_vid_file is not None:
                curr = cv2.VideoWriter(save_vid_file, 
                            fourcc = self.fourcc,
                            fps = 30,
                            frameSize = (frame.shape[1], frame.shape[0]),
                            isColor = True)
            if first_frame and save_results_vid_file is not None:
                result = cv2.VideoWriter(save_results_vid_file, 
                            fourcc = self.fourcc,
                            fps = 30,
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
            return saved
    
    
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

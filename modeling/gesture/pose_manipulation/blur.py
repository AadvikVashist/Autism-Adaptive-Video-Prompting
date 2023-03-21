from modeling.gesture.gesture_tracker import gesture_tracker 
import cv2
import numpy as np
import modeling.gesture.pose_manipulation.training_points as points
import noduro
from screeninfo import get_monitors
import time
class blur(gesture_tracker):
    def start(self):
        self.draw = True
    def while_processing(self, frame, process):
        landmark_list = self.processed_frame["holistic"].face_landmarks
    # roi_corners = np.array([[(180,300),(120,540),(110,480),(160,350)]],dtype = np.int32)
    # # Read the original Image:

    # 
        listed = []
        if landmark_list is None: #if there has been a previous comparison point set, and less than a second has passed since that time. No landmarks found
            try:
                value = self.px["face_outline"]
                for val in value:
                    listed.append([val[0], val[1]])
            except:
                pass
        elif landmark_list is not None: #if landmarks found
            image_rows, image_cols, _ = frame.shape
            self.px = {}
            for index, (key, value) in enumerate(self.gesture_point_dict["face"].items()):
                value = [self._normalized_to_pixel_coordinates(landmark_list.landmark[val].x,  landmark_list.landmark[val].y, image_cols, image_rows) for val in value]
                self.px[key] = value
                if key == "face_outline":
                    for val in value:
                        listed.append([val[0], val[1]])    

        lister = np.array(self.px["face_outline"])
        mean = np.mean(lister, axis = 0)
        mins = np.min(lister, axis = 0)
        maxs = np.max(lister, axis = 0)
        lister = np.array([[int(a) for a in ((l-mean)[0]*1.5,(l-mean)[1]*1.5)+mean] for l in lister],dtype = np.int32)
        # # create a blurred copy of the entire image:
        lister = [(l[0],l[1]) for l in lister]
        if "blur" not in self.etc:
            self.etc["blur"] = (int((frame.shape[0]+frame.shape[1])/40)+1, int((frame.shape[0]+frame.shape[1])/40)+1)
        blurred_image = cv2.GaussianBlur(frame,self.etc["blur"], 50)
        # create a mask for the ROI and fill in the ROI with (255,255,255) color :
        mask = np.zeros(frame.shape, dtype=np.uint8)
        channel_count = frame.shape[2]
        ignore_mask_color = (255,)*channel_count
        # cv2.fillPoly(mask, np.int32([lister]), ignore_mask_color)
        hull = cv2.convexHull(np.int32([lister]))
        cv2.fillConvexPoly(mask, hull, ignore_mask_color)
        x = [(frame.shape[1]*4/5,frame.shape[0]*8/9),
             (frame.shape[1]*4/5,frame.shape[0]*1/5),
             (frame.shape[1],frame.shape[0]*1/5),
             (frame.shape[1],frame.shape[0]*8/9)
            ]
        # create a mask for everywhere in the original image except the ROI, (hence mask_inverse) :
        for box in self.boxes:
            cv2.rectangle(mask, np.int32((box[0],box[1])),np.int32((box[0]+box[2],box[1]+box[3])), ignore_mask_color,-2)

        
        mask_inverse = np.ones(mask.shape).astype(np.uint8)*255 - mask
        # # combine all the masks and above images in the following way :

        frame = cv2.bitwise_and(blurred_image, mask) + cv2.bitwise_and(frame, mask_inverse)
        return frame
    def draw_hands_proprietary(self, frame: np.array, left_hand_landmarks, right_hand_landmarks, individual_show = False): #draws each hand seperately, as one may be tracked while the other isn't. Uses .json file
        frame_left = None; frame_right = None
        lines = [(0,1),(1,2),(2,3),(3,4),(3,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17),(0,13),(0,9)]
        if left_hand_landmarks == None and len(self.landmark_px["left_hand"]) != 0:  #if there has been a previous comparison point set, and less than a second has passed since that time. No landmarks found
            if not("previous_elapsed" in self.etc["timers"]["left_hand"] and self.etc["timers"]["left_hand"]["previous_elapsed"] > self.etc["max_wait_time"]):
                for index, (key, value) in enumerate(self.landmark_px["left_hand"].items()):
                    for line in lines:
                        cv2.line(frame, value[line[0]],value[line[1]],(0,255,0),3)

                    for val in value:
                        cv2.circle(frame, [val[0], val[1]], 6, np.array(self.etc["colormap"]["cmap"](index*self.etc["colormap"]["cmap_spacing"]))*255, 6)
        elif left_hand_landmarks is not None: #landmarks found
            image_rows, image_cols, _ = frame.shape
            for index, (key, value) in enumerate(self.gesture_point_dict["left_hand"].items()):
                value = [self._normalized_to_pixel_coordinates(left_hand_landmarks.landmark[val].x,  left_hand_landmarks.landmark[val].y, image_cols, image_rows) for val in value]
                self.landmark_px["left_hand"][key] = value
                for line in lines:
                    cv2.line(frame, value[line[0]],value[line[1]],(0,255,0),3)

                for val in value:
                    cv2.circle(frame, [val[0], val[1]], 6, np.array(self.etc["colormap"]["cmap"](index*self.etc["colormap"]["cmap_spacing"]))*255, 6)
            
            lister = np.array(points.flatten(self.landmark_px["left_hand"]))
            mins = np.min(lister, axis = 0)
            maxs = np.max(lister, axis = 0)
            frame_left = frame[int(mins[1]*0.8):int(maxs[1]*1.25), int(mins[0]*0.8):int(maxs[0]*1.25)]
                    
        if right_hand_landmarks == None and len(self.landmark_px["right_hand"]) != 0:  #if there has been a previous comparison point set, and less than a second has passed since that time. No landmarks found
            if not("previous_elapsed" in self.etc["timers"]["right_hand"] and self.etc["timers"]["right_hand"]["previous_elapsed"] > self.etc["max_wait_time"]):
                for index, (key, value) in enumerate(self.landmark_px["right_hand"].items()):
                    for line in lines:
                        cv2.line(frame, value[line[0]],value[line[1]],(0,255,0),3)

                    for val in value:
                        cv2.circle(frame, [val[0], val[1]], 6, np.array(self.etc["colormap"]["cmap"](index*self.etc["colormap"]["cmap_spacing"]))*255, 6)
        elif right_hand_landmarks is not None: #landmarks found
            image_rows, image_cols, _ = frame.shape
            for index, (key, value) in enumerate(self.gesture_point_dict["right_hand"].items()):
                value = [self._normalized_to_pixel_coordinates(right_hand_landmarks.landmark[val].x,  right_hand_landmarks.landmark[val].y, image_cols, image_rows) for val in value]
                self.landmark_px["right_hand"][key] = value
                for line in lines:
                    cv2.line(frame, value[line[0]],value[line[1]],(0,255,0),3)

                for val in value:
                    cv2.circle(frame, [val[0], val[1]], 6, np.array(self.etc["colormap"]["cmap"](index*self.etc["colormap"]["cmap_spacing"]))*255, 6)
            
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

    def draw_face_proprietary(self,frame, landmark_list, individual_show = False): #draw face using gesture points from .json 
        return frame
    
    def draw_pose_proprietary(self, frame: np.array, pose_landmark_list, individual_show = False): #draw points for pose using .json file 
        return frame
    def make_boxes(self,path = None):
        if path is None:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(path)
        cap,_,_ = noduro.get_maximum_resolution(cap)
        frame = cap.read()[1]
        oldshape = frame.shape
        image,w,h = noduro.scale_image_to_window(frame)
        oldshape = oldshape[0]/image.shape[0]
        boxes = []
        while True:
            a = cv2.selectROI("selector", image, showCrosshair = True)
            cv2.rectangle(image, (int(a[0]), int(a[1])), (int(a[0]+a[2]), int(a[1]+a[3])), (200,200,200), -1)
            boxes.append((a[0]*oldshape,a[1]*oldshape,a[2]*oldshape,a[3]*oldshape))
            cv2.destroyAllWindows()
            inp = input("do you want to make another rectangle")
            if noduro.check_boolean_input(inp):
                continue
            else:
                break
        self.boxes = boxes
        return boxes
if __name__ == '__main__':
    a = blur(frameskip = True)
    path = "C:/Users/aadvi/Desktop/IMG_1004.mov"
    a.make_boxes()
    # a.realtime_analysis(0)
    a.realtime_analysis()
